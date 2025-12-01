"""
Professional RAG service using LlamaIndex with conversation memory
Handles document retrieval, chat memory, and response generation
"""
from llama_index.core import VectorStoreIndex, Document as LlamaDocument, StorageContext, Settings
from llama_index.core.memory import ChatMemoryBuffer  
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.storage.chat_store.postgres import PostgresChatStore
from typing import List, Dict, Optional, Generator
import logging
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import Document as DBDocument
from llamaindex_config import init_llamaindex, EMBEDDING_DIM, CHAT_MEMORY_TOKEN_LIMIT, DEFAULT_TOP_K
from vector_store import search_similar_chunks  # Import our direct search function
import os

logger = logging.getLogger(__name__)

# Initialize LlamaIndex on module import
init_llamaindex()

class RAGService:
    """
    Professional RAG service with conversation memory using LlamaIndex
    
    Features:
    - Persistent conversation memory in PostgreSQL
    - Document-aware context retrieval
    - User and chat session isolation
    - Streaming and non-streaming responses
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize RAG service with database connection
        
        Args:
            db_session: SQLAlchemy database session
        """
        self.db = db_session
        
        # Parse DATABASE_URL for PGVectorStore
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL not found in environment variables")
        
        # Initialize PostgreSQL vector store for LlamaIndex
        # Parse connection details from DATABASE_URL
        logger.info("Initializing PostgreSQL vector store...")
        
        # Parse DATABASE_URL properly with urllib
        from urllib.parse import urlparse
        parsed = urlparse(db_url)
        
        self.vector_store = PGVectorStore.from_params(
            database=parsed.path.lstrip('/'),  # Remove leading /
            host=parsed.hostname,
            password=parsed.password,
            port=parsed.port or 5432,
            user=parsed.username,
            table_name="documents",
            embed_dim=EMBEDDING_DIM,  # 768 for Gemini
            hybrid_search=False,  # Set to True later for hybrid search
            text_search_config="english"
        )
        logger.info("✅ Vector store initialized")
        
        # Initialize PostgreSQL chat store for persistent memory
        logger.info("Initializing PostgreSQL chat store...")
        self.chat_store = PostgresChatStore.from_uri(
            uri=db_url,
            table_name="llamaindex_chat_store"  # Separate table for LlamaIndex
        )
        logger.info("✅ Chat store initialized")
    
    def create_chat_engine(
        self, 
        user_id: str,
        chat_id: str,
        selected_documents: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Create a chat engine with conversation memory
        
        Args:
            user_id: User identifier
            chat_id: Unique chat session identifier
            selected_documents: Optional list of document filenames to filter
            system_prompt: Optional custom system prompt
        
        Returns:
            LlamaIndex chat engine with memory
        """
        logger.info(f"Creating chat engine for user={user_id}, chat={chat_id}")
        
        # Create index from vector store
        index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store
        )
        
        # Create memory with chat store
        # Key format: "user_{user_id}_chat_{chat_id}"
        chat_store_key = f"user_{user_id}_chat_{chat_id}"
        
        memory = ChatMemoryBuffer.from_defaults(
            token_limit=CHAT_MEMORY_TOKEN_LIMIT,  # Keep last ~3000 tokens
            chat_store=self.chat_store,
            chat_store_key=chat_store_key
        )
        logger.info(f"Memory initialized with key: {chat_store_key}")
        
        # Build retriever kwargs with user filtering
        # PHASE 4: Smart document filtering with post-processing
        # Filter by user_id first, then post-filter for:
        # - Knowledge base docs (upload_type='knowledge_base')
        # - This chat's attachments (upload_type='attachment' AND chat_id=this_chat)
        from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
        
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="user_id",
                    value=user_id,
                    operator=FilterOperator.EQ
                )
            ]
        )
        
        # Store chat_id for post-filtering
        retriever_kwargs = {
            "similarity_top_k": DEFAULT_TOP_K * 2,  # Get more to account for filtering
            "filters": filters  # Filter by user
        }
        
        logger.info(f"Retriever configured: user={user_id}, post-filter for knowledge_base + chat={chat_id} attachments")
        
        # Create custom node post-processor for Phase 4 isolation
        from llama_index.core.postprocessor.types import BaseNodePostprocessor
        from llama_index.core.schema import NodeWithScore, QueryBundle
        from typing import List as TypingList
        
        class ChatIsolationPostprocessor(BaseNodePostprocessor):
            """Filter nodes to only include knowledge base + this chat's attachments"""
            
            _chat_id: str  # Pydantic private attribute
            
            def __init__(self, chat_id: str, **kwargs):
                super().__init__(**kwargs)
                object.__setattr__(self, '_chat_id', chat_id)
            
            def _postprocess_nodes(
                self, nodes: TypingList[NodeWithScore], query_bundle: QueryBundle = None
            ) -> TypingList[NodeWithScore]:
                """Filter nodes based on upload_type and chat_id"""
                filtered = []
                for node in nodes:
                    metadata = node.node.metadata
                    upload_type = metadata.get("upload_type", "")
                    node_chat_id = metadata.get("chat_id", "")
                    
                    # Keep if: knowledge_base OR this chat's attachment
                    if upload_type == "knowledge_base" or (upload_type == "attachment" and node_chat_id == self._chat_id):
                        filtered.append(node)
                    else:
                        logger.debug(f"Filtered out: {metadata.get('filename')} (upload_type={upload_type}, chat_id={node_chat_id})")
                
                # If no documents match, fallback to all user documents (old behavior)
                # This prevents empty responses when user has docs but none in KB or this chat
                if len(filtered) == 0 and len(nodes) > 0:
                    logger.warning(f"No KB or chat-specific docs found, using all {len(nodes)} user documents as fallback")
                    return nodes
                
                logger.info(f"Post-filter: {len(nodes)} -> {len(filtered)} nodes (kb + chat {self._chat_id[:8]})")
                return filtered
        
        
        node_postprocessors = [ChatIsolationPostprocessor(chat_id)]
        
        # Check if vector store has any documents
        # Use condense_plus_context for RAG when documents exist
        try:
            # Try to create retriever to test if docs exist
            retriever = index.as_retriever(**retriever_kwargs)
            
            # Use RAG mode with retrieval and post-processing
            chat_engine = index.as_chat_engine(
                chat_mode="condense_plus_context",
                memory=memory,
                system_prompt=system_prompt or self._get_default_system_prompt(),
                verbose=True,
                node_postprocessors=node_postprocessors,
                **retriever_kwargs
            )
            logger.info("✅ Chat engine created with RAG (condense_plus_context + isolation post-processor)")
            
        except Exception as e:
            # Fallback to simple mode if vector store is empty or has issues
            logger.warning(f"Could not create RAG engine, using simple mode: {e}")
            from llama_index.core.chat_engine import SimpleChatEngine
            
            chat_engine = SimpleChatEngine.from_defaults(
                llm=Settings.llm,
                memory=memory,
                system_prompt=system_prompt or self._get_default_system_prompt()
            )
            logger.info("✅ Chat engine created in simple mode (no RAG)")
        
        return chat_engine
    
    def chat(
        self,
        user_id: str,
        chat_id: str,
        message: str,
        selected_documents: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ) -> tuple:
        """
        Chat with RAG and conversation memory (non-streaming)
        
        Args:
            user_id: User identifier  
            chat_id: Chat session identifier
            message: User message
            selected_documents: Optional list of documents to search within
            system_prompt: Optional custom system prompt
        
        Returns:
            tuple: (response_text, sources_list)
            - response_text: str - Bot response
            - sources_list: list - Source nodes with metadata [{filename, category, page, chunk_index, storage_url}, ...]
        """
        logger.info(f"Processing chat message: '{message[:50]}...'")
        
        chat_engine = self.create_chat_engine(
            user_id=user_id,
            chat_id=chat_id,
            selected_documents=selected_documents,
            system_prompt=system_prompt
        )
        
        response = chat_engine.chat(message)
        
        logger.info(f"Response generated: {len(str(response))} characters")
        
        # **PHASE 4**: Extract source nodes from response and enrich with storage_url
        sources = []
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for source_node in response.source_nodes:
                try:
                    # Extract metadata from source node
                    metadata = source_node.metadata if hasattr(source_node, 'metadata') else {}
                    source_info = {
                        "filename": metadata.get('filename', source_node.node_id if hasattr(source_node, 'node_id') else 'unknown'),
                        "page_number": metadata.get('page_number'),
                        "chunk_index": metadata.get('chunk_index', 0),
                        "category": metadata.get('category'),
                        "storage_url": metadata.get('storage_url')
                    }
                    
                    # PHASE 4 ENHANCED: Comprehensive storage_url enrichment with fallback
                    if not source_info.get("storage_url") and source_info.get("filename"):
                        try:
                            result = self.db.execute(
                                text("SELECT storage_url, metadata FROM documents WHERE filename = :filename LIMIT 1"),
                                {"filename": source_info["filename"]}
                            ).first()
                            if result:
                                # Try column first
                                source_info["storage_url"] = result.storage_url
                                
                                # Fallback to metadata fields
                                if not source_info["storage_url"] and result.metadata:
                                    meta = result.metadata if isinstance(result.metadata, dict) else {}
                                    source_info["storage_url"] = meta.get("storage_url") or meta.get("file_url")
                                    source_info["category"] = meta.get("category")
                                
                                # Last resort: construct URL from filename and category
                                if not source_info["storage_url"]:
                                    category = source_info.get("category") or "ai-docs"
                                    from urllib.parse import quote
                                    encoded_filename = quote(source_info["filename"])
                                    source_info["storage_url"] = f"{os.getenv('SUPABASE_URL')}/storage/v1/object/public/knowledge-base/{category}/{encoded_filename}"
                                    logger.info(f"Constructed fallback URL for {source_info['filename']}: {source_info['storage_url']}")
                                else:
                                    logger.debug(f"Enriched {source_info['filename']} with storage_url from DB")
                        except Exception as e:
                            logger.warning(f"Could not enrich source metadata: {e}")
                            # Try to construct URL anyway
                            try:
                                from urllib.parse import quote
                                encoded_filename = quote(source_info["filename"])
                                source_info["storage_url"] = f"{os.getenv('SUPABASE_URL')}/storage/v1/object/public/knowledge-base/ai-docs/{encoded_filename}"
                                logger.info(f"Using emergency fallback URL for {source_info['filename']}")
                            except:
                                pass
                    
                    sources.append(source_info)
                    logger.debug(f"Extracted source: {source_info['filename']}")
                except Exception as e:
                    logger.warning(f"Error extracting source node metadata: {e}")
                    continue
        
        logger.info(f"Extracted {len(sources)} sources from response with storage URLs")
        return str(response), sources
    
    def chat_direct(
        self,
        user_id: Optional[str],
        chat_id: str,
        message: str,
        selected_documents: Optional[List[str]] = None,
        category: Optional[str] = None,
        knowledge_base_mode: str = "none",
        system_prompt: Optional[str] = None
    ) -> tuple:
        """
        Chat with direct retrieval from our documents table (bypasses LlamaIndex PGVectorStore)
        
        Knowledge Base Modes:
        - "none": AI is completely blind, no document access
        - "folder": Only search within specified category
        - "file": Only search within selected document filenames
        - "all": Search entire knowledge base
        
        Args:
            user_id: User identifier  
            chat_id: Chat session identifier
            message: User message
            selected_documents: Optional list of documents to search within
            category: Optional category filter (privacy-policies, cvs, terms-and-conditions, ai-docs)
            knowledge_base_mode: Knowledge base access mode
            system_prompt: Optional custom system prompt
        
        Returns:
            tuple: (response_text, sources_list)
        """
        logger.info(f"[DIRECT RAG] Processing message: '{message[:50]}...' (mode: {knowledge_base_mode})")
        
        # Handle different knowledge base modes
        if knowledge_base_mode == "none":
            # AI is completely blind - no document access
            relevant_chunks = []
            context = "No knowledge base access enabled. I can only respond based on my general knowledge."
        else:
            # Step 1: Retrieve relevant context using our direct search
            search_category = None
            search_filenames = None
            
            if knowledge_base_mode == "folder" and category:
                search_category = category
            elif knowledge_base_mode == "file" and selected_documents:
                search_filenames = selected_documents
            elif knowledge_base_mode == "all":
                # Search all documents (no filters)
                pass
            
            relevant_chunks = search_similar_chunks(
                db=self.db,
                query=message,
                top_k=DEFAULT_TOP_K,
                user_id=user_id,
                filenames=search_filenames,
                category=search_category
            )
            
            logger.info(f"[DIRECT RAG] Retrieved {len(relevant_chunks)} relevant chunks (mode: {knowledge_base_mode})")
            
            # Step 2: Build context from retrieved chunks
            if relevant_chunks:
                context_parts = []
                for i, chunk in enumerate(relevant_chunks):
                    context_parts.append(f"[Source {i+1}: {chunk['filename']}, Page {chunk.get('page_number', 'N/A')}, Chunk {chunk['chunk_index']}]\n{chunk['content']}")
                context = "\n\n---\n\n".join(context_parts)
            else:
                if knowledge_base_mode == "all":
                    context = "No relevant documents found in the knowledge base."
                elif knowledge_base_mode == "folder":
                    context = f"No relevant documents found in the {category} folder."
                elif knowledge_base_mode == "file":
                    context = f"No relevant documents found in the selected files."
                else:
                    context = "No relevant documents found."
        
        # Step 3: Build prompt with context
        base_system = system_prompt or self._get_default_system_prompt()
        full_prompt = f"""{base_system}

RETRIEVED DOCUMENT CONTEXT:
{context}

USER QUESTION: {message}

Please answer based on the context provided above. If citing specific information, 
reference the source number and page."""
        
        # Step 4: Generate response using LLM
        try:
            llm = Settings.llm
            response = llm.complete(full_prompt)
            response_text = str(response)
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            response_text = "I apologize, but I encountered an error generating a response."
        
        # Step 5: Build sources with full metadata including storage_url
        # PHASE 5: Conditional Citations - only include high-relevance sources
        HIGH_RELEVANCE_THRESHOLD = 0.4  # Only show sources with similarity > 40%
        
        # Use a dict to deduplicate by (filename, page_number) - keep highest similarity
        unique_sources = {}
        for chunk in relevant_chunks:
            similarity = chunk.get("similarity", 0)
            
            # Skip low-relevance sources (they likely aren't actually used by the LLM)
            if similarity < HIGH_RELEVANCE_THRESHOLD:
                logger.debug(f"Skipping low-relevance source: {chunk['filename']} (similarity={similarity:.2f})")
                continue
            
            # Create unique key for this page
            page_key = (chunk["filename"], chunk.get("page_number"))
            
            # Only keep if this is the first occurrence or has higher similarity
            if page_key not in unique_sources or similarity > unique_sources[page_key]["similarity"]:
                source = {
                    "filename": chunk["filename"],
                    "page_number": chunk.get("page_number"),
                    "chunk_index": chunk["chunk_index"],
                    "similarity": similarity,
                    "text_snippet": chunk["content"][:200] if chunk.get("content") else ""
                }
                unique_sources[page_key] = source
        
        # Convert back to list for processing
        sources = []
        for source in unique_sources.values():
            
            # Get storage_url and category from database with comprehensive fallback
            try:
                result = self.db.execute(
                    text("SELECT storage_url, metadata FROM documents WHERE filename = :filename LIMIT 1"),
                    {"filename": chunk["filename"]}
                ).first()
                if result:
                    # Try column first
                    source["storage_url"] = result.storage_url
                    
                    # Extract metadata
                    if result.metadata:
                        meta = result.metadata if isinstance(result.metadata, dict) else {}
                        # Fallback to metadata if column is null
                        if not source["storage_url"]:
                            source["storage_url"] = meta.get("storage_url") or meta.get("file_url")
                        source["category"] = meta.get("category")
                        source["upload_type"] = meta.get("upload_type")
                    
                    # Last resort: construct URL from filename and category
                    if not source["storage_url"]:
                        category = source.get("category") or "ai-docs"
                        from urllib.parse import quote
                        encoded_filename = quote(chunk["filename"])
                        source["storage_url"] = f"{os.getenv('SUPABASE_URL')}/storage/v1/object/public/knowledge-base/{category}/{encoded_filename}"
                        logger.info(f"Constructed fallback URL for {chunk['filename']}: {source['storage_url']}")
                    else:
                        logger.debug(f"Retrieved storage_url for {chunk['filename']}: {source['storage_url']}")
            except Exception as e:
                logger.warning(f"Could not enrich source: {e}")
                # Emergency fallback: construct URL with default category
                try:
                    from urllib.parse import quote
                    encoded_filename = quote(chunk["filename"])
                    source["storage_url"] = f"{os.getenv('SUPABASE_URL')}/storage/v1/object/public/knowledge-base/ai-docs/{encoded_filename}"
                    logger.info(f"Using emergency fallback URL for {chunk['filename']}")
                except:
                    pass
            
            sources.append(source)
        
        logger.info(f"[DIRECT RAG] Response: {len(response_text)} chars, {len(sources)} unique pages from {len(relevant_chunks)} chunks (threshold={HIGH_RELEVANCE_THRESHOLD})")
        return response_text, sources

    def chat_stream_direct(
        self,
        user_id: Optional[str],
        chat_id: str,
        message: str,
        selected_documents: Optional[List[str]] = None,
        category: Optional[str] = None,
        knowledge_base_mode: str = "none",
        system_prompt: Optional[str] = None
    ) -> Generator[Dict, None, None]:
        """
        Streaming chat with direct retrieval (bypasses LlamaIndex PGVectorStore)
        
        Knowledge Base Modes:
        - "none": AI is completely blind, no document access
        - "folder": Only search within specified category
        - "file": Only search within selected document filenames
        - "all": Search entire knowledge base
        
        Args:
            user_id: User identifier
            chat_id: Chat session identifier
            message: User message
            selected_documents: Optional list of documents to search within
            category: Optional category filter (privacy-policies, cvs, terms-and-conditions, ai-docs)
            knowledge_base_mode: Knowledge base access mode
            system_prompt: Optional custom system prompt
            system_prompt: Optional custom system prompt
        
        Yields:
            Dict: Response with 'token' and optional 'sources'
        """
        logger.info(f"[DIRECT RAG STREAM] Processing message: '{message[:50]}...' (mode: {knowledge_base_mode})")
        
        # Handle different knowledge base modes
        if knowledge_base_mode == "none":
            # AI is completely blind - no document access
            relevant_chunks = []
            context = "No knowledge base access enabled. I can only respond based on my general knowledge."
        else:
            # Step 1: Retrieve relevant context
            search_category = None
            search_filenames = None
            
            if knowledge_base_mode == "folder" and category:
                search_category = category
            elif knowledge_base_mode == "file" and selected_documents:
                search_filenames = selected_documents
            elif knowledge_base_mode == "all":
                # Search all documents (no filters)
                pass
            
            relevant_chunks = search_similar_chunks(
                db=self.db,
                query=message,
                top_k=DEFAULT_TOP_K,
                user_id=user_id,
                filenames=search_filenames,
                category=search_category
            )
            
            logger.info(f"[DIRECT RAG STREAM] Retrieved {len(relevant_chunks)} relevant chunks (mode: {knowledge_base_mode})")
            
            # Step 2: Build context
            if relevant_chunks:
                context_parts = []
                for i, chunk in enumerate(relevant_chunks):
                    context_parts.append(f"[Source {i+1}: {chunk['filename']}, Page {chunk.get('page_number', 'N/A')}]\n{chunk['content']}")
                context = "\n\n---\n\n".join(context_parts)
            else:
                if knowledge_base_mode == "all":
                    context = "No relevant documents found in the knowledge base."
                elif knowledge_base_mode == "folder":
                    context = f"No relevant documents found in the {category} folder."
                elif knowledge_base_mode == "file":
                    context = f"No relevant documents found in the selected files."
                else:
                    context = "No relevant documents found."
        
        # Step 3: Build prompt
        base_system = system_prompt or self._get_default_system_prompt()
        full_prompt = f"""{base_system}

RETRIEVED DOCUMENT CONTEXT:
{context}

USER QUESTION: {message}

Please answer based on the context provided above."""
        
        # Step 4: Stream response using LLM
        try:
            llm = Settings.llm
            streaming_response = llm.stream_complete(full_prompt)
            
            for chunk in streaming_response:
                yield {"token": chunk.delta}
                
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            yield {"token": "I apologize, but I encountered an error generating a response."}
        
        # Step 5: After streaming, yield sources
        # PHASE 5: Conditional Citations - only include high-relevance sources
        HIGH_RELEVANCE_THRESHOLD = 0.2  # Only show sources with similarity > 20%
        
        # Use a dict to deduplicate by (filename, page_number) - keep highest similarity
        unique_sources = {}
        for chunk in relevant_chunks:
            similarity = chunk.get("similarity", 0)
            
            # Skip low-relevance sources
            if similarity < HIGH_RELEVANCE_THRESHOLD:
                logger.debug(f"Skipping low-relevance source: {chunk['filename']} (similarity={similarity:.2f})")
                continue
            
            # Create unique key for this page
            page_key = (chunk["filename"], chunk.get("page_number"))
            
            # Only keep if this is the first occurrence or has higher similarity
            if page_key not in unique_sources or similarity > unique_sources[page_key]["similarity"]:
                source = {
                    "filename": chunk["filename"],
                    "page_number": chunk.get("page_number"),
                    "chunk_index": chunk["chunk_index"],
                    "similarity": similarity,
                    "text_snippet": chunk["content"][:200] if chunk.get("content") else ""
                }
                unique_sources[page_key] = source
        
        # Convert back to list for processing
        sources = []
        for source in unique_sources.values():
            
            # Enrich with storage_url
            try:
                result = self.db.execute(
                    text("SELECT storage_url, metadata FROM documents WHERE filename = :filename LIMIT 1"),
                    {"filename": chunk["filename"]}
                ).first()
                if result:
                    source["storage_url"] = result.storage_url
                    if result.metadata:
                        meta = result.metadata if isinstance(result.metadata, dict) else {}
                        if not source["storage_url"]:
                            source["storage_url"] = meta.get("storage_url")
                        source["category"] = meta.get("category")
            except Exception as e:
                logger.warning(f"Could not enrich source: {e}")
            
            sources.append(source)
        
        # Yield sources at the end (only if we have high-relevance ones)
        if sources:
            yield {"sources": sources}
        
        logger.info(f"[DIRECT RAG STREAM] Completed with {len(sources)} unique pages from {len(relevant_chunks)} chunks (threshold={HIGH_RELEVANCE_THRESHOLD})")

    def chat_stream(
        self,
        user_id: str,
        chat_id: str,
        message: str,
        selected_documents: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ) -> Generator[Dict, None, None]:
        """
        Chat with RAG and conversation memory (streaming)
        
        Args:
            user_id: User identifier
            chat_id: Chat session identifier
            message: User message
            selected_documents: Optional list of documents to search within
            system_prompt: Optional custom system prompt
        
        Yields:
            Dict: Response with 'token' and optional 'sources'
        """
        logger.info(f"Processing streaming chat message: '{message[:50]}...'")
        
        chat_engine = self.create_chat_engine(
            user_id=user_id,
            chat_id=chat_id,
            selected_documents=selected_documents,
            system_prompt=system_prompt
        )
        
        streaming_response = chat_engine.stream_chat(message)
        
        # Yield tokens as they're generated
        token_count = 0
        full_response = ""
        for token in streaming_response.response_gen:
            token_count += 1
            full_response += token
            logger.debug(f"Token {token_count}: '{token}'")
            yield {"token": token}
        
        # After streaming completes, extract and yield sources
        source_nodes = streaming_response.source_nodes if hasattr(streaming_response, 'source_nodes') else []
        
        # Response quality analysis
        has_sources = len(source_nodes) > 0
        confidence = "high" if has_sources else "low"
        source_based = has_sources
        
        logger.info(f"Response analysis: {{'has_sources': {has_sources}, 'confidence': '{confidence}', 'source_based': {source_based}}}")
        
        if source_nodes:
            sources = []
            for node in source_nodes:
                source = {
                    "filename": node.metadata.get("filename", "Unknown"),
                    "page_number": node.metadata.get("page_number"),
                    "chunk_index": node.metadata.get("chunk_index"),
                    "category": node.metadata.get("category"),
                    "storage_url": node.metadata.get("storage_url"),
                    "text_snippet": node.text[:200] if node.text else ""
                }
                
                # PHASE 4: If storage_url is missing, look it up from our documents table
                if not source.get("storage_url") and source.get("filename"):
                    try:
                        result = self.db.execute(
                            text("SELECT storage_url, metadata FROM documents WHERE filename = :filename LIMIT 1"),
                            {"filename": source["filename"]}
                        ).first()
                        if result:
                            source["storage_url"] = result.storage_url
                            # Also try to get from metadata if column is null
                            if not source["storage_url"] and result.metadata:
                                meta = result.metadata
                                source["storage_url"] = meta.get("storage_url") or meta.get("file_url")
                            if not source.get("category") and result.metadata:
                                source["category"] = result.metadata.get("category")
                            logger.debug(f"Enriched source {source['filename']} with storage_url from DB")
                    except Exception as e:
                        logger.warning(f"Could not enrich source metadata: {e}")
                
                sources.append(source)
            
            logger.info(f"Extracted {len(sources)} source citations with storage URLs")
            yield {"sources": sources}
        
        # Quality check: warn if response is suspiciously short
        if token_count < 5:
            logger.warning(f"Very short response detected: {token_count} tokens, text='{full_response[:100]}'")
        
        logger.info(f"Streaming response completed - {token_count} tokens sent, quality={confidence}")
    
    def get_chat_history(self, user_id: str, chat_id: str) -> List[Dict]:
        """
        Get conversation history for a chat session
        
        Args:
            user_id: User identifier
            chat_id: Chat session identifier
        
        Returns:
            List[Dict]: Chat history with role and content
        """
        chat_store_key = f"user_{user_id}_chat_{chat_id}"
        messages = self.chat_store.get_messages(chat_store_key)
        
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.additional_kwargs.get("timestamp")
            }
            for msg in messages
        ]
    
    def delete_chat_history(self, user_id: str, chat_id: str):
        """
        Delete conversation history for a chat session
        
        Args:
            user_id: User identifier
            chat_id: Chat session identifier
        """
        chat_store_key = f"user_{user_id}_chat_{chat_id}"
        self.chat_store.delete_messages(chat_store_key)
        logger.info(f"Deleted chat history for {chat_store_key}")
    
    def add_knowledge_base_document(self, chunks: List[tuple], filename: str, user_id: str):
        """
        Add permanent document to knowledge base (available to all chats)
        
        Args:
            chunks: List of tuples (chunk_text, chunk_index, page_number)
            filename: Name of the file
            user_id: User identifier for metadata
        """
        try:
            logger.info(f"Adding {len(chunks)} chunks from {filename} to KNOWLEDGE BASE")
            
            # Create LlamaIndex documents from chunks
            documents = []
            for chunk_text, chunk_index, page_number in chunks:
                # Create metadata - NO chat_id for knowledge base
                metadata = {
                    "filename": filename,
                    "chunk_index": chunk_index,
                    "user_id": user_id,
                    "upload_type": "knowledge_base"  # Permanent, available to all chats
                }
                
                if page_number is not None:
                    metadata["page_number"] = page_number
                
                # Create LlamaIndex document
                doc = LlamaDocument(
                    text=chunk_text,
                    metadata=metadata
                )
                documents.append(doc)
            
            # Add to vector store via index
            index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)
            
            # Insert documents
            for doc in documents:
                index.insert(doc)
            
            logger.info(f"✅ Successfully added {len(documents)} chunks to KNOWLEDGE BASE")
            
        except Exception as e:
            logger.error(f"Error adding knowledge base document: {e}")
            raise RuntimeError(f"Failed to add document to knowledge base: {str(e)}")
    
    def add_temporary_documents(self, chunks: List[tuple], filename: str, user_id: str, chat_id: str):
        """
        Add temporary document chunks to vector store for THIS CHAT ONLY
        
        Args:
            chunks: List of tuples (chunk_text, chunk_index, page_number)
            filename: Name of the file
            user_id: User identifier for metadata
            chat_id: Chat session identifier for metadata (ISOLATION)
        """
        try:
            logger.info(f"Adding {len(chunks)} chunks from {filename} to vector store")
            
            # Create LlamaIndex documents from chunks
            documents = []
            for chunk_text, chunk_index, page_number in chunks:
                # Create metadata
                # PHASE 4: Mark as attachment, bound to specific chat
                metadata = {
                    "filename": filename,
                    "chunk_index": chunk_index,
                    "user_id": user_id,
                    "chat_id": chat_id,  # Bound to this chat only
                    "upload_type": "attachment"  # Temporary, chat-specific
                }
                
                if page_number is not None:
                    metadata["page_number"] = page_number
                
                # Create LlamaIndex document
                doc = LlamaDocument(
                    text=chunk_text,
                    metadata=metadata
                )
                documents.append(doc)
            
            # Add to vector store via index
            index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)
            
            # Insert documents
            for doc in documents:
                index.insert(doc)
            
            logger.info(f"✅ Successfully added {len(documents)} document chunks to vector store")
            
        except Exception as e:
            logger.error(f"Error adding temporary documents: {e}")
            raise RuntimeError(f"Failed to add documents to vector store: {str(e)}")
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for RubAI"""
        return """You are RubAI, a sophisticated and determined AI solutions assistant.

IMPORTANT CONTEXT HANDLING:
- If document context is provided, use it to give accurate, sourced answers
- If NO document context is available, engage naturally in conversation
- For greetings like "Hi", "Hello", etc., respond warmly and introduce yourself
- Always provide a thoughtful, complete response - never return empty or minimal text

When answering WITH document context:
1. Use the retrieved information to provide accurate, well-sourced answers
2. Reference specific page numbers when citing information from documents
3. If the context doesn't contain the answer, clearly say so
4. Maintain conversation context from previous messages in this chat

When answering WITHOUT document context (no documents uploaded):
1. Engage conversationally and naturally - you can still be helpful!
2. For greetings, respond warmly and introduce your capabilities
3. Explain what you can help with: document analysis, answering questions from uploaded PDFs, etc.
4. Suggest uploading documents if the user has specific questions about content
5. Maintain conversation context from previous messages

CONVERSATION MEMORY: You have access to the full conversation history. Reference previous 
exchanges naturally with phrases like "As I mentioned earlier..." or "Building on our previous 
discussion..." When a user says "tell me more" or "elaborate", understand what they're referring 
to based on context.

Be professional, precise, helpful, and always engage - even without documents you can introduce 
yourself and explain how you can assist once documents are uploaded.
"""
