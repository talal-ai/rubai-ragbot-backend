from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Tuple, Dict, Optional
import io
import logging
import base64
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def process_base64_attachment(base64_data: str, filename: str) -> List[Tuple[str, int, Optional[int]]]:
    """
    Process a Base64 encoded attachment into chunks
    
    Args:
        base64_data: Base64 encoded file data (may include data URI prefix)
        filename: Name of the file
        
    Returns:
        List of tuples (chunk_text, chunk_index, page_number)
    """
    try:
        # Remove data URI prefix if present (e.g., "data:application/pdf;base64,")
        if ',' in base64_data and base64_data.startswith('data:'):
            base64_data = base64_data.split(',', 1)[1]
        
        # Decode Base64 to bytes
        file_bytes = base64.b64decode(base64_data)
        
        # Determine file type from filename
        file_type = "pdf" if filename.lower().endswith('.pdf') else "txt"
        
        # Process using existing pipeline
        logger.info(f"Processing Base64 attachment: {filename}")
        return process_document(file_bytes, filename, file_type)
        
    except Exception as e:
        logger.error(f"Error processing Base64 attachment {filename}: {e}")
        raise RuntimeError(f"Failed to process attachment: {str(e)}")


def extract_text_from_pdf_with_pages(file_bytes: bytes) -> List[Dict[str, any]]:
    """
    Extract text from PDF file bytes with page number tracking
    
    Args:
        file_bytes: PDF file as bytes
        
    Returns:
        List of dicts with 'text' and 'page_number' for each page
    """
    try:
        pdf_file = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_file)
        
        if not reader.pages:
            raise ValueError("PDF has no pages")
        
        pages = []
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    pages.append({
                        "text": page_text.strip(),
                        "page_number": page_num
                    })
            except Exception as e:
                logger.warning(f"Could not extract text from page {page_num}: {e}")
                continue
        
        if not pages:
            raise ValueError("No text content found in PDF")
        
        logger.info(f"Extracted text from {len(pages)} pages")
        return pages
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        raise RuntimeError(f"Failed to extract PDF content: {str(e)}")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from PDF file bytes (legacy function for backwards compatibility)
    
    Args:
        file_bytes: PDF file as bytes
        
    Returns:
        Extracted text content
    """
    pages = extract_text_from_pdf_with_pages(file_bytes)
    return "\n".join([p["text"] for p in pages])

def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
    """
    Split text into chunks using RecursiveCharacterTextSplitter
    
    Args:
        text: The text to split
        chunk_size: Maximum size of each chunk (default from config)
        chunk_overlap: Number of characters to overlap between chunks (default from config)
    
    Returns:
        List of text chunks
    """
    try:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
            
        chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
        chunk_overlap = chunk_overlap or DEFAULT_CHUNK_OVERLAP
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        
        # Filter out empty chunks
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        if not chunks:
            raise ValueError("No chunks generated from text")
            
        logger.info(f"Generated {len(chunks)} chunks from text")
        return chunks
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        raise RuntimeError(f"Failed to chunk text: {str(e)}")


def chunk_text_with_pages(pages: List[Dict[str, any]], chunk_size: int = None, chunk_overlap: int = None) -> List[Dict[str, any]]:
    """
    Split text into chunks while preserving page number information
    
    Args:
        pages: List of dicts with 'text' and 'page_number'
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap
    
    Returns:
        List of dicts with 'text', 'page_number', and 'chunk_index'
    """
    try:
        chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
        chunk_overlap = chunk_overlap or DEFAULT_CHUNK_OVERLAP
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        all_chunks = []
        chunk_index = 0
        
        for page_data in pages:
            page_text = page_data["text"]
            page_number = page_data["page_number"]
            
            # Split this page's text into chunks
            page_chunks = text_splitter.split_text(page_text)
            
            for chunk in page_chunks:
                chunk = chunk.strip()
                if chunk:
                    all_chunks.append({
                        "text": chunk,
                        "page_number": page_number,
                        "chunk_index": chunk_index
                    })
                    chunk_index += 1
        
        if not all_chunks:
            raise ValueError("No chunks generated from pages")
        
        logger.info(f"Generated {len(all_chunks)} chunks from {len(pages)} pages")
        return all_chunks
        
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error chunking pages: {e}")
        raise RuntimeError(f"Failed to chunk pages: {str(e)}")

def process_document(file_bytes: bytes, filename: str, file_type: str = "pdf") -> List[Tuple[str, int, Optional[int]]]:
    """
    Process a document: extract text and chunk it with page tracking
    
    Args:
        file_bytes: Document file as bytes
        filename: Name of the file
        file_type: Type of file (pdf or txt)
        
    Returns:
        List of tuples (chunk_text, chunk_index, page_number)
        page_number is None for text files
    """
    try:
        logger.info(f"Processing document: {filename} ({file_type})")
        
        # Extract text based on file type
        if file_type.lower() == "pdf":
            # Use page-aware extraction for PDFs
            pages = extract_text_from_pdf_with_pages(file_bytes)
            
            # Chunk while preserving page info
            chunks_with_pages = chunk_text_with_pages(pages)
            
            # Return chunks with indices and page numbers
            result = [
                (chunk["text"], chunk["chunk_index"], chunk["page_number"]) 
                for chunk in chunks_with_pages
            ]
            logger.info(f"Document processed successfully: {len(result)} chunks created from {len(pages)} pages")
            return result
            
        elif file_type.lower() in ["txt", "text"]:
            try:
                text = file_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # Try with different encoding
                text = file_bytes.decode('latin-1')
            
            if not text.strip():
                raise ValueError("Text file is empty")
            
            # Chunk the text (no page numbers for txt files)
            chunks = chunk_text(text)
            
            # Return chunks with their indices, page_number=None for text files
            result = [(chunk, idx, None) for idx, chunk in enumerate(chunks)]
            logger.info(f"Document processed successfully: {len(result)} chunks created")
            return result
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error processing document {filename}: {e}")
        raise RuntimeError(f"Failed to process document: {str(e)}")
