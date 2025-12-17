"""
Chat session synchronization utilities
Keeps chat_sessions and llamaindex_chat_store in sync
"""
import json
import datetime
import logging
from typing import Optional, List, Any, Dict
from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger(__name__)

async def sync_chat_session(
    db: Session,
    user_id: str,
    chat_id: str,
    user_message: str,
    ai_response: str,
    attachment: Optional[Dict[str, Any]] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
    tone: Optional[str] = None
):
    """
    Synchronize chat_sessions table with LlamaIndex conversation memory
    
    Args:
        db: Database session
        user_id: User identifier
        chat_id: Chat session identifier
        user_message: User's message
        ai_response: AI's response
        attachment: Optional attachment metadata
        sources: Optional source citations
        tone: Optional tone used
    """
    try:
        # Check if session exists
        result = db.execute(
            text("SELECT messages FROM chat_sessions WHERE id = :chat_id AND user_id = :user_id"),
            {"chat_id": chat_id, "user_id": user_id}
        )
        row = result.fetchone()
        
        # Prepare new messages
        timestamp = datetime.datetime.now().isoformat()
        
        user_msg: Dict[str, Any] = {
            "role": "user",
            "content": user_message,
            "timestamp": timestamp
        }
        if attachment:
            user_msg["attachment"] = attachment
        if tone:
            user_msg["tone"] = tone
        
        ai_msg: Dict[str, Any] = {
            "role": "model",
            "content": ai_response,
            "timestamp": timestamp
        }
        if sources:
            ai_msg["sources"] = sources
        
        if row:
            # Update existing session
            existing_messages = row[0] if row[0] else []
            existing_messages.extend([user_msg, ai_msg])
            
            db.execute(
                text("""
                    UPDATE chat_sessions 
                    SET messages = :messages, updated_at = now()
                    WHERE id = :chat_id AND user_id = :user_id
                """),
                {
                    "messages": json.dumps(existing_messages),
                    "chat_id": chat_id,
                    "user_id": user_id
                }
            )
        else:
            # Create new session
            db.execute(
                text("""
                    INSERT INTO chat_sessions (id, user_id, title, messages)
                    VALUES (:chat_id, :user_id, :title, :messages)
                """),
                {
                    "chat_id": chat_id,
                    "user_id": user_id,
                    "title": user_message[:50] + "..." if len(user_message) > 50 else user_message,
                    "messages": json.dumps([user_msg, ai_msg])
                }
            )
        
        db.commit()
        logger.info(f"Chat session {chat_id} synchronized to database")
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error synchronizing chat session: {e}")
        raise
