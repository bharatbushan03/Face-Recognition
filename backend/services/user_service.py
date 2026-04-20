from sqlalchemy.orm import Session
from typing import List
import numpy as np
import logging
from backend.models.user import User


logger = logging.getLogger(__name__)

def create_user(db: Session, name: str, face_encoding: np.ndarray) -> User:
    """Create a new user with their face encoding"""
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("User name cannot be empty")

    db_user = User(name=clean_name)
    db_user.set_encoding(face_encoding)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    logger.info("User created: id=%s name=%s", db_user.id, db_user.name)
    return db_user

def get_all_users(db: Session) -> List[User]:
    """Retrieve all users"""
    return db.query(User).all()

def delete_user(db: Session, user_id: int):
    """Delete a user by ID"""
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        db.delete(user)
        db.commit()
        logger.info("User deleted: id=%s name=%s", user.id, user.name)
