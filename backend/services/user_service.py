from sqlalchemy.orm import Session
from typing import List, Optional
import numpy as np
from backend.models.user import User

def create_user(db: Session, name: str, face_encoding: np.ndarray) -> User:
    """Create a new user with their face encoding"""
    db_user = User(name=name)
    db_user.set_encoding(face_encoding)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
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
