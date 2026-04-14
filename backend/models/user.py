import datetime
import json
import numpy as np

from sqlalchemy import Column, Integer, String, Text, DateTime
from backend.models.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), index=True, nullable=False)
    # Storing 128-d face encoding as a JSON string
    face_encoding = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    def set_encoding(self, encoding: np.ndarray):
        """Convert numpy array to JSON string for storage"""
        self.face_encoding = json.dumps(encoding.tolist())

    def get_encoding(self) -> np.ndarray:
        """Parse JSON string back to numpy array"""
        return np.array(json.loads(self.face_encoding))
