import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import numpy as np

from backend.app import app
from backend.models.database import Base, get_db
from backend.models.user import User  # Must import to register with Base
from backend.services.face_service import extract_encoding, compare_faces

# Use an in-memory SQLite database for testing

SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(scope="module")
def setup_database():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "API is running"}

def test_face_service_compare():
    # Dummy encodings
    enc1 = np.ones(128) * 0.1
    enc2 = np.ones(128) * 0.12 # Closer to enc1. Diff is 0.02. Dist = sqrt(128 * 0.0004) = sqrt(0.0512) = 0.226 < 0.5
    enc3 = np.ones(128) * 0.9  # Far from enc1
    
    # Test match
    match_index, confidence = compare_faces(enc1, [enc3, enc2], tolerance=0.5)
    assert match_index == 1 # Matches enc2
    assert confidence > 0.5
    
    # Test no match
    match_index, confidence = compare_faces(enc3, [enc1, enc2], tolerance=0.5)
    assert match_index == -1
    assert confidence < 0.5

def test_missing_image_payload(setup_database):
    response = client.post("/api/face/register", data={"name": "Test User"})
    assert response.status_code == 400
    assert response.json()["message"] == "Missing image data. Provide an image file or base64 string."

def test_get_users(setup_database):
    response = client.get("/api/face/users")
    assert response.status_code == 200
    assert response.json() == []

# Notice: We can't easily unit-test the register/recognize endpoints fully without
# valid base64 face images or mocking. In a real scenario, we'd add base64 fixtures or mock `extract_encoding`.
