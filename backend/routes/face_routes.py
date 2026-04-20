from __future__ import annotations

from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List
import logging
import os

from backend.core.security import require_admin_auth
from backend.models.database import get_db
from backend.models.schemas import RegisterResponse, RecognizeResponse, UserResponse
from backend.services.user_service import create_user, get_all_users
from backend.services.face_service import extract_encoding, extract_face_data, compare_faces
from backend.utils.image_processing import process_base64_image, process_upload_file
from backend.utils.error_handlers import ImageProcessError, AppException

router = APIRouter(prefix="/api/face", tags=["Face Recognition"])
logger = logging.getLogger(__name__)


def _recognition_tolerance_from_env() -> float:
    raw = os.getenv("FR_RECOGNITION_TOLERANCE", "0.5")
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid FR_RECOGNITION_TOLERANCE=%s. Falling back to 0.5", raw)
        return 0.5

    if value <= 0.0 or value > 1.0:
        logger.warning("Out-of-range FR_RECOGNITION_TOLERANCE=%s. Falling back to 0.5", raw)
        return 0.5
    return value

@router.post("/register", response_model=RegisterResponse)
async def register_face(
    image: UploadFile = File(None),
    image_base64: str = Form(None),
    name: str = Form(...),
    db: Session = Depends(get_db),
    _admin_auth: None = Depends(require_admin_auth),
):
    """
    Registers a new user by analyzing an image for their face encoding.
    Accepts EITHER an image file upload OR a base64 string.
    """
    if not image and not image_base64:
        raise AppException("Missing image data. Provide an image file or base64 string.")

    clean_name = name.strip()
    if not clean_name:
        raise AppException("Name cannot be empty.")

    if image:
        contents = await image.read()
        if not contents:
            raise AppException("Uploaded image is empty.")
        image_rgb = process_upload_file(contents)
    else:
        image_rgb = process_base64_image(image_base64)
        
    # Extract encoding (throws exceptions if no face or multiple faces)
    encoding = extract_encoding(image_rgb, enforce_single_face=True)
    
    # Save to db
    db_user = create_user(db, name=clean_name, face_encoding=encoding)
    logger.info("Registered new user: %s", db_user.name)
    
    return RegisterResponse(
        message=f"Successfully registered user: {clean_name}",
        user=db_user
    )

@router.post("/recognize", response_model=RecognizeResponse)
async def recognize_face(
    image: UploadFile = File(None),
    image_base64: str = Form(None),
    db: Session = Depends(get_db)
):
    """
    Attempts to recognize the face in the provided image.
    Accepts EITHER an image file upload OR a base64 string.
    """
    if not image and not image_base64:
        raise AppException("Missing image data. Provide an image file or base64 string.")

    if image:
        contents = await image.read()
        if not contents:
            raise AppException("Uploaded image is empty.")
        image_rgb = process_upload_file(contents)
    else:
        image_rgb = process_base64_image(image_base64)
    
    # Find the face encoding and extra data to query
    encoding, box, is_smiling = extract_face_data(image_rgb, enforce_single_face=True)
    
    # Fetch all known users
    all_users = get_all_users(db)
    
    if not all_users:
        return RecognizeResponse(
            message="No registered users found in the database.",
            match_found=False
        )
        
    # Prepare list of known encodings for the comparison library
    known_encodings = [user.get_encoding() for user in all_users]
    
    # Compare using a configurable tolerance.
    tolerance = _recognition_tolerance_from_env()
    match_index, confidence = compare_faces(encoding, known_encodings, tolerance=tolerance)
    
    if match_index != -1:
        matched_user = all_users[match_index]
        return RecognizeResponse(
            message="Face recognized.",
            match_found=True,
            user=matched_user,
            confidence=confidence,
            box=box,
            is_smiling=is_smiling
        )
    else:
        return RecognizeResponse(
            message="Unknown face.",
            match_found=False,
            confidence=confidence,
            box=box,
            is_smiling=is_smiling
        )

@router.get("/users", response_model=List[UserResponse])
def list_users(
    db: Session = Depends(get_db),
    _admin_auth: None = Depends(require_admin_auth),
):
    """List all registered users"""
    return get_all_users(db)
