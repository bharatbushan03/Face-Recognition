import face_recognition
import numpy as np
import logging
from typing import Tuple

from backend.utils.error_handlers import FaceNotFoundError, MultipleFacesError, ImageProcessError

logger = logging.getLogger(__name__)

def extract_encoding(image_rgb: np.ndarray, enforce_single_face: bool = True) -> np.ndarray:
    """
    Extract perfectly one face encoding from an image.
    Throws FaceNotFoundError if no face is found.
    Throws MultipleFacesError if >1 face is found and enforce_single_face=True.
    """
    face_locations = face_recognition.face_locations(image_rgb)
    num_faces = len(face_locations)
    
    if num_faces == 0:
        logger.warning("No face found in image.")
        raise FaceNotFoundError()
        
    if num_faces > 1 and enforce_single_face:
        logger.warning(f"Multiple faces ({num_faces}) found. Expecting exactly one.")
        raise MultipleFacesError()

    # Get face encodings for the face locations
    encodings = face_recognition.face_encodings(image_rgb, known_face_locations=face_locations)
    
    if not encodings:
        raise ImageProcessError("Failed to extract face encodings from the discovered face.")
        
    # Return the first (and supposedly only) and its bounding box?
    # This just returns the 128-d numpy array
    return encodings[0]

def extract_face_data(image_rgb: np.ndarray, enforce_single_face: bool = True) -> Tuple[np.ndarray, list, bool]:
    """
    Extracts encoding, bounding box location, and smile heuristic.
    """
    face_locations = face_recognition.face_locations(image_rgb)
    num_faces = len(face_locations)
    
    if num_faces == 0:
        raise FaceNotFoundError()
        
    if num_faces > 1 and enforce_single_face:
        raise MultipleFacesError()

    location = face_locations[0] # (top, right, bottom, left)
    encodings = face_recognition.face_encodings(image_rgb, known_face_locations=face_locations)[0]
    
    # Smile detection using landmarks
    landmarks = face_recognition.face_landmarks(image_rgb, face_locations)
    is_smiling = False
    
    if landmarks and len(landmarks) > 0:
        face_marks = landmarks[0]
        if 'top_lip' in face_marks and 'bottom_lip' in face_marks:
            # corners of the mouth are usually the first and 7th points of the top_lip
            left_corner = face_marks['top_lip'][0]
            right_corner = face_marks['top_lip'][6]
            
            mouth_width = ((right_corner[0] - left_corner[0]) ** 2 + (right_corner[1] - left_corner[1]) ** 2) ** 0.5
            
            top, right, bottom, left = location
            face_width = max(right - left, 1) # prevent div by zero
            
            # Simple heuristic: if mouth width is more than 40% of the bounding face box width, they might be smiling
            ratio = mouth_width / face_width
            is_smiling = ratio > 0.38
            
    return encodings, list(location), is_smiling

def compare_faces(
    unknown_encoding: np.ndarray, 
    known_encodings: list[np.ndarray], 
    tolerance: float = 0.5
) -> Tuple[int, float]:
    """
    Compares unknown_encoding against a list of known_encodings.
    Returns:
       best_match_index (int or -1 if no match)
       confidence (float 0.0 to 1.0)
    """
    if not known_encodings:
        return -1, 0.0
        
    # Calculate face distances
    distances = face_recognition.face_distance(known_encodings, unknown_encoding)
    best_match_index = np.argmin(distances)
    min_distance = distances[best_match_index]
    
    # Distance is 0 (identical) to 1.0 (completely opposite)
    # Convert distance to confidence score: 1.0 - distance
    confidence = max(0.0, 1.0 - min_distance)
    
    if min_distance <= tolerance:
        return int(best_match_index), confidence
    else:
        return -1, confidence
