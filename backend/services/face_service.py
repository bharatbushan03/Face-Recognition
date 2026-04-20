import numpy as np
import logging
from typing import Tuple
import cv2

from backend.utils.error_handlers import FaceNotFoundError, MultipleFacesError, ImageProcessError

logger = logging.getLogger(__name__)


def _get_face_recognition_module():
    try:
        import face_recognition
    except ImportError as exc:
        raise ImportError(
            "face_recognition is not installed. Run: python -m pip install face_recognition"
        ) from exc
    return face_recognition


def _enhance_if_low_light(image_rgb: np.ndarray, brightness_threshold: float = 70.0) -> np.ndarray:
    """Apply CLAHE on luminance channel when the frame is too dark."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(np.mean(gray))
    if brightness >= brightness_threshold:
        return image_rgb

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

def extract_encoding(
    image_rgb: np.ndarray,
    enforce_single_face: bool = True,
    auto_enhance_low_light: bool = True,
) -> np.ndarray:
    """
    Extract perfectly one face encoding from an image.
    Throws FaceNotFoundError if no face is found.
    Throws MultipleFacesError if >1 face is found and enforce_single_face=True.
    """
    face_recognition = _get_face_recognition_module()
    image_for_detection = _enhance_if_low_light(image_rgb) if auto_enhance_low_light else image_rgb
    face_locations = face_recognition.face_locations(image_for_detection)

    if not face_locations and auto_enhance_low_light:
        # Fallback to original frame in case enhancement over-corrected the image.
        image_for_detection = image_rgb
        face_locations = face_recognition.face_locations(image_for_detection)

    num_faces = len(face_locations)
    
    if num_faces == 0:
        logger.warning("No face found in image.")
        raise FaceNotFoundError()
        
    if num_faces > 1 and enforce_single_face:
        logger.warning(f"Multiple faces ({num_faces}) found. Expecting exactly one.")
        raise MultipleFacesError()

    # Get face encodings for the face locations
    encodings = face_recognition.face_encodings(image_for_detection, known_face_locations=face_locations)
    
    if not encodings:
        raise ImageProcessError("Failed to extract face encodings from the discovered face.")
        
    # Return the first (and supposedly only) and its bounding box?
    # This just returns the 128-d numpy array
    return encodings[0]

def extract_face_data(
    image_rgb: np.ndarray,
    enforce_single_face: bool = True,
    auto_enhance_low_light: bool = True,
) -> Tuple[np.ndarray, list, bool]:
    """
    Extracts encoding, bounding box location, and smile heuristic.
    """
    face_recognition = _get_face_recognition_module()
    image_for_detection = _enhance_if_low_light(image_rgb) if auto_enhance_low_light else image_rgb
    face_locations = face_recognition.face_locations(image_for_detection)

    if not face_locations and auto_enhance_low_light:
        image_for_detection = image_rgb
        face_locations = face_recognition.face_locations(image_for_detection)

    num_faces = len(face_locations)
    
    if num_faces == 0:
        raise FaceNotFoundError()
        
    if num_faces > 1 and enforce_single_face:
        raise MultipleFacesError()

    location = face_locations[0] # (top, right, bottom, left)
    encodings = face_recognition.face_encodings(image_for_detection, known_face_locations=face_locations)
    if not encodings:
        raise ImageProcessError("Failed to extract face encodings from the discovered face.")
    encoding = encodings[0]
    
    # Smile detection using landmarks
    landmarks = face_recognition.face_landmarks(image_for_detection, face_locations)
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
            
    return encoding, list(location), is_smiling

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

    known_matrix = np.asarray(known_encodings, dtype=np.float32)
    query_vector = np.asarray(unknown_encoding, dtype=np.float32)
    if known_matrix.ndim != 2 or known_matrix.shape[1] != 128 or query_vector.shape != (128,):
        logger.warning("Unexpected encoding shape while comparing faces")
        return -1, 0.0
        
    # Vectorized L2 distances: lower distance means better match.
    distances = np.linalg.norm(known_matrix - query_vector, axis=1)
    best_match_index = np.argmin(distances)
    min_distance = distances[best_match_index]
    
    # Distance is 0 (identical) to 1.0 (completely opposite)
    # Convert distance to confidence score: 1.0 - distance
    confidence = max(0.0, 1.0 - min_distance)
    
    if min_distance <= tolerance:
        return int(best_match_index), confidence
    else:
        return -1, confidence
