import base64
import cv2
import numpy as np
from backend.utils.error_handlers import ImageProcessError

def process_base64_image(base64_str: str) -> np.ndarray:
    """Converts a base64 string to a cv2 numpy array (RGB)"""
    try:
        # Some base64 strings come with 'data:image/jpeg;base64,' prefix 
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
            
        img_data = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ImageProcessError("Could not decode the image.")
            
        # face_recognition expects RGB. cv2 uses BGR.
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb_img
    except Exception as e:
        if isinstance(e, ImageProcessError):
            raise e
        raise ImageProcessError(f"Error processing base64 image: {str(e)}")

def process_upload_file(file_bytes: bytes) -> np.ndarray:
    """Converts raw file bytes to a cv2 numpy array (RGB)"""
    try:
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ImageProcessError("Could not decode the uploaded image.")
            
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb_img
    except Exception as e:
        if isinstance(e, ImageProcessError):
            raise e
        raise ImageProcessError(f"Error processing image file: {str(e)}")
