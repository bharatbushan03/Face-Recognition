import base64
import binascii
import cv2
import numpy as np
import logging
from backend.utils.error_handlers import ImageProcessError


logger = logging.getLogger(__name__)
MAX_IMAGE_BYTES = 10 * 1024 * 1024


def _decode_image_bytes(image_bytes: bytes, source: str) -> np.ndarray:
    if not image_bytes:
        raise ImageProcessError(f"{source} image payload is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ImageProcessError(
            f"{source} image payload is too large ({len(image_bytes)} bytes). "
            f"Max allowed: {MAX_IMAGE_BYTES} bytes."
        )

    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageProcessError("Could not decode the image.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def process_base64_image(base64_str: str) -> np.ndarray:
    """Converts a base64 string to a cv2 numpy array (RGB)"""
    try:
        # Some base64 strings come with 'data:image/jpeg;base64,' prefix 
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]

        img_data = base64.b64decode(base64_str, validate=True)
        return _decode_image_bytes(img_data, source="Base64")
    except binascii.Error as exc:
        logger.warning("Invalid base64 image payload: %s", exc)
        raise ImageProcessError("Invalid base64 image format.") from exc
    except Exception as e:
        if isinstance(e, ImageProcessError):
            raise e
        raise ImageProcessError(f"Error processing base64 image: {str(e)}")

def process_upload_file(file_bytes: bytes) -> np.ndarray:
    """Converts raw file bytes to a cv2 numpy array (RGB)"""
    try:
        return _decode_image_bytes(file_bytes, source="Uploaded")
    except Exception as e:
        if isinstance(e, ImageProcessError):
            raise e
        raise ImageProcessError(f"Error processing image file: {str(e)}")
