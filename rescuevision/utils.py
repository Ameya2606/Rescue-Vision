import os
import re
import io
import base64
from typing import Union, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image

try:
    from tensorflow.keras.applications.resnet50 import preprocess_input as tf_resnet_preprocess
    HAS_TF_PREPROCESS = True
except Exception:
    HAS_TF_PREPROCESS = False

# ResNet50 default input shape
TARGET_IMAGE_SIZE = (224, 224)


def parse_gps_coordinates(identifier: Union[str, Path]) -> Tuple[Optional[float], Optional[float]]:
    """
    Attempt to extract latitude and longitude from satellite image filenames.
    Hurricane Harvey filenames often follow: '-95.12345_29.87654.jpeg' (<lon>_<lat>).
    
    Returns:
        (latitude, longitude) or (None, None)
    """
    name = Path(identifier).stem
    # Match patterns like: -95.12345_29.87654 or 29.87654_-95.12345
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", name)
    if len(matches) >= 2:
        try:
            val1 = float(matches[0])
            val2 = float(matches[1])
            # Latitude is usually between -90 and +90 (for Texas, ~28 to 32)
            # Longitude is between -180 and +180 (for Texas, ~-93 to -98)
            if -90.0 <= val1 <= 90.0 and -180.0 <= val2 <= 180.0 and abs(val2) > abs(val1):
                # val1 is lat, val2 is lon
                return round(val1, 6), round(val2, 6)
            elif -90.0 <= val2 <= 90.0 and -180.0 <= val1 <= 180.0:
                # val1 is lon, val2 is lat
                return round(val2, 6), round(val1, 6)
        except (ValueError, TypeError):
            pass
    return None, None


def load_as_pil_image(image_input: Union[str, Path, bytes, io.BytesIO, np.ndarray, Image.Image]) -> Image.Image:
    """
    Unified image loader.
    Accepts:
        - str/Path: file path
        - bytes: raw image bytes (e.g. from FastAPI/Flask file upload)
        - io.BytesIO: memory buffer
        - np.ndarray: OpenCV / NumPy array
        - Image.Image: already a PIL image
        - base64 string: data URI (data:image/jpeg;base64,...) or raw base64 string
    
    Returns:
        RGB PIL Image
    """
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    if isinstance(image_input, (str, Path)):
        str_val = str(image_input).strip()
        # Check if base64 data URI or raw base64
        if str_val.startswith("data:image") or (len(str_val) > 200 and not os.path.exists(str_val)):
            if "," in str_val:
                str_val = str_val.split(",", 1)[1]
            image_bytes = base64.b64decode(str_val)
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Local file path
        if not os.path.exists(str_val):
            raise FileNotFoundError(f"Image file does not exist: {str_val}")
        return Image.open(str_val).convert("RGB")

    if isinstance(image_input, bytes):
        return Image.open(io.BytesIO(image_input)).convert("RGB")

    if isinstance(image_input, io.BytesIO):
        image_input.seek(0)
        return Image.open(image_input).convert("RGB")

    if isinstance(image_input, np.ndarray):
        # Check if already 3 channels or 1 channel
        if image_input.ndim == 2:
            return Image.fromarray(image_input).convert("RGB")
        elif image_input.ndim == 3:
            # If float in [0, 1], scale to [0, 255]
            arr = image_input
            if arr.dtype in [np.float32, np.float64] and arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")

    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def preprocess_for_resnet50(
    image: Image.Image,
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE
) -> np.ndarray:
    """
    Resize to target_size and apply ResNet50 caffe-style preprocessing:
    Converts RGB -> BGR and subtracts ImageNet channel means:
    [103.939, 116.779, 123.68]
    
    Returns:
        Numpy array of shape (target_size[0], target_size[1], 3)
    """
    # High-quality resize
    resized = image.resize(target_size, Image.Resampling.BILINEAR)
    img_array = np.array(resized, dtype=np.float32)

    if HAS_TF_PREPROCESS:
        return tf_resnet_preprocess(img_array)
    else:
        # Fallback pure numpy implementation of tf.keras.applications.resnet50.preprocess_input ('caffe' mode)
        # 1. RGB to BGR
        img_array = img_array[..., ::-1]
        # 2. Subtract ImageNet means
        img_array[..., 0] -= 103.939
        img_array[..., 1] -= 116.779
        img_array[..., 2] -= 123.68
        return img_array
