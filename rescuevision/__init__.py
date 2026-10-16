import sys

# Configure UTF-8 stdout for Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from .predictor import RescueVisionModel
from .schema import RescuePrediction, PriorityInfo, map_priority, PRIORITY_DEFINITIONS
from .utils import (
    load_as_pil_image,
    preprocess_for_resnet50,
    parse_gps_coordinates,
    TARGET_IMAGE_SIZE
)

__version__ = "1.0.0"
__all__ = [
    "RescueVisionModel",
    "RescuePrediction",
    "PriorityInfo",
    "map_priority",
    "PRIORITY_DEFINITIONS",
    "load_as_pil_image",
    "preprocess_for_resnet50",
    "parse_gps_coordinates",
    "TARGET_IMAGE_SIZE",
]
