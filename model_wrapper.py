"""
Convenience entry-point and top-level wrapper for RescueVision.

Usage in your code:
    from model_wrapper import RescueVisionModel

    model = RescueVisionModel('rescuevision_model.keras')
    result = model.predict('sample.jpeg')
    print(result.summary())

Command line usage:
    python model_wrapper.py path/to/satellite_image.jpeg
"""

import sys
import json
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Import from rescuevision package
from rescuevision.predictor import RescueVisionModel
from rescuevision.schema import RescuePrediction, PriorityInfo
from rescuevision.utils import parse_gps_coordinates, load_as_pil_image

__all__ = ["RescueVisionModel", "RescuePrediction", "PriorityInfo"]


def main():
    if len(sys.argv) < 2:
        print("\n🚨 RescueVision CLI Usage:")
        print("  python model_wrapper.py <image_path> [--mock] [--visualize]")
        print("\nExample:")
        print("  python model_wrapper.py dataset/test/damage/-95.548123_30.900623.jpeg\n")
        sys.exit(1)

    image_path = sys.argv[1]
    mock = "--mock" in sys.argv
    vis = "--visualize" in sys.argv

    try:
        model = RescueVisionModel(mock_mode=mock)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    print(f"\n🔍 Analyzing image: {image_path} ...")
    prediction = model.predict(image_path)

    print("\n" + "=" * 60)
    print("  RESCUEVISION AI ASSESSMENT REPORT")
    print("=" * 60)
    print(f"  Target Image:        {prediction.image_source}")
    if prediction.latitude and prediction.longitude:
        print(f"  Coordinates:         {prediction.latitude:.5f}°N, {abs(prediction.longitude):.5f}°W")
    print(f"  Classification:      {prediction.prediction.upper()} (Damaged: {prediction.is_damaged})")
    print(f"  Damage Confidence:   {prediction.damage_percentage}")
    print(f"  Rescue Priority:     {prediction.emoji} {prediction.priority}")
    print(f"  Recommended Action:  {prediction.action}")
    print(f"  Raw Sigmoid Output:  {prediction.raw_score}")
    print("=" * 60 + "\n")

    if vis:
        save_path = Path("prediction_output.png")
        model.visualize(image_path, prediction=prediction, save_path=save_path)
        print(f"Visual assessment card saved to: {save_path.resolve()}\n")


if __name__ == "__main__":
    main()
