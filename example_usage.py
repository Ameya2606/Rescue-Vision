import os
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from model_wrapper import RescueVisionModel

def run_demo():
    print("🚀 Initializing RescueVision Model...")

    # Check if real model exists; if not, run mock mode so the script runs out of the box!
    has_weights = os.path.exists("rescuevision_model.keras")
    mock_mode = not has_weights

    if mock_mode:
        print("ℹ️ Note: 'rescuevision_model.keras' not found yet. Running demo in mock mode.")
        print("  Once your Google Colab run is done, download 'rescuevision_model.keras'")
        print("  into this folder to use your real ResNet50 weights.\n")

    model = RescueVisionModel(mock_mode=mock_mode)

    # Pick a couple of sample images from dataset
    base_dir = Path("dataset") if Path("dataset").exists() else Path("archive(1)")
    damage_dir = base_dir / "test" / "damage"
    no_damage_dir = base_dir / "test" / "no_damage"

    sample_images = []
    if damage_dir.exists():
        damage_imgs = list(damage_dir.glob("*.jpeg"))
        if damage_imgs:
            sample_images.append(damage_imgs[0])
    if no_damage_dir.exists():
        no_damage_imgs = list(no_damage_dir.glob("*.jpeg"))
        if no_damage_imgs:
            sample_images.append(no_damage_imgs[0])

    if not sample_images:
        print("No test images found in dataset/test/. Please specify custom image paths.")
        return

    # 1. Single Image Inference
    first_image = sample_images[0]
    print(f"\n--- 1. Single Prediction on: {first_image.name} ---")
    prediction = model.predict(first_image)
    print(prediction.summary())

    print("\nFormatted JSON output (for APIs / microservices):")
    print(prediction.to_json(indent=2))

    # 2. Batch Inference
    if len(sample_images) > 1:
        print(f"\n--- 2. Batch Prediction on {len(sample_images)} images ---")
        batch_results = model.predict_batch(sample_images)
        for res in batch_results:
            print(f" -> {res.summary()}")

    # 3. Generate Diagnostic Card Visualization
    output_card = "demo_assessment.png"
    print(f"\n--- 3. Generating visual inspection card: {output_card} ---")
    model.visualize(first_image, prediction=prediction, save_path=output_card)
    print(f"✅ Generated {output_card} successfully!")


if __name__ == "__main__":
    run_demo()
