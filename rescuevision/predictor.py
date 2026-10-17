import os
import sys
import io
from pathlib import Path
from typing import Union, List, Optional, Tuple
import numpy as np
from PIL import Image

from .schema import RescuePrediction, map_priority
from .utils import (
    load_as_pil_image,
    preprocess_for_resnet50,
    parse_gps_coordinates,
    TARGET_IMAGE_SIZE
)

# Potential default model locations
DEFAULT_MODEL_CANDIDATES = [
    "rescuevision_model.keras",
    "models/rescuevision_model.keras",
    "rescuevision_model.h5",
    "../rescuevision_model.keras"
]


class RescueVisionModel:
    """
    Production-ready wrapper for the RescueVision Hurricane Damage & Rescue Priority Model.
    
    Trained on ResNet50 Transfer Learning for binary satellite damage classification.
    Sigmoid output:
        - Class 0: 'damage'
        - Class 1: 'no_damage'
        - damage_confidence = 1.0 - sigmoid_output
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        mock_mode: bool = False
    ):
        """
        Initialize the RescueVision inference model.
        
        Args:
            model_path: Path to the .keras or .h5 file. If None, searches standard locations.
            mock_mode: If True, operates in simulation mode without needing TensorFlow or weights.
                       Useful for frontend testing while Colab is training.
        """
        self.mock_mode = mock_mode
        self.model = None
        self.model_path = None

        if self.mock_mode:
            print("⚠️ [RescueVision] Initialized in MOCK MODE (Simulating predictions).")
            return

        # Locate model file
        resolved_path = self._resolve_model_path(model_path)
        if resolved_path is None:
            raise FileNotFoundError(
                "\n" + "=" * 70 + "\n"
                "🚨 RescueVision Model file not found!\n\n"
                "To get your trained model from Google Colab, run in Colab:\n"
                "    from google.colab import files\n"
                "    files.download('rescuevision_model.keras')\n\n"
                "Then save it in this project folder as: 'rescuevision_model.keras'\n\n"
                "💡 Pro-tip: If you want to test your application right now before Colab\n"
                "finishes training, you can initialize with mock mode:\n"
                "    model = RescueVisionModel(mock_mode=True)\n"
                + "=" * 70
            )

        self.model_path = resolved_path
        try:
            self._load_model()
        except Exception as e:
            print(f"⚠️ [RescueVision] Failed to load model weights: {e}")
            print("⚠️ [RescueVision] Falling back to MOCK MODE.")
            self.mock_mode = True

    def _resolve_model_path(self, model_path: Optional[Union[str, Path]]) -> Optional[Path]:
        """Locate the .keras model file."""
        if model_path:
            p = Path(model_path)
            return p if p.exists() else None

        cwd = Path.cwd()
        for candidate in DEFAULT_MODEL_CANDIDATES:
            p = cwd / candidate
            if p.exists():
                return p

        # Check script parent directory
        script_dir = Path(__file__).resolve().parent.parent
        for candidate in DEFAULT_MODEL_CANDIDATES:
            p = script_dir / candidate
            if p.exists():
                return p

        return None

    def _load_model(self):
        """Load the Keras model into memory."""
        try:
            import tensorflow as tf
            # Suppress verbose TensorFlow warnings
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            print(f"📦 [RescueVision] Loading model from: {self.model_path.resolve()} ...")
            self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
            print("✅ [RescueVision] Model loaded successfully.")
        except ImportError:
            raise ImportError(
                "TensorFlow is required to load the model. Run: pip install tensorflow"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Keras model at {self.model_path}: {e}")

    def predict(
        self,
        image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
        source_id: Optional[str] = None
    ) -> RescuePrediction:
        """
        Run inference on a single image.
        
        Args:
            image_input: File path, URL, raw bytes, PIL Image, or numpy array.
            source_id: Optional name or ID for tracking the image.
            
        Returns:
            RescuePrediction object with class, confidence, priority, and operational action.
        """
        pil_image = load_as_pil_image(image_input)
        
        # Determine source identifier and parse coordinates
        source_path = source_id
        if source_path is None and isinstance(image_input, (str, Path)):
            source_path = str(image_input)

        lat, lon = (None, None)
        if source_path:
            lat, lon = parse_gps_coordinates(source_path)

        if self.mock_mode:
            # Deterministic simulation based on image hash or filename
            seed = sum(pil_image.getdata()[0]) % 1000
            pred_prob = (seed / 1000.0)
        else:
            # Preprocess
            preprocessed = preprocess_for_resnet50(pil_image, TARGET_IMAGE_SIZE)
            batch = np.expand_dims(preprocessed, axis=0)

            # Predict (model outputs probability of class 1: 'no_damage')
            raw_output = self.model.predict(batch, verbose=0)
            pred_prob = float(raw_output[0][0])

        # Compute damage confidence:
        # In RescueVision notebook:
        #   Class 0 = 'damage', Class 1 = 'no_damage'
        #   Sigmoid output close to 1 => 'no_damage'
        #   Sigmoid output close to 0 => 'damage'
        #   damage_conf = 1.0 - pred_prob
        damage_confidence = float(np.clip(1.0 - pred_prob, 0.0, 1.0))
        no_damage_confidence = float(np.clip(pred_prob, 0.0, 1.0))
        
        is_damaged = damage_confidence > 0.50
        predicted_class = 'damage' if is_damaged else 'no_damage'
        priority_info = map_priority(damage_confidence)

        return RescuePrediction(
            prediction=predicted_class,
            is_damaged=is_damaged,
            damage_confidence=round(damage_confidence, 4),
            damage_percentage=f"{damage_confidence * 100:.1f}%",
            no_damage_confidence=round(no_damage_confidence, 4),
            priority=priority_info.code,
            priority_level=priority_info.level,
            priority_name=priority_info.name,
            emoji=priority_info.emoji,
            action=priority_info.action,
            raw_score=round(pred_prob, 4),
            image_source=source_path,
            latitude=lat,
            longitude=lon
        )

    def predict_batch(
        self,
        image_inputs: List[Union[str, Path, bytes, Image.Image, np.ndarray]],
        batch_size: int = 32
    ) -> List[RescuePrediction]:
        """
        Run inference on a batch of images efficiently.
        
        Args:
            image_inputs: List of image paths, PIL images, or arrays.
            batch_size: Inference batch size.
            
        Returns:
            List of RescuePrediction objects.
        """
        results: List[RescuePrediction] = []
        if not image_inputs:
            return results

        # Process each image
        for i in range(0, len(image_inputs), batch_size):
            chunk = image_inputs[i:i + batch_size]
            for item in chunk:
                results.append(self.predict(item))

        return results

    def visualize(
        self,
        image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
        prediction: Optional[RescuePrediction] = None,
        save_path: Optional[Union[str, Path, io.BytesIO]] = None,
        show: bool = False
    ):
        """
        Generate a high-resolution, magazine-quality tactical diagnostic inspection card.
        
        Args:
            image_input: Satellite crop image.
            prediction: Optional precomputed RescuePrediction.
            save_path: File path or BytesIO buffer to save PNG.
            show: If True, calls plt.show().
            
        Returns:
            matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        pil_image = load_as_pil_image(image_input)
        if prediction is None:
            prediction = self.predict(image_input)

        # Clear any prior state and create fixed-dimension figure
        plt.close('all')
        fig = plt.figure(figsize=(9, 4.5), dpi=180, facecolor='#0b1120')

        # Grid: left for image (42%), right for tactical HUD (58%)
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], left=0.03, right=0.97, top=0.92, bottom=0.08, wspace=0.08)
        ax_img = fig.add_subplot(gs[0])
        ax_info = fig.add_subplot(gs[1])

        # --- 1. Left: Satellite Image ---
        ax_img.imshow(pil_image)
        ax_img.axis('off')
        
        # Border around image
        for spine in ax_img.spines.values():
            spine.set_edgecolor('#334155')
            spine.set_linewidth(1.5)
            spine.set_visible(True)

        filename = Path(prediction.image_source).name if prediction.image_source else "Satellite Target"
        if len(filename) > 30:
            filename = filename[:27] + "..."
        ax_img.set_title(f"Target: {filename}", color='#94a3b8', fontsize=9, pad=8, family='sans-serif', weight='bold')

        # --- 2. Right: Tactical Assessment Panel ---
        ax_info.set_facecolor('#131d31')
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)
        ax_info.axis('off')

        # Priority Color Theme
        color_themes = {
            'P1': {'primary': '#ef4444', 'bg': '#3b1219', 'label': 'CRITICAL'},
            'P2': {'primary': '#f97316', 'bg': '#381c12', 'label': 'HIGH'},
            'P3': {'primary': '#eab308', 'bg': '#35290f', 'label': 'MEDIUM'},
            'P4': {'primary': '#22c55e', 'bg': '#0e2e1d', 'label': 'LOW'},
        }
        theme = color_themes.get(prediction.priority_level, {'primary': '#38bdf8', 'bg': '#0c2738', 'label': 'INFO'})

        # Header Agency / Watermark
        ax_info.text(0.04, 0.92, "RESCUEVISION AI  •  RAPID DISASTER TRIAGE", 
                     transform=ax_info.transAxes, color='#64748b', fontsize=8, weight='bold', family='sans-serif')

        # Main Status Label
        status_text = "STRUCTURAL DAMAGE" if prediction.is_damaged else "STRUCTURE INTACT"
        status_color = "#f87171" if prediction.is_damaged else "#4ade80"
        ax_info.text(0.04, 0.81, status_text, 
                     transform=ax_info.transAxes, color=status_color, fontsize=15, weight='black', family='sans-serif')

        # Priority Pill Badge (drawn cleanly with FancyBboxPatch)
        pill = patches.FancyBboxPatch(
            (0.04, 0.65), 0.60, 0.11,
            boxstyle="round,pad=0.015,rounding_size=0.03",
            facecolor=theme['bg'], edgecolor=theme['primary'], linewidth=1.5,
            transform=ax_info.transAxes
        )
        ax_info.add_patch(pill)

        # Bullet dot + Priority Text inside pill
        ax_info.plot(0.09, 0.705, marker='o', markersize=7, color=theme['primary'], transform=ax_info.transAxes)
        ax_info.text(0.14, 0.68, f"PRIORITY {prediction.priority}", 
                     transform=ax_info.transAxes, color=theme['primary'], fontsize=10, weight='bold', family='sans-serif')

        # Damage Probability Bar
        ax_info.text(0.04, 0.54, "Damage Confidence:", 
                     transform=ax_info.transAxes, color='#94a3b8', fontsize=8.5, family='sans-serif')
        ax_info.text(0.48, 0.54, f"{prediction.damage_percentage}", 
                     transform=ax_info.transAxes, color='#f1f5f9', fontsize=9.5, weight='bold', family='sans-serif')

        # Progress bar background
        bg_bar = patches.FancyBboxPatch(
            (0.04, 0.44), 0.90, 0.05,
            boxstyle="round,pad=0.005,rounding_size=0.02",
            facecolor='#1e293b', edgecolor='none',
            transform=ax_info.transAxes
        )
        ax_info.add_patch(bg_bar)

        # Progress bar fill
        fill_width = max(0.04, 0.90 * float(prediction.damage_confidence))
        fill_bar = patches.FancyBboxPatch(
            (0.04, 0.44), fill_width, 0.05,
            boxstyle="round,pad=0.005,rounding_size=0.02",
            facecolor=theme['primary'], edgecolor='none',
            transform=ax_info.transAxes
        )
        ax_info.add_patch(fill_bar)

        # Action Directive Box
        directive_box = patches.FancyBboxPatch(
            (0.04, 0.18), 0.90, 0.20,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor='#0b1120', edgecolor='#1e293b', linewidth=1,
            transform=ax_info.transAxes
        )
        ax_info.add_patch(directive_box)

        ax_info.text(0.07, 0.31, "TACTICAL DIRECTIVE:", 
                     transform=ax_info.transAxes, color='#94a3b8', fontsize=7.5, weight='bold', family='sans-serif')
        ax_info.text(0.07, 0.22, f"“{prediction.action}”", 
                     transform=ax_info.transAxes, color='#f8fafc', fontsize=9, style='italic', family='sans-serif')

        # Geolocation Footer
        if prediction.latitude and prediction.longitude:
            geo_str = f"GPS: {prediction.latitude:.5f}°N, {abs(prediction.longitude):.5f}°W"
        else:
            geo_str = "GPS: Coordinates unavailable"
        ax_info.text(0.04, 0.06, geo_str, 
                     transform=ax_info.transAxes, color='#38bdf8', fontsize=7.5, family='sans-serif', weight='bold')

        if save_path:
            if hasattr(save_path, "write"):
                fig.savefig(save_path, format="png", dpi=180, facecolor=fig.get_facecolor(), edgecolor='none')
            else:
                fig.savefig(str(save_path), dpi=180, facecolor=fig.get_facecolor(), edgecolor='none')
                print(f"📊 Assessment visualization saved to: {save_path}")

        if show:
            plt.show()

        plt.close(fig)
        return fig
