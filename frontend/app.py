import os
import sys
import io
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pydeck as pdk

# Configure stdout utf-8 for Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add root folder to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Import wrapper directly as fallback / direct engine
try:
    from model_wrapper import RescueVisionModel
    from rescuevision.schema import PRIORITY_DEFINITIONS
    HAS_LOCAL_WRAPPER = True
except Exception as e:
    HAS_LOCAL_WRAPPER = False

# Page Configuration
st.set_page_config(
    page_title="RescueVision AI | Disaster Damage & Rescue Triage",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Emergency Operations Styling
st.markdown("""
<style>
    /* Dark Emergency Command Center Theme */
    .stApp {
        background-color: #090d16;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        backdrop-filter: blur(8px);
    }
    
    /* Metric Cards */
    .metric-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .metric-val {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Priority Badges */
    .badge-p1 {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-p2 {
        background-color: rgba(249, 115, 22, 0.2);
        color: #f97316;
        border: 1px solid #f97316;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-p3 {
        background-color: rgba(234, 179, 8, 0.2);
        color: #eab308;
        border: 1px solid #eab308;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-p4 {
        background-color: rgba(34, 197, 94, 0.2);
        color: #22c55e;
        border: 1px solid #22c55e;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }

    /* St tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        color: #cbd5e1;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ef4444 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_cached_model():
    """Cache model loading so it's loaded only once."""
    has_weights = (ROOT_DIR / "rescuevision_model.keras").exists()
    return RescueVisionModel(mock_mode=not has_weights)


# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/satellite.png", width=70)
    st.title("RescueVision AI")
    st.caption("Tactical Hurricane Disaster Response Platform")
    st.markdown("---")

    # Backend Connection Mode
    st.subheader("⚙️ System Status")
    model = load_cached_model()
    
    if (ROOT_DIR / "rescuevision_model.keras").exists():
        st.success("✅ Real Model Active (ResNet50 Weights Loaded)")
    else:
        st.warning("⚠️ Simulation Mode (Awaiting Colab `.keras` file)")
        with st.expander("📥 How to load real weights"):
            st.markdown("""
            1. In Colab, run:
            ```python
            from google.colab import files
            files.download('rescuevision_model.keras')
            ```
            2. Place `rescuevision_model.keras` into the project root directory.
            3. Restart this web app!
            """)

    st.markdown("---")
    st.markdown("### 🎯 Priority Tier Guide")
    st.markdown("""
    - 🔴 **P1 Critical (>90%)**: Immediate rescue squad dispatch
    - 🟠 **P2 High (70-90%)**: Urgent structural assessment
    - 🟡 **P3 Medium (50-70%)**: Scheduled inspection
    - 🟢 **P4 Low (<50%)**: Structure intact, no action
    """)
    st.markdown("---")
    st.caption("Engine: PyTorch/TensorFlow 2.x • Transfer Learning • ResNet50")


# Hero Header
st.markdown("""
<div class="hero-banner">
    <h1 style="margin: 0; color: #ffffff; font-size: 2.2rem; display: flex; align-items: center; gap: 12px;">
        🚨 RescueVision — Disaster Damage & Rescue Triage
    </h1>
    <p style="margin: 8px 0 0 0; color: #cbd5e1; font-size: 1.05rem;">
        Automated aerial assessment of disaster-affected structures using Deep Transfer Learning. 
        Instant damage classification and rescue priority assignment for emergency squads.
    </p>
</div>
""", unsafe_allow_html=True)


# Tabs for different user flows
tab_single, tab_batch, tab_map, tab_docs = st.tabs([
    "🛰️ Live Satellite Crop Assessment",
    "📋 Batch Triage & Reconnaissance",
    "🗺️ Geospatial Rescue Map",
    "📖 Model Architecture & Info"
])

# -------------------------------------------------------------
# TAB 1: Single Image Inference
# -------------------------------------------------------------
with tab_single:
    col_input, col_result = st.columns([1, 1.2], gap="large")

    with col_input:
        st.subheader("1. Provide Satellite Tile")
        
        input_method = st.radio(
            "Input Mode:",
            ["Choose from Sample Dataset", "Upload Satellite Crop"],
            horizontal=True
        )

        selected_image_path = None
        uploaded_file = None

        if input_method == "Choose from Sample Dataset":
            # Search available test images
            sample_dir = ROOT_DIR / "dataset" / "test"
            sample_options = {}

            if sample_dir.exists():
                damage_files = list((sample_dir / "damage").glob("*.jpeg"))[:10]
                no_damage_files = list((sample_dir / "no_damage").glob("*.jpeg"))[:10]

                for p in damage_files:
                    sample_options[f"🔴 Damaged Sample: {p.name}"] = p
                for p in no_damage_files:
                    sample_options[f"🟢 Intact Sample: {p.name}"] = p

            if sample_options:
                selection = st.selectbox("Select sample satellite crop:", list(sample_options.keys()))
                selected_image_path = sample_options[selection]
                st.image(str(selected_image_path), caption=f"Selected: {selected_image_path.name}", use_container_width=True)
            else:
                st.info("No sample images found in dataset folder. Please upload an image.")
        else:
            uploaded_file = st.file_uploader(
                "Upload aerial / satellite crop (JPEG/PNG):",
                type=["jpeg", "jpg", "png"]
            )
            if uploaded_file is not None:
                st.image(uploaded_file, caption="Uploaded Satellite Image", use_container_width=True)

    with col_result:
        st.subheader("2. AI Damage & Rescue Assessment")

        target_input = None
        source_name = None

        if input_method == "Choose from Sample Dataset" and selected_image_path:
            target_input = selected_image_path
            source_name = selected_image_path.name
        elif uploaded_file is not None:
            target_input = uploaded_file.getvalue()
            source_name = uploaded_file.name

        if target_input is not None:
            with st.spinner("Analyzing structural integrity with ResNet50..."):
                t0 = time.time()
                pred = model.predict(target_input, source_id=source_name)
                lat_ms = (time.time() - t0) * 1000

            # Badge Styling
            badge_class = f"badge-{pred.priority_level.lower()}"
            status_text = "STRUCTURAL DAMAGE CONFIRMED" if pred.is_damaged else "NO MAJOR DAMAGE"
            status_color = "#ef4444" if pred.is_damaged else "#22c55e"

            st.markdown(f"""
            <div style="background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 1.25rem; font-weight: 800; color: {status_color};">
                        {status_text}
                    </span>
                    <span class="{badge_class}">
                        {pred.emoji} {pred.priority}
                    </span>
                </div>
                <div style="font-size: 0.95rem; color: #94a3b8; margin-bottom: 16px;">
                    <strong>Recommended Action:</strong> <span style="color: #f1f5f9;">{pred.action}</span>
                </div>
                <div style="margin-bottom: 8px; display: flex; justify-content: space-between;">
                    <span style="color: #cbd5e1; font-weight: 600;">Damage Probability:</span>
                    <span style="color: #f87171; font-weight: 700;">{pred.damage_percentage}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence Progress Bar
            st.progress(pred.damage_confidence)

            # Location and Model Metadata
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                lat_str = f"{pred.latitude:.4f}° N" if pred.latitude else "N/A"
                st.metric("Latitude", lat_str)
            with m_col2:
                lon_str = f"{abs(pred.longitude):.4f}° W" if pred.longitude else "N/A"
                st.metric("Longitude", lon_str)
            with m_col3:
                st.metric("Inference Latency", f"{lat_ms:.1f} ms")

            # Visual Assessment Card Generator
            st.markdown("#### 📊 Diagnostic Inspection Card")
            card_buf = io.BytesIO()
            model.visualize(target_input, prediction=pred, save_path=card_buf)
            card_buf.seek(0)
            
            st.image(card_buf, use_container_width=True)
            
            # Download actions
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    "📥 Download Inspection Card (PNG)",
                    data=card_buf.getvalue(),
                    file_name=f"assessment_{source_name or 'building'}.png",
                    mime="image/png",
                    use_container_width=True
                )
            with dl_col2:
                st.download_button(
                    "📄 Export JSON Report",
                    data=pred.to_json(indent=2),
                    file_name=f"triage_{source_name or 'building'}.json",
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.info("👈 Select a sample satellite image or upload an aerial crop on the left to begin analysis.")


# -------------------------------------------------------------
# TAB 2: Batch Assessment & Triage Table
# -------------------------------------------------------------
with tab_batch:
    st.subheader("📋 Reconnaissance Batch Triage")
    st.write("Scan entire hurricane sectors, view priority distributions, and generate tactical field triage lists.")

    sample_dir = ROOT_DIR / "dataset" / "test"
    available_samples = []
    if sample_dir.exists():
        available_samples = list((sample_dir / "damage").glob("*.jpeg"))[:25] + \
                            list((sample_dir / "no_damage").glob("*.jpeg"))[:25]

    batch_source = st.radio(
        "Batch Data Source:",
        ["Process 50 Preloaded Hurricane Harvey Satellite Images", "Upload Multiple Image Crops"],
        horizontal=True
    )

    batch_inputs = []
    if batch_source == "Upload Multiple Image Crops":
        uploaded_batch = st.file_uploader(
            "Upload multiple satellite crops:",
            type=["jpeg", "jpg", "png"],
            accept_multiple_files=True
        )
        if uploaded_batch:
            batch_inputs = [(f.name, f.getvalue()) for f in uploaded_batch]
    else:
        if available_samples:
            batch_inputs = [(p.name, p) for p in available_samples]
            st.caption(f"Loaded {len(batch_inputs)} aerial crops from Hurricane Harvey test partition.")
        else:
            st.warning("Sample dataset archive not found. Please upload images directly.")

    if batch_inputs:
        if st.button("🚀 Execute Batch Aerial Triage", type="primary"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            results = []
            for idx, (fname, item) in enumerate(batch_inputs):
                status_text.text(f"Scanning structure {idx+1}/{len(batch_inputs)}: {fname}...")
                pred = model.predict(item, source_id=fname)
                results.append(pred)
                progress_bar.progress((idx + 1) / len(batch_inputs))

            status_text.text(f"✅ Completed triage of {len(batch_inputs)} structures.")
            
            # Store results in session state
            st.session_state["batch_results"] = results

    # Render Batch Results if available
    if "batch_results" in st.session_state and st.session_state["batch_results"]:
        results = st.session_state["batch_results"]
        
        # Summary KPI cards
        total_count = len(results)
        p1_count = sum(1 for r in results if r.priority_level == "P1")
        p2_count = sum(1 for r in results if r.priority_level == "P2")
        damaged_count = sum(1 for r in results if r.is_damaged)
        damage_rate = (damaged_count / total_count * 100) if total_count > 0 else 0

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #f1f5f9;">{total_count}</div>
                <div class="metric-label">Structures Scanned</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #ef4444;">{p1_count}</div>
                <div class="metric-label">🔴 P1 Critical Rescues</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #f97316;">{p2_count}</div>
                <div class="metric-label">🟠 P2 Urgent Sites</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #38bdf8;">{damage_rate:.1f}%</div>
                <div class="metric-label">Sector Damage Rate</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Priority Breakdown Chart
        st.subheader("📊 Priority Level Distribution")
        p_counts = {"P1 - CRITICAL": 0, "P2 - HIGH": 0, "P3 - MEDIUM": 0, "P4 - LOW": 0}
        for r in results:
            if r.priority in p_counts:
                p_counts[r.priority] += 1
        
        chart_df = pd.DataFrame({
            "Priority": list(p_counts.keys()),
            "Structures": list(p_counts.values())
        }).set_index("Priority")
        st.bar_chart(chart_df)

        # Detailed Data Table
        st.subheader("📋 Triage Target Manifest")
        data_rows = []
        for r in results:
            data_rows.append({
                "Filename": Path(r.image_source).name if r.image_source else "Unknown",
                "Status": "DAMAGED" if r.is_damaged else "INTACT",
                "Confidence": r.damage_percentage,
                "Priority": r.priority,
                "Action": r.action,
                "Latitude": r.latitude,
                "Longitude": r.longitude
            })
        df_manifest = pd.DataFrame(data_rows)

        # Filters
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            priority_filter = st.multiselect(
                "Filter by Priority:",
                ["P1 - CRITICAL", "P2 - HIGH", "P3 - MEDIUM", "P4 - LOW"],
                default=["P1 - CRITICAL", "P2 - HIGH", "P3 - MEDIUM", "P4 - LOW"]
            )
        with filter_col2:
            status_filter = st.multiselect(
                "Filter by Structural State:",
                ["DAMAGED", "INTACT"],
                default=["DAMAGED", "INTACT"]
            )

        filtered_df = df_manifest[
            df_manifest["Priority"].isin(priority_filter) &
            df_manifest["Status"].isin(status_filter)
        ]

        st.dataframe(filtered_df, use_container_width=True)

        # Export CSV
        csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Field Triage Manifest (CSV)",
            data=csv_bytes,
            file_name="rescue_triage_manifest.csv",
            mime="text/csv",
            type="primary"
        )


# -------------------------------------------------------------
# TAB 3: Interactive Rescue Map
# -------------------------------------------------------------
with tab_map:
    st.subheader("🗺️ Hurricane Sector Tactical Map")
    st.write("Geospatial distribution of damaged structures color-coded by rescue priority.")

    # Collect coordinates from batch or preloaded samples
    map_data = []
    if "batch_results" in st.session_state and st.session_state["batch_results"]:
        target_results = st.session_state["batch_results"]
    else:
        # Run auto-scan on 30 sample images to populate map immediately
        sample_dir = ROOT_DIR / "dataset" / "test"
        sample_files = []
        if sample_dir.exists():
            sample_files = list((sample_dir / "damage").glob("*.jpeg"))[:25] + \
                           list((sample_dir / "no_damage").glob("*.jpeg"))[:15]
        target_results = [model.predict(f, source_id=f.name) for f in sample_files]

    for r in target_results:
        if r.latitude is not None and r.longitude is not None:
            # Color based on priority
            if r.priority_level == "P1":
                color = [239, 68, 68, 220]  # Red
            elif r.priority_level == "P2":
                color = [249, 115, 22, 220] # Orange
            elif r.priority_level == "P3":
                color = [234, 179, 8, 220]  # Yellow
            else:
                color = [34, 197, 94, 220]  # Green

            map_data.append({
                "latitude": r.latitude,
                "longitude": r.longitude,
                "priority": r.priority,
                "damage": r.damage_percentage,
                "color": color,
                "action": r.action
            })

    if map_data:
        df_map = pd.DataFrame(map_data)
        mid_lat = df_map["latitude"].mean()
        mid_lon = df_map["longitude"].mean()

        view_state = pdk.ViewState(
            latitude=mid_lat,
            longitude=mid_lon,
            zoom=11,
            pitch=35
        )

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_map,
            get_position=["longitude", "latitude"],
            get_color="color",
            get_radius=180,
            pickable=True
        )

        r_deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "Priority: {priority}\nDamage Conf: {damage}\nAction: {action}"},
            map_style="mapbox://styles/mapbox/dark-v10"
        )

        st.pydeck_chart(r_deck)
        st.caption("🔴 Red = P1 Critical | 🟠 Orange = P2 High | 🟡 Yellow = P3 Medium | 🟢 Green = P4 Low")
    else:
        st.info("No GPS coordinates found in dataset filenames.")


# -------------------------------------------------------------
# TAB 4: Model Architecture & Documentation
# -------------------------------------------------------------
with tab_docs:
    st.subheader("📖 RescueVision Deep Learning Architecture")
    
    doc_col1, doc_col2 = st.columns(2)
    with doc_col1:
        st.markdown("""
        ### 🧠 Model Specifications
        - **Base Network**: `ResNet50` (Residual Networks with 50 layers)
        - **Pretrained Weights**: `ImageNet`
        - **Fine-Tuning**: Last 20 convolutional layers unfreezed
        - **Classification Head**:
          - `GlobalAveragePooling2D`
          - `BatchNormalization` + `Dropout(0.5)`
          - `Dense(256, activation='relu')`
          - `BatchNormalization` + `Dropout(0.3)`
          - `Dense(1, activation='sigmoid')`
        - **Input Tensor**: `(224, 224, 3)` with Caffe-style mean subtraction
        - **Dataset**: Satellite Images of Hurricane Harvey (Texas)
        """)

    with doc_col2:
        st.markdown("""
        ### 🎯 Benchmark Metrics (Notebook Validation)
        - **Test Accuracy**: `~95%+`
        - **ROC AUC Score**: `~0.98`
        - **Loss Function**: `Binary Crossentropy`
        - **Optimizer**: `Adam (lr=1e-5 fine-tuning)`
        
        ### 📡 REST API Integration
        The FastAPI backend is deployable at port `8000`:
        - `GET /health` — Service liveness & model readiness
        - `POST /api/v1/predict` — Single satellite crop assessment
        - `POST /api/v1/predict-batch` — Multi-image reconnaissance
        - `POST /api/v1/assess-card` — Generates visual PNG triage card
        """)
