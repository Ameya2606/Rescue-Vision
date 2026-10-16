"""
RescueVision Unified Launcher
Runs both the FastAPI backend (port 8000) and the Streamlit frontend (port 8501) simultaneously.

Usage:
    python run_app.py
"""

import sys
import subprocess
import time
import os

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def main():
    print("=" * 65)
    print("🚀 Launching RescueVision Emergency Command Center...")
    print("=" * 65)

    has_model = os.path.exists("rescuevision_model.keras")
    if has_model:
        print("✅ Found 'rescuevision_model.keras'. Full deep learning mode active.")
    else:
        print("ℹ️ 'rescuevision_model.keras' not found yet.")
        print("   Running in Simulation Mode. Download the file from Colab to enable real inference.")

    print("\n[1/2] Starting FastAPI Backend on http://localhost:8000 ...")
    backend_proc = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", "8000"
    ])

    time.sleep(2)

    print("[2/2] Starting Streamlit Frontend on http://localhost:8501 ...")
    frontend_proc = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "frontend/app.py", "--server.port", "8501"
    ])

    print("\n" + "=" * 65)
    print("🎯 RescueVision is LIVE!")
    print("👉 Frontend Dashboard: http://localhost:8501")
    print("👉 FastAPI Swagger Docs: http://localhost:8000/docs")
    print("=" * 65)
    print("Press Ctrl+C to stop both servers.\n")

    try:
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down RescueVision services...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("Goodbye!")


if __name__ == "__main__":
    main()
