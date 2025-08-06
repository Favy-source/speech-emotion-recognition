#!/usr/bin/env python3
"""
Setup script for Speech Emotion Recognition project.
"""
import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd: str, check: bool = True):
    """Run shell command."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=check)
    return result.returncode == 0

def setup_project():
    """Setup the complete project environment."""
    print("🚀 Setting up Speech Emotion Recognition project...")
    
    # Create .env file
    env_content = """# Environment variables for SER project
PYTHONPATH=.
LOG_LEVEL=INFO
DEVICE=auto
API_HOST=0.0.0.0
API_PORT=8000

# Dataset paths
RAVDESS_PATH=data/raw/ravdess
TESS_PATH=data/raw/tess
FER_PLUS_PATH=data/raw/fer_plus

# Model paths
MODEL_SAVE_DIR=data/models
CHECKPOINT_DIR=data/models/checkpoints

# Logging
LOG_FILE=logs/ser.log
"""
    
    with open(".env", "w") as f:
        f.write(env_content.strip())
    print("✅ Created .env file")
    
    # Create placeholder files
    placeholder_dirs = [
        "data/raw", "data/processed", "data/models", "data/results", "logs"
    ]
    
    for directory in placeholder_dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / ".gitkeep").touch()
    
    print("✅ Created data directories with .gitkeep files")
    
    print("🎉 Project setup complete!")
    print("\nNext steps:")
    print("1. pip install -r requirements.txt")
    print("2. python scripts/download_datasets.py  # (when implemented)")
    print("3. python src/main.py --mode train")
    print("4. python src/main.py --mode api")
    
    return True

if __name__ == "__main__":
    setup_project()
