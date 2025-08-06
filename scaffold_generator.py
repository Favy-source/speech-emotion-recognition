#!/usr/bin/env python3
"""
Speech Emotion Recognition Project Scaffold Generator
======================================================

This script generates a complete project structure for the Speech Emotion Recognition system.
Run this script in an empty directory to create the full project scaffold.

Usage:
    python scaffold_generator.py

Author: Nwachukwu Favour Chinemerem
"""

import os
import sys
from pathlib import Path
from typing import Dict, List


class ProjectScaffold:
    """Generates a complete SER project scaffold."""
    
    def __init__(self, project_name: str = "speech-emotion-recognition"):
        self.project_name = project_name
        self.base_path = Path(project_name)
        
    def create_directory_structure(self) -> None:
        """Create the complete directory structure."""
        directories = [
            # Core source directories
            "src",
            "src/audio",
            "src/vision", 
            "src/models",
            "src/data",
            "src/training",
            "src/api",
            "src/api/routes",
            "src/api/models",
            "src/api/middleware",
            "src/realtime",
            "src/utils",
            
            # Configuration and scripts
            "configs",
            "scripts",
            
            # Tests
            "tests",
            "tests/test_audio",
            "tests/test_vision",
            "tests/test_models",
            "tests/test_api",
            
            # Data and results
            "data",
            "data/raw",
            "data/processed", 
            "data/models",
            "data/results",
            
            # Documentation and notebooks
            "docs",
            "notebooks",
            "logs",
            
            # Future implementations
            "rust_inference",
            "rust_inference/src",
            "rust_inference/models",
            "dashboard",
            "dashboard/src",
            "dashboard/public",
        ]
        
        print("📁 Creating directory structure...")
        for directory in directories:
            dir_path = self.base_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ {directory}")
    
    def get_file_contents(self) -> Dict[str, str]:
        """Return all file contents as a dictionary."""
        return {
            # Root configuration files
            "requirements.txt": '''# Core ML/AI
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
pandas>=2.0.0

# Audio Processing
librosa>=0.10.0
soundfile>=0.12.0
pyaudio>=0.2.11
audioread>=3.0.0

# Computer Vision
opencv-python>=4.8.0
mediapipe>=0.10.0
Pillow>=10.0.0

# API and Web
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
websockets>=11.0
pydantic>=2.0.0
python-multipart>=0.0.6

# Data Science
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
jupyter>=1.0.0

# Utilities
PyYAML>=6.0
python-dotenv>=1.0.0
tqdm>=4.65.0
click>=8.1.0
loguru>=0.7.0

# Development
pytest>=7.4.0
black>=23.0.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.5.0

# Optional: For ONNX export
onnx>=1.14.0
onnxruntime>=1.15.0
''',

            "setup.py": '''from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="speech-emotion-recognition",
    version="1.0.0",
    author="Nwachukwu Favour Chinemerem",
    author_email="human@nwachukwufavour.com",
    description="Multimodal Speech Emotion Recognition System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ser-train=src.main:main",
            "ser-api=src.api.main:main",
        ],
    },
)
''',

            ".gitignore": '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
ser_env/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep
data/models/*
!data/models/.gitkeep
logs/*.log
.env

# Jupyter Notebooks
.ipynb_checkpoints

# Model files
*.pt
*.pth
*.onnx
*.pkl

# Audio/Video files
*.wav
*.mp3
*.mp4
*.avi
''',

            ".env.example": '''# Environment variables for SER project
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
''',

            "README.md": '''# Speech Emotion Recognition (SER) System

A multimodal real-time emotion recognition system that combines audio and visual cues for enhanced emotional awareness in healthcare and support contexts.

## 🎯 Features

- **Multimodal Processing**: Audio (MFCCs, pitch, energy) + Visual (facial expressions, pose)
- **Real-time Inference**: <200ms latency for wearable deployment
- **Color-coded Alerts**: Red (Unsafe), Yellow (Caution), Green (Stable)
- **Edge Deployment**: Optimized for NVIDIA Jetson and similar hardware
- **RESTful API**: FastAPI backend with WebSocket support

## 🚀 Quick Start

```bash
# 1. Setup environment
python -m venv ser_env
source ser_env/bin/activate  # Windows: ser_env\\Scripts\\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup project
python scripts/setup_environment.py

# 4. Download datasets
python scripts/download_datasets.py

# 5. Train model
python src/main.py --mode train

# 6. Start API
python src/main.py --mode api
```

## 📊 Architecture

- **Audio Branch**: LSTM with attention for temporal modeling
- **Visual Branch**: CNN with attention pooling for spatial features
- **Fusion Layer**: Multimodal feature integration
- **Output**: 5 emotion classes (anger, fear, happiness, sadness, calm)

## 🛠️ Development

```bash
# Train model
make train

# Start API
make api

# Run tests
make test

# Format code
make format
```

## 📈 Performance

- **Accuracy**: >80% on clean conditions
- **Latency**: <200ms end-to-end
- **Power**: <15W for continuous operation

## 🤝 Contributing

This is a research project by Nwachukwu Favour Chinemerem. For collaborations, contact: human@nwachukwufavour.com

## 📄 License

MIT License - see LICENSE file for details.
''',

            "Makefile": '''.PHONY: setup install test train api clean lint format

# Setup project
setup:
\tpython scripts/setup_environment.py

# Install dependencies
install:
\tpip install -r requirements.txt

# Run tests
test:
\tpytest tests/ -v

# Train model
train:
\tpython src/main.py --mode train

# Start API
api:
\tpython src/main.py --mode api

# Start real-time processing
realtime:
\tpython src/main.py --mode realtime

# Clean temporary files
clean:
\tfind . -type f -name "*.pyc" -delete
\tfind . -type d -name "__pycache__" -delete
\trm -rf .pytest_cache/

# Lint code
lint:
\tflake8 src/ tests/
\tmypy src/

# Format code
format:
\tblack src/ tests/
\tisort src/ tests/

# Download datasets
data:
\tpython scripts/download_datasets.py

# Run all checks
check: lint test

# Docker build
docker-build:
\tdocker-compose build

# Docker run
docker-run:
\tdocker-compose up
''',

            # Configuration files
            "configs/model_config.yaml": '''# Speech Emotion Recognition Configuration

# Model Parameters
model:
  audio:
    sample_rate: 22050
    n_mfcc: 13
    hidden_size: 128
    num_layers: 2
    dropout: 0.3
    bidirectional: true
  
  visual:
    input_size: [224, 224]
    num_channels: 3
    backbone: "resnet18"
    pretrained: true
  
  fusion:
    hidden_size: 256
    num_classes: 5  # anger, fear, happiness, sadness, calm
    dropout: 0.3

# Training Parameters
training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100
  weight_decay: 0.0001
  patience: 10
  device: "auto"  # auto, cpu, cuda

# Data Parameters
data:
  datasets:
    - "RAVDESS"
    - "TESS"
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  augmentation: true

# API Parameters
api:
  host: "0.0.0.0"
  port: 8000
  max_file_size: 50  # MB
  cors_origins: ["*"]

# Real-time Parameters
realtime:
  buffer_size: 1024
  inference_rate: 5  # Hz
  confidence_threshold: 0.7

# Logging
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/ser.log"
''',

            # Core source files
            "src/__init__.py": '',
            
            "src/main.py": '''"""
Main entry point for the Speech Emotion Recognition system.
"""
import argparse
import asyncio
from pathlib import Path

from src.utils.logger import setup_logger
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Speech Emotion Recognition System")
    parser.add_argument("--mode", choices=["train", "api", "realtime"], 
                       required=True, help="Operation mode")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml",
                       help="Configuration file path")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cpu, cuda, auto)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("SER_MAIN")
    logger.info(f"Starting SER system in {args.mode} mode")
    
    # Load configuration
    config = load_config(args.config)
    
    if args.mode == "train":
        from src.training.trainer import train_model
        train_model(config, args.device)
        
    elif args.mode == "api":
        from src.api.main import start_api
        asyncio.run(start_api(config))
        
    elif args.mode == "realtime":
        from src.realtime.processor import start_realtime_processing
        asyncio.run(start_realtime_processing(config))


if __name__ == "__main__":
    main()
''',

            # Audio processing module
            "src/audio/__init__.py": '',
            
            "src/audio/processor.py": '''"""
Audio processing module for feature extraction and real-time analysis.
"""
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, Tuple, Optional
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AudioProcessor:
    """Main audio processing class for emotion recognition."""
    
    def __init__(self, sample_rate: int = 22050, n_mfcc: int = 13):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.logger = logger
        
    def load_audio(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio file and return audio data and sample rate."""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            self.logger.info(f"Loaded audio: {file_path}, duration: {len(audio)/sr:.2f}s")
            return audio, sr
        except Exception as e:
            self.logger.error(f"Error loading audio {file_path}: {e}")
            raise
    
    def extract_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract comprehensive audio features for emotion recognition."""
        features = {}
        
        try:
            # MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc
            )
            features['mfcc'] = mfccs
            
            # Pitch/Fundamental frequency
            pitches, magnitudes = librosa.piptrack(
                y=audio, sr=self.sample_rate, threshold=0.1
            )
            features['pitch'] = pitches
            
            # Energy/RMS
            rms = librosa.feature.rms(y=audio)
            features['energy'] = rms
            
            # Chroma features
            chroma = librosa.feature.chroma(y=audio, sr=self.sample_rate)
            features['chroma'] = chroma
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio, sr=self.sample_rate
            )
            features['spectral_centroid'] = spectral_centroids
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)
            features['zcr'] = zcr
            
            self.logger.debug(f"Extracted {len(features)} feature types")
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            raise
    
    def preprocess_for_model(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Preprocess features for model input."""
        # Combine features into single array
        feature_list = []
        
        for feature_name, feature_data in features.items():
            if len(feature_data.shape) > 1:
                # Take mean across time axis for multi-dimensional features
                feature_vector = np.mean(feature_data, axis=1)
            else:
                feature_vector = feature_data
            
            feature_list.append(feature_vector.flatten())
        
        combined_features = np.concatenate(feature_list)
        
        self.logger.debug(f"Combined feature vector shape: {combined_features.shape}")
        return combined_features
''',

            # Models
            "src/models/__init__.py": '',
            
            "src/models/audio_lstm.py": '''"""
LSTM model for audio-based emotion recognition.
"""
import torch
import torch.nn as nn
from typing import Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AudioLSTM(nn.Module):
    """LSTM model for processing sequential audio features."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,  # anger, fear, happiness, sadness, calm
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super(AudioLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention mechanism
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        logger.info(f"Initialized AudioLSTM: {input_size} -> {num_classes} classes")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling over sequence dimension
        pooled = torch.mean(attended, dim=1)
        
        # Classification
        out = self.dropout(pooled)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    
    def get_feature_size(self) -> int:
        """Return the expected input feature size."""
        return self.input_size
''',

            # Utilities
            "src/utils/__init__.py": '',
            
            "src/utils/logger.py": '''"""
Logging utilities for the Speech Emotion Recognition system.
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(name: str, log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """Setup a logger with console and file handlers."""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get an existing logger or create a basic one."""
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set up basic logging
    if not logger.handlers:
        setup_logger(name)
    
    return logger
''',

            "src/utils/config.py": '''"""
Configuration management utilities.
"""
import yaml
from pathlib import Path
from typing import Dict, Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file."""
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved configuration to {config_path}")
        
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        raise
''',

            # API
            "src/api/__init__.py": '',
            
            "src/api/main.py": '''"""
FastAPI application for real-time emotion recognition API.
"""
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.utils.logger import setup_logger, get_logger
from src.utils.config import load_config

logger = get_logger(__name__)

# Global model storage
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting up SER API...")
    
    # Load models here
    # app_state["model"] = load_models()
    
    yield
    
    # Shutdown
    logger.info("Shutting down SER API...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Speech Emotion Recognition API",
        description="Real-time multimodal emotion recognition system",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Speech Emotion Recognition API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


async def start_api(config: dict):
    """Start the API server."""
    setup_logger("SER_API")
    
    host = config.get("api", {}).get("host", "0.0.0.0")
    port = config.get("api", {}).get("port", 8000)
    
    logger.info(f"Starting API server on {host}:{port}")
    
    config_uvicorn = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="info",
        reload=False
    )
    
    server = uvicorn.Server(config_uvicorn)
    await server.serve()


if __name__ == "__main__":
    config = load_config("configs/model_config.yaml")
    asyncio.run(start_api(config))
''',

            # Scripts
            "scripts/setup_environment.py": '''#!/usr/bin/env python3
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
    print("\\nNext steps:")
    print("1. pip install -r requirements.txt")
    print("2. python scripts/download_datasets.py  # (when implemented)")
    print("3. python src/main.py --mode train")
    print("4. python src/main.py --mode api")
    
    return True

if __name__ == "__main__":
    setup_project()
''',

            "scripts/download_datasets.py": '''#!/usr/bin/env python3
"""
Dataset download script for Speech Emotion Recognition project.
"""
import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

def download_file(url: str, filename: str) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)
        
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return False

def download_datasets():
    """Download required datasets."""
    print("📊 Starting dataset downloads...")
    
    # Create data directories
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = {
        "RAVDESS": {
            "url": "https://example.com/ravdess.zip",  # Replace with actual URL
            "description": "Ryerson Audio-Visual Database of Emotional Speech and Song"
        },
        "TESS": {
            "url": "https://example.com/tess.zip",  # Replace with actual URL  
            "description": "Toronto Emotional Speech Set"
        }
    }
    
    print("⚠️  Manual download required!")
    print("\\nPlease manually download the following datasets:")
    print("\\n1. RAVDESS Dataset:")
    print("   - Visit: https://zenodo.org/record/1188976")
    print("   - Download and extract to: data/raw/ravdess/")
    print("\\n2. TESS Dataset:")
    print("   - Visit: https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi:10.5683/SP2/E8H2MF")
    print("   - Download and extract to: data/raw/tess/")
    print("\\n3. FER+ Dataset (optional):")
    print("   - Visit: https://github.com/Microsoft/FERPlus")
    print("   - Download and extract to: data/raw/fer_plus/")
    
    # Create sample structure
    for dataset in ["ravdess", "tess", "fer_plus"]:
        dataset_dir = data_dir / dataset
        dataset_dir.mkdir(exist_ok=True)
        (dataset_dir / "README.txt").write_text(f"Download {dataset.upper()} dataset here")
    
    print("\\n✅ Created dataset directory structure")
    print("📥 Please download datasets manually and place them in the appropriate directories")

if __name__ == "__main__":
    download_datasets()
''',

            # Test files
            "tests/__init__.py": '',
            
            "tests/test_audio/__init__.py": '',
            
            "tests/test_audio/test_processor.py": '''"""
Tests for audio processing module.
"""
import pytest
import numpy as np
from src.audio.processor import AudioProcessor


class TestAudioProcessor:
    """Test cases for AudioProcessor class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = AudioProcessor()
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        assert self.processor.sample_rate == 22050
        assert self.processor.n_mfcc == 13
    
    def test_feature_extraction(self):
        """Test feature extraction with dummy audio."""
        # Create dummy audio (1 second of random noise)
        dummy_audio = np.random.randn(22050)
        
        features = self.processor.extract_features(dummy_audio)
        
        # Check that all expected features are present
        expected_features = ['mfcc', 'pitch', 'energy', 'chroma', 'spectral_centroid', 'zcr']
        for feature in expected_features:
            assert feature in features
            assert isinstance(features[feature], np.ndarray)
    
    def test_preprocess_for_model(self):
        """Test model preprocessing."""
        dummy_audio = np.random.randn(22050)
        features = self.processor.extract_features(dummy_audio)
        
        processed = self.processor.preprocess_for_model(features)
        
        assert isinstance(processed, np.ndarray)
        assert len(processed.shape) == 1  # Should be 1D array
        assert processed.shape[0] > 0  # Should have some features
''',

            # Empty __init__.py files for other modules
            "src/vision/__init__.py": '',
            "src/data/__init__.py": '',
            "src/training/__init__.py": '',
            "src/api/routes/__init__.py": '',
            "src/api/models/__init__.py": '',
            "src/api/middleware/__init__.py": '',
            "src/realtime/__init__.py": '',
            "tests/test_vision/__init__.py": '',
            "tests/test_models/__init__.py": '',
            "tests/test_api/__init__.py": '',
            "configs/__init__.py": '',
        }
    
    def create_files(self) -> None:
        """Create all project files with their content."""
        print("📝 Creating project files...")
        
        file_contents = self.get_file_contents()
        
        for file_path, content in file_contents.items():
            full_path = self.base_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"  ✅ {file_path}")
    
    def generate_scaffold(self) -> None:
        """Generate the complete project scaffold."""
        print(f"🏗️  Generating Speech Emotion Recognition project: {self.project_name}")
        print("=" * 60)
        
        # Create directory structure
        self.create_directory_structure()
        
        # Create all files
        self.create_files()
        
        print("=" * 60)
        print("🎉 Project scaffold generated successfully!")
        print(f"📁 Project location: {self.base_path.absolute()}")
        print()
        print("🚀 Next steps:")
        print(f"1. cd {self.project_name}")
        print("2. python -m venv ser_env")
        print("3. source ser_env/bin/activate  # Windows: ser_env\\Scripts\\activate")
        print("4. pip install -r requirements.txt")
        print("5. python scripts/setup_environment.py")
        print("6. python scripts/download_datasets.py")
        print()
        print("🎯 Quick test:")
        print("python -c \"from src.audio.processor import AudioProcessor; print('✅ Setup successful!')\"")


def main():
    """Main function to generate the scaffold."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate SER project scaffold")
    parser.add_argument(
        "--name", 
        default="speech-emotion-recognition",
        help="Project name (default: speech-emotion-recognition)"
    )
    
    args = parser.parse_args()
    
    # Check if directory already exists
    if Path(args.name).exists():
        response = input(f"Directory '{args.name}' already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Generate scaffold
    scaffold = ProjectScaffold(args.name)
    scaffold.generate_scaffold()


if __name__ == "__main__":
    main()