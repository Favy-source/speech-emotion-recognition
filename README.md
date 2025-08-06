# Speech Emotion Recognition (SER) System

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
source ser_env/bin/activate  # Windows: ser_env\Scripts\activate

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
