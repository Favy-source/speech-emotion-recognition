"""
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
