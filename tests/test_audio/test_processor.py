"""
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
