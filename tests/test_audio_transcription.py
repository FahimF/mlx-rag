"""Integration tests for audio transcription functionality."""

import pytest
import tempfile
import wave
import numpy as np
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.mlx_gui.server import app
from src.mlx_gui.audio_manager import AudioManager, WhisperError


def get_test_wav_file():
    """Get path to the existing test.wav file."""
    test_wav_path = os.path.join(os.path.dirname(__file__), "test.wav")
    if not os.path.exists(test_wav_path):
        raise FileNotFoundError(f"test.wav not found at {test_wav_path}")
    return test_wav_path


def create_test_mp3_file():
    """Create a minimal test MP3 file name (without actual content)."""
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
        # Just create empty file with .mp3 extension for format testing
        f.write(b'fake mp3 content')
        return f.name


class TestAudioManager:
    """Test cases for the AudioManager class."""
    
    def test_audio_manager_initialization(self):
        """Test AudioManager initializes correctly."""
        manager = AudioManager()
        assert manager.default_model == "mlx-community/whisper-tiny"
        assert '.wav' in manager.supported_formats
        assert '.mp3' in manager.supported_formats
    
    def test_validate_audio_file_valid_formats(self):
        """Test validation of supported audio formats."""
        manager = AudioManager()
        
        # Test valid formats
        assert manager.validate_audio_file("test.wav") == True
        assert manager.validate_audio_file("test.mp3") == True
        assert manager.validate_audio_file("test.m4a") == True
        assert manager.validate_audio_file("test.flac") == True
        assert manager.validate_audio_file("test.ogg") == True
        assert manager.validate_audio_file("test.webm") == True
    
    def test_validate_audio_file_invalid_formats(self):
        """Test validation rejects unsupported formats."""
        manager = AudioManager()
        
        # Test invalid formats
        assert manager.validate_audio_file("test.txt") == False
        assert manager.validate_audio_file("test.py") == False
        assert manager.validate_audio_file("test.jpg") == False
        assert manager.validate_audio_file("test") == False
    
    def test_get_available_models(self):
        """Test getting list of available models."""
        manager = AudioManager()
        models = manager.get_available_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert "mlx-community/whisper-tiny" in models
        assert "mlx-community/whisper-base" in models
        assert "mlx-community/whisper-small" in models
    
    def test_get_supported_formats(self):
        """Test getting list of supported formats."""
        manager = AudioManager()
        formats = manager.get_supported_formats()
        
        assert isinstance(formats, list)
        assert '.wav' in formats
        assert '.mp3' in formats
        assert '.m4a' in formats
    
    @patch('mlx_whisper.transcribe')
    def test_transcribe_audio_success(self, mock_transcribe):
        """Test successful audio transcription."""
        manager = AudioManager()
        
        # Mock the MLX Whisper transcribe function
        mock_transcribe.return_value = {
            "text": "Hello, this is a test transcription.",
            "language": "en",
            "segments": []
        }
        
        # Use existing test.wav file
        audio_file = get_test_wav_file()
        
        result = manager.transcribe_audio(audio_file)
        
        assert "text" in result
        assert result["text"] == "Hello, this is a test transcription."
        assert result["language"] == "en"
        assert result["model"] == "mlx-community/whisper-tiny"
        
        # Verify mlx_whisper.transcribe was called correctly
        mock_transcribe.assert_called_once_with(
            audio_file,
            path_or_hf_repo="mlx-community/whisper-tiny",
            language=None,
            word_timestamps=False
        )
    
    @patch('mlx_whisper.transcribe')
    def test_transcribe_audio_with_language(self, mock_transcribe):
        """Test audio transcription with specified language."""
        manager = AudioManager()
        
        mock_transcribe.return_value = {
            "text": "Hola, esta es una prueba.",
            "language": "es"
        }
        
        audio_file = get_test_wav_file()
        
        result = manager.transcribe_audio(
            audio_file, 
            language="es",
            word_timestamps=True
        )
        
        assert result["text"] == "Hola, esta es una prueba."
        assert result["language"] == "es"
        
        # Verify correct parameters were passed
        mock_transcribe.assert_called_once_with(
            audio_file,
            path_or_hf_repo="mlx-community/whisper-tiny",
            language="es",
            word_timestamps=True
        )
    
    def test_transcribe_audio_file_not_found(self):
        """Test transcription with non-existent file."""
        manager = AudioManager()
        
        with pytest.raises(WhisperError, match="Audio file not found"):
            manager.transcribe_audio("/non/existent/file.wav")
    
    def test_transcribe_audio_unsupported_format(self):
        """Test transcription with unsupported file format."""
        manager = AudioManager()
        
        # Create temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b'not audio content')
            temp_file = f.name
        
        try:
            with pytest.raises(WhisperError, match="Unsupported audio format"):
                manager.transcribe_audio(temp_file)
        finally:
            os.unlink(temp_file)
    
    @patch('mlx_whisper.transcribe')
    def test_transcribe_audio_mlx_error(self, mock_transcribe):
        """Test transcription when MLX Whisper fails."""
        manager = AudioManager()
        
        # Mock MLX Whisper to raise an exception
        mock_transcribe.side_effect = Exception("MLX processing failed")
        
        audio_file = get_test_wav_file()
        
        with pytest.raises(WhisperError, match="Transcription failed: MLX processing failed"):
            manager.transcribe_audio(audio_file)


class TestAudioTranscriptionAPI:
    """Test cases for the audio transcription API endpoints."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    @patch('src.mlx_gui.queued_inference.queued_transcribe_audio')
    @patch('src.mlx_gui.server.get_model_manager')
    @patch('src.mlx_gui.server.get_db_session')
    def test_transcription_endpoint_success(self, mock_db, mock_model_manager, mock_queued_transcribe):
        """Test successful transcription via API endpoint."""
        # Mock database session and model
        mock_db.return_value = MagicMock()
        mock_model_record = MagicMock()
        mock_model_record.name = "mlx-community/whisper-tiny"
        mock_model_record.path = "/fake/path"
        mock_db.return_value.query.return_value.filter.return_value.first.return_value = mock_model_record
        
        # Mock model manager
        mock_manager = MagicMock()
        mock_loaded_model = MagicMock()
        mock_loaded_model.mlx_wrapper.transcribe_audio = MagicMock()
        mock_manager.get_model_for_inference.return_value = mock_loaded_model
        mock_model_manager.return_value = mock_manager
        
        # Mock queued transcription
        mock_queued_transcribe.return_value = {
            "text": "This is a test transcription."
        }
        
        # Use existing test.wav file
        audio_file = get_test_wav_file()
        
        with open(audio_file, 'rb') as f:
            response = self.client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", f, "audio/wav")},
                data={
                    "model": "whisper-tiny",
                    "response_format": "json"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert data["text"] == "This is a test transcription."
    
    def test_transcription_endpoint_no_file(self):
        """Test transcription endpoint with no file provided."""
        response = self.client.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-tiny"}
        )
        
        assert response.status_code == 422  # Validation error
    
    @patch('src.mlx_gui.server.get_db_session')
    def test_transcription_endpoint_model_not_found(self, mock_db):
        """Test transcription endpoint with non-existent model."""
        # Mock database to return no model
        mock_db.return_value = MagicMock()
        mock_db.return_value.query.return_value.filter.return_value.first.return_value = None
        
        audio_file = get_test_wav_file()
        
        with open(audio_file, 'rb') as f:
            response = self.client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"model": "non-existent-model"}
            )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    @patch('src.mlx_gui.queued_inference.queued_transcribe_audio')
    @patch('src.mlx_gui.server.get_model_manager')
    @patch('src.mlx_gui.server.get_db_session')
    def test_transcription_endpoint_text_format(self, mock_db, mock_model_manager, mock_queued_transcribe):
        """Test transcription endpoint with text response format."""
        # Setup mocks (same as success test)
        mock_db.return_value = MagicMock()
        mock_model_record = MagicMock()
        mock_model_record.name = "mlx-community/whisper-tiny"
        mock_db.return_value.query.return_value.filter.return_value.first.return_value = mock_model_record
        
        mock_manager = MagicMock()
        mock_loaded_model = MagicMock()
        mock_loaded_model.mlx_wrapper.transcribe_audio = MagicMock()
        mock_manager.get_model_for_inference.return_value = mock_loaded_model
        mock_model_manager.return_value = mock_manager
        
        mock_queued_transcribe.return_value = {
            "text": "Plain text transcription."
        }
        
        audio_file = get_test_wav_file()
        
        with open(audio_file, 'rb') as f:
            response = self.client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", f, "audio/wav")},
                data={
                    "model": "whisper-tiny",
                    "response_format": "text"
                }
            )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        assert response.text == "Plain text transcription."
    
    @patch('src.mlx_gui.queued_inference.queued_transcribe_audio')
    @patch('src.mlx_gui.server.get_model_manager')
    @patch('src.mlx_gui.server.get_db_session')
    def test_transcription_endpoint_with_language(self, mock_db, mock_model_manager, mock_queued_transcribe):
        """Test transcription endpoint with language parameter."""
        # Setup mocks
        mock_db.return_value = MagicMock()
        mock_model_record = MagicMock()
        mock_model_record.name = "mlx-community/whisper-tiny"
        mock_db.return_value.query.return_value.filter.return_value.first.return_value = mock_model_record
        
        mock_manager = MagicMock()
        mock_loaded_model = MagicMock()
        mock_loaded_model.mlx_wrapper.transcribe_audio = MagicMock()
        mock_manager.get_model_for_inference.return_value = mock_loaded_model
        mock_model_manager.return_value = mock_manager
        
        mock_queued_transcribe.return_value = {
            "text": "Bonjour le monde."
        }
        
        audio_file = get_test_wav_file()
        
        with open(audio_file, 'rb') as f:
            response = self.client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", f, "audio/wav")},
                data={
                    "model": "whisper-small",
                    "language": "fr",
                    "response_format": "json"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Bonjour le monde."
        
        # Verify queued_transcribe was called with correct language
        mock_queued_transcribe.assert_called_once()
        call_args = mock_queued_transcribe.call_args
        assert call_args[1]["language"] == "fr"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])