from evaluation.base.whisper_runner import WhisperEncoder
from faster_whisper import WhisperModel
import numpy as np


class FasterWhisperEncoder(WhisperEncoder):
    """
    Encoder wrapper for faster-whisper.

    Note: faster-whisper handles encoding internally in the transcribe() method.
    This wrapper provides compatibility with the existing interface but the actual
    encoding happens in FasterWhisperTranscriber for optimal performance.
    """

    # Model size to input length mapping (in seconds)
    MODEL_INPUT_LENGTHS = {
        "tiny": 10,
        "tiny.en": 10,
        "base": 5,
        "base.en": 5,
        "small": 30,
        "small.en": 30,
        "medium": 30,
        "medium.en": 30,
        "large": 30,
        "large-v1": 30,
        "large-v2": 30,
        "large-v3": 30,
        "distil-small.en": 30,
        "distil-medium.en": 30,
        "distil-large-v2": 30,
        "distil-large-v3": 30,
    }

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        """
        Initialize the faster-whisper encoder.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large, or distil-*)
            device: Device to run on ("cpu" or "cuda")
            compute_type: Computation type ("int8", "float16", "float32")
        """
        # Don't call super().__init__() with path since we don't have a model file
        self.encoder_model_path = None
        self.backend = "faster-whisper"
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

        # Set input length based on model size
        self.input_audio_length = self.MODEL_INPUT_LENGTHS.get(model_size, 30)

        # The model is loaded lazily or shared with the transcriber
        self._model = None

    def _get_model(self):
        """Lazy load the model."""
        if self._model is None:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
        return self._model

    def encode(self, input_mel):
        """
        Encode mel spectrogram to features.

        Note: For faster-whisper, encoding is typically done internally during
        transcription. This method is provided for interface compatibility.

        Args:
            input_mel: Mel spectrogram input (not used directly in faster-whisper)

        Returns:
            The input as-is, since faster-whisper handles encoding internally.
        """
        # faster-whisper's encode method expects audio, not mel spectrogram
        # Return input as-is for interface compatibility
        return input_mel

    def get_input_length(self):
        """Return the expected input audio length in seconds."""
        return self.input_audio_length

    def set_model(self, model: WhisperModel):
        """Set a shared model instance (for use with transcriber)."""
        self._model = model
