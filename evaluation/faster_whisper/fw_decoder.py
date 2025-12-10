from evaluation.base.whisper_runner import WhisperDecoder
from faster_whisper import WhisperModel
import numpy as np


class FasterWhisperDecoder(WhisperDecoder):
    """
    Decoder wrapper for faster-whisper.

    Note: faster-whisper handles decoding internally in the transcribe() method.
    This wrapper provides compatibility with the existing interface but the actual
    decoding happens in FasterWhisperTranscriber for optimal performance.
    """

    # Model size to sequence length mapping
    MODEL_SEQUENCE_LENGTHS = {
        "tiny": 32,
        "tiny.en": 32,
        "base": 24,
        "base.en": 24,
        "small": 128,
        "small.en": 128,
        "medium": 128,
        "medium.en": 128,
        "large": 128,
        "large-v1": 128,
        "large-v2": 128,
        "large-v3": 128,
        "distil-small.en": 128,
        "distil-medium.en": 128,
        "distil-large-v2": 128,
        "distil-large-v3": 128,
    }

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        """
        Initialize the faster-whisper decoder.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large, or distil-*)
            device: Device to run on ("cpu" or "cuda")
            compute_type: Computation type ("int8", "float16", "float32")
        """
        # Don't call super().__init__() with path since we don't have a model file
        self.decoder_model_path = None
        self.backend = "faster-whisper"
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

        # Set sequence length based on model size
        self.decoding_sequence_length = self.MODEL_SEQUENCE_LENGTHS.get(model_size, 128)

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

    def decode(self, encoded_features):
        """
        Decode encoded features to text.

        Note: For interface compatibility only. In faster-whisper, use
        FasterWhisperTranscriber.transcribe() for actual transcription.

        Args:
            encoded_features: Encoded features (not used directly)

        Returns:
            Empty list (use transcriber for actual decoding)
        """
        # faster-whisper's transcribe method handles both encoding and decoding
        # This method exists for interface compatibility
        return []

    def set_model(self, model: WhisperModel):
        """Set a shared model instance (for use with transcriber)."""
        self._model = model
