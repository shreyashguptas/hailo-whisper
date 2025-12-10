from faster_whisper import WhisperModel
from typing import Generator, List, Optional, Tuple, Union
import numpy as np


class FasterWhisperTranscriber:
    """
    High-performance transcriber using faster-whisper (CTranslate2).

    Supports:
    - All Whisper model sizes (tiny, base, small, medium, large)
    - Distil-Whisper models for better speed/quality trade-off
    - Voice Activity Detection (VAD) for intelligent chunking
    - Streaming output with word-level timestamps
    - INT8 quantization for CPU optimization (good for Raspberry Pi)
    """

    SUPPORTED_MODELS = [
        "tiny", "tiny.en",
        "base", "base.en",
        "small", "small.en",
        "medium", "medium.en",
        "large", "large-v1", "large-v2", "large-v3",
        "distil-small.en", "distil-medium.en", "distil-large-v2", "distil-large-v3",
    ]

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: Optional[str] = None,
    ):
        """
        Initialize the transcriber.

        Args:
            model_size: Whisper model size. Options:
                - Standard: tiny, base, small, medium, large, large-v2, large-v3
                - English-only: tiny.en, base.en, small.en, medium.en
                - Distil: distil-small.en, distil-medium.en, distil-large-v2, distil-large-v3
            device: Device to run on ("cpu" or "cuda")
            compute_type: Computation type:
                - "int8": Best for CPU, good speed/quality (recommended for Pi)
                - "float16": Good for GPU
                - "float32": Highest precision, slowest
            download_root: Optional path to download models to
        """
        if model_size not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model size: {model_size}. "
                f"Supported: {self.SUPPORTED_MODELS}"
            )

        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

        # Load the model
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
        )

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: int = 5,
        vad_filter: bool = False,
        vad_parameters: Optional[dict] = None,
        word_timestamps: bool = False,
        initial_prompt: Optional[str] = None,
    ) -> Tuple[List[dict], dict]:
        """
        Transcribe audio to text.

        Args:
            audio: Audio file path or numpy array (16kHz mono)
            language: Language code (e.g., "en", "es"). None for auto-detection.
            task: "transcribe" or "translate" (to English)
            beam_size: Beam size for decoding (higher = better quality, slower)
            vad_filter: Enable Voice Activity Detection to skip silence
            vad_parameters: VAD parameters dict:
                - threshold: Speech probability threshold (default: 0.5)
                - min_speech_duration_ms: Min speech segment (default: 250)
                - min_silence_duration_ms: Min silence to split (default: 2000)
                - speech_pad_ms: Padding around speech (default: 400)
            word_timestamps: Return word-level timestamps
            initial_prompt: Optional prompt to guide transcription

        Returns:
            Tuple of (segments_list, info_dict)
            - segments_list: List of segment dicts with keys:
                - text: Transcribed text
                - start: Start time in seconds
                - end: End time in seconds
                - words: List of word dicts (if word_timestamps=True)
            - info_dict: Transcription info with keys:
                - language: Detected/specified language
                - language_probability: Detection confidence
                - duration: Audio duration in seconds
        """
        # Set default VAD parameters for low-latency operation
        if vad_filter and vad_parameters is None:
            vad_parameters = {
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 300,  # Shorter for faster response
                "speech_pad_ms": 200,
            }

        segments, info = self.model.transcribe(
            audio,
            language=language,
            task=task,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
        )

        # Convert generator to list of dicts
        segments_list = []
        for segment in segments:
            seg_dict = {
                "text": segment.text,
                "start": segment.start,
                "end": segment.end,
            }
            if word_timestamps and segment.words:
                seg_dict["words"] = [
                    {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                    for w in segment.words
                ]
            segments_list.append(seg_dict)

        info_dict = {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
        }

        return segments_list, info_dict

    def transcribe_streaming(
        self,
        audio: Union[str, np.ndarray],
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: int = 5,
        vad_filter: bool = True,
        word_timestamps: bool = True,
    ) -> Generator[dict, None, None]:
        """
        Stream transcription results as they become available.

        Yields segments as soon as they're ready, enabling real-time display.

        Args:
            audio: Audio file path or numpy array (16kHz mono)
            language: Language code or None for auto-detection
            task: "transcribe" or "translate"
            beam_size: Beam size for decoding
            vad_filter: Enable VAD (recommended for streaming)
            word_timestamps: Include word-level timestamps

        Yields:
            Segment dicts with text, start, end, and optionally words
        """
        vad_parameters = {
            "threshold": 0.5,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 300,
            "speech_pad_ms": 200,
        }

        segments, info = self.model.transcribe(
            audio,
            language=language,
            task=task,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters,
            word_timestamps=word_timestamps,
        )

        # Yield segments as they're generated (streaming)
        for segment in segments:
            seg_dict = {
                "text": segment.text,
                "start": segment.start,
                "end": segment.end,
            }
            if word_timestamps and segment.words:
                seg_dict["words"] = [
                    {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                    for w in segment.words
                ]
            yield seg_dict

    def get_full_text(
        self,
        audio: Union[str, np.ndarray],
        language: Optional[str] = None,
        vad_filter: bool = False,
    ) -> str:
        """
        Simple helper to get just the transcribed text.

        Args:
            audio: Audio file path or numpy array
            language: Language code or None for auto-detection
            vad_filter: Enable VAD to skip silence

        Returns:
            Full transcribed text as a single string
        """
        segments, _ = self.transcribe(
            audio,
            language=language,
            vad_filter=vad_filter,
        )
        return " ".join(seg["text"].strip() for seg in segments)

    @staticmethod
    def get_available_models() -> List[str]:
        """Return list of supported model sizes."""
        return FasterWhisperTranscriber.SUPPORTED_MODELS.copy()
