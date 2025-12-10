"""
Real-time speech transcription using faster-whisper with VAD.

This module provides live transcription from the microphone with:
- Voice Activity Detection (VAD) for intelligent chunking
- Low-latency processing (processes speech as soon as silence is detected)
- Streaming output (text appears as it's transcribed)
- Support for all Whisper model sizes

Usage:
    python -m evaluation.live_transcribe --model small --vad
    python -m evaluation.live_transcribe --model base --min-silence 0.5
"""

import argparse
import sys
import time
import queue
import threading
import numpy as np
import sounddevice as sd
from typing import Optional

from evaluation.base.whisper_factory import get_faster_whisper_transcriber
from common.log_utils import logger


# Audio settings (Whisper expects 16kHz mono)
SAMPLE_RATE = 16000
CHANNELS = 1


class LiveTranscriber:
    """
    Real-time speech transcriber with VAD-based chunking.

    Uses faster-whisper for high-performance transcription and Silero-VAD
    for detecting speech boundaries to minimize latency.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = None,
        min_audio_length: float = 0.5,
        max_audio_length: float = 30.0,
        min_silence_duration: float = 0.3,
        speech_threshold: float = 0.5,
    ):
        """
        Initialize the live transcriber.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large, distil-*)
            device: Device to run on ("cpu" or "cuda")
            compute_type: Compute type ("int8", "float16", "float32")
            language: Language code or None for auto-detection
            min_audio_length: Minimum audio length to process (seconds)
            max_audio_length: Maximum audio length before forced processing (seconds)
            min_silence_duration: Minimum silence to trigger processing (seconds)
            speech_threshold: VAD speech detection threshold (0-1)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.min_audio_length = min_audio_length
        self.max_audio_length = max_audio_length
        self.min_silence_duration = min_silence_duration
        self.speech_threshold = speech_threshold

        # Initialize transcriber
        logger.info(f"Loading model: {model_size} (device={device}, compute_type={compute_type})")
        self.transcriber = get_faster_whisper_transcriber(
            model_size=model_size,
            device=device,
            compute_type=compute_type
        )
        logger.info("Model loaded successfully")

        # Audio buffer and state
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        self.is_speaking = False
        self.silence_start = None
        self.recording_start = None

        # Control flags
        self.running = False
        self.audio_queue = queue.Queue()

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input stream."""
        if status:
            logger.warning(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())

    def _get_audio_energy(self, audio_chunk: np.ndarray) -> float:
        """Calculate RMS energy of audio chunk."""
        return np.sqrt(np.mean(audio_chunk ** 2))

    def _simple_vad(self, audio_chunk: np.ndarray, energy_threshold: float = 0.01) -> bool:
        """
        Simple energy-based VAD for real-time processing.

        Note: faster-whisper's Silero VAD is used during transcription for accuracy,
        but this simple VAD provides quick speech/silence detection for chunking.
        """
        energy = self._get_audio_energy(audio_chunk)
        return energy > energy_threshold

    def _process_buffer(self) -> Optional[str]:
        """Process the current audio buffer and return transcription."""
        with self.buffer_lock:
            if not self.audio_buffer:
                return None

            audio_data = np.concatenate(self.audio_buffer, axis=0)
            self.audio_buffer = []

        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Check minimum length
        audio_duration = len(audio_data) / SAMPLE_RATE
        if audio_duration < self.min_audio_length:
            return None

        # Transcribe with VAD for accurate speech detection
        try:
            segments, info = self.transcriber.transcribe(
                audio_data,
                language=self.language,
                vad_filter=True,
                vad_parameters={
                    "threshold": self.speech_threshold,
                    "min_speech_duration_ms": 250,
                    "min_silence_duration_ms": int(self.min_silence_duration * 1000),
                    "speech_pad_ms": 200,
                },
                beam_size=3,  # Faster decoding
            )

            text = " ".join(seg["text"].strip() for seg in segments)
            return text.strip() if text.strip() else None

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

    def _process_audio_loop(self):
        """Main audio processing loop."""
        chunk_duration = 0.1  # 100ms chunks for responsiveness
        chunk_samples = int(SAMPLE_RATE * chunk_duration)

        while self.running:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Flatten if needed
            if audio_chunk.ndim > 1:
                audio_chunk = np.mean(audio_chunk, axis=1)

            current_time = time.time()
            is_speech = self._simple_vad(audio_chunk)

            with self.buffer_lock:
                self.audio_buffer.append(audio_chunk)
                buffer_duration = sum(len(c) for c in self.audio_buffer) / SAMPLE_RATE

            if is_speech:
                self.is_speaking = True
                self.silence_start = None
                if self.recording_start is None:
                    self.recording_start = current_time
            else:
                if self.is_speaking:
                    if self.silence_start is None:
                        self.silence_start = current_time
                    elif current_time - self.silence_start >= self.min_silence_duration:
                        # Silence detected after speech - process buffer
                        self.is_speaking = False
                        self.recording_start = None

                        text = self._process_buffer()
                        if text:
                            print(f"\n>> {text}")
                            sys.stdout.flush()

            # Force processing if buffer too long
            if buffer_duration >= self.max_audio_length:
                text = self._process_buffer()
                if text:
                    print(f"\n>> {text}")
                    sys.stdout.flush()
                self.is_speaking = False
                self.silence_start = None
                self.recording_start = None

    def run(self):
        """Start live transcription."""
        print("\n" + "=" * 60)
        print("Live Transcription - Press Ctrl+C to stop")
        print("=" * 60)
        print(f"Model: {self.model_size} | Device: {self.device}")
        print(f"Min silence: {self.min_silence_duration}s | Language: {self.language or 'auto'}")
        print("-" * 60)
        print("Speak now... (text will appear after pauses)\n")

        self.running = True

        # Start audio processing thread
        process_thread = threading.Thread(target=self._process_audio_loop, daemon=True)
        process_thread.start()

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                callback=self._audio_callback,
                blocksize=int(SAMPLE_RATE * 0.1),  # 100ms blocks
            ):
                while self.running:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\nStopping transcription...")
        finally:
            self.running = False
            process_thread.join(timeout=1.0)

        print("Transcription stopped.")


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time speech transcription with faster-whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with base model
    python -m evaluation.live_transcribe

    # Use small model for better accuracy
    python -m evaluation.live_transcribe --model small

    # Faster response with shorter silence threshold
    python -m evaluation.live_transcribe --model base --min-silence 0.3

    # English-only model for better English transcription
    python -m evaluation.live_transcribe --model small.en --language en
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        default="base",
        help="Whisper model size (tiny, base, small, medium, large, distil-*)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on (cpu recommended for Pi)"
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="int8",
        choices=["int8", "float16", "float32"],
        help="Compute type (int8 recommended for CPU)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g., 'en'). None for auto-detection."
    )
    parser.add_argument(
        "--min-silence",
        type=float,
        default=0.3,
        help="Minimum silence duration to trigger transcription (seconds)"
    )
    parser.add_argument(
        "--min-audio",
        type=float,
        default=0.5,
        help="Minimum audio length to process (seconds)"
    )
    parser.add_argument(
        "--max-audio",
        type=float,
        default=30.0,
        help="Maximum audio length before forced processing (seconds)"
    )

    return parser.parse_args()


def main():
    args = get_args()

    transcriber = LiveTranscriber(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        min_audio_length=args.min_audio,
        max_audio_length=args.max_audio,
        min_silence_duration=args.min_silence,
    )

    transcriber.run()


if __name__ == "__main__":
    main()
