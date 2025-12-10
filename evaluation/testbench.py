from evaluation.base.whisper_factory import (
    get_encoder, get_decoder, get_faster_whisper_transcriber
)
from common.preprocessing import preprocess, improve_input_audio
import whisper
import argparse
import os
from common.log_utils import logger


# All supported model variants for faster-whisper
FASTER_WHISPER_MODELS = [
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large", "large-v1", "large-v2", "large-v3",
    "distil-small.en", "distil-medium.en", "distil-large-v2", "distil-large-v3",
]

# Original ONNX/Hailo variants
ONNX_HAILO_VARIANTS = ["tiny", "tiny.en", "base", "base.en"]


def get_args():
    """
    Initialize and run the argument parser.

    Return:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Whisper inference testbench with multiple backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Original ONNX/Hailo backend
  python -m evaluation.testbench --variant tiny --encoder encoder.onnx --decoder decoder.onnx

  # faster-whisper backend (recommended for speed)
  python -m evaluation.testbench --backend faster-whisper --model small audio.wav

  # faster-whisper with VAD and streaming
  python -m evaluation.testbench --backend faster-whisper --model base --vad --stream audio.wav
        """
    )

    # Backend selection
    parser.add_argument(
        "--backend",
        type=str,
        default="onnx-hailo",
        choices=["onnx-hailo", "faster-whisper"],
        help="Inference backend: 'onnx-hailo' (original) or 'faster-whisper' (4-6x faster)"
    )

    # Original ONNX/Hailo arguments
    parser.add_argument(
        "--variant",
        type=str,
        choices=ONNX_HAILO_VARIANTS,
        help="Whisper model variant for ONNX/Hailo backend"
    )
    parser.add_argument(
        "--encoder",
        type=str,
        help="Whisper encoder path (ONNX / HAR) for ONNX/Hailo backend"
    )
    parser.add_argument(
        "--decoder",
        type=str,
        help="Whisper decoder path (ONNX / HAR) for ONNX/Hailo backend"
    )
    parser.add_argument(
        "--encoder-target",
        type=str,
        default="native",
        choices=["native", "quantized", "hw"],
        help="Target for encoder (ONNX/Hailo backend)"
    )
    parser.add_argument(
        "--decoder-target",
        type=str,
        default="native",
        choices=["native", "quantized", "hw"],
        help="Target for decoder (ONNX/Hailo backend)"
    )

    # faster-whisper arguments
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=FASTER_WHISPER_MODELS,
        help="Model size for faster-whisper backend"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for faster-whisper (cpu recommended for Pi)"
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="int8",
        choices=["int8", "float16", "float32"],
        help="Compute type for faster-whisper (int8 recommended for CPU)"
    )
    parser.add_argument(
        "--vad",
        action="store_true",
        help="Enable Voice Activity Detection (faster-whisper only)"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming output (faster-whisper only)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g., 'en', 'es'). None for auto-detection."
    )

    # Common arguments
    parser.add_argument(
        "--audio-path",
        type=str,
        default=None,
        help="Audio file path for the test (if None, uses a default file)"
    )
    # Positional audio path for convenience
    parser.add_argument(
        "audio",
        type=str,
        nargs="?",
        help="Audio file path (alternative to --audio-path)"
    )

    return parser.parse_args()


def run_onnx_hailo(args):
    """Run inference using the original ONNX/Hailo backend."""
    if not args.variant or not args.encoder or not args.decoder:
        raise ValueError(
            "ONNX/Hailo backend requires --variant, --encoder, and --decoder arguments"
        )

    whisper_encoder = get_encoder(args.encoder, target=args.encoder_target)
    chunk_length = whisper_encoder.get_input_length()
    whisper_decoder = get_decoder(args.decoder, variant=args.variant, target=args.decoder_target)

    is_nhwc = whisper_encoder.backend != "onnx"

    audio_path = args.audio or args.audio_path
    if audio_path is None:
        audio_path = "audio/dev-clean/LibriSpeech/dev-clean/84/121123/84-121123-0020.flac"
        logger.warning(f"No audio path provided, using default: {audio_path}")
    elif not os.path.exists(audio_path):
        raise FileNotFoundError(f"Provided audio file not found: {audio_path}")

    # Load and preprocess audio
    audio = whisper.load_audio(audio_path)
    audio, start_time = improve_input_audio(audio, vad=False, low_audio_gain=True)

    mel_spectrograms = preprocess(
        audio=audio, is_nhwc=is_nhwc,
        chunk_length=chunk_length, chunk_offset=start_time
    )

    for mel in mel_spectrograms:
        encoded_features = whisper_encoder.encode(mel)
        transcription = whisper_decoder.decode(encoded_features)
        print(transcription)


def run_faster_whisper(args):
    """Run inference using the faster-whisper backend."""
    audio_path = args.audio or args.audio_path
    if audio_path is None:
        audio_path = "audio/dev-clean/LibriSpeech/dev-clean/84/121123/84-121123-0020.flac"
        logger.warning(f"No audio path provided, using default: {audio_path}")
    elif not os.path.exists(audio_path):
        raise FileNotFoundError(f"Provided audio file not found: {audio_path}")

    logger.info(f"Loading faster-whisper model: {args.model}")
    logger.info(f"Device: {args.device}, Compute type: {args.compute_type}")

    transcriber = get_faster_whisper_transcriber(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type
    )

    if args.stream:
        # Streaming mode - output segments as they're ready
        logger.info("Streaming transcription...")
        for segment in transcriber.transcribe_streaming(
            audio_path,
            language=args.language,
            vad_filter=args.vad,
            word_timestamps=True
        ):
            print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")
    else:
        # Batch mode
        segments, info = transcriber.transcribe(
            audio_path,
            language=args.language,
            vad_filter=args.vad,
            word_timestamps=args.vad  # Word timestamps useful with VAD
        )

        logger.info(f"Detected language: {info['language']} (probability: {info['language_probability']:.2f})")
        logger.info(f"Audio duration: {info['duration']:.2f}s")

        for segment in segments:
            print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")


def main():
    args = get_args()

    if args.backend == "faster-whisper":
        run_faster_whisper(args)
    else:
        run_onnx_hailo(args)


if __name__ == "__main__":
    main()
