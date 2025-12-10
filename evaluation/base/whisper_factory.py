from evaluation.onnxruntime.onnx_encoder import ONNXWhisperEncoder
from evaluation.onnxruntime.onnx_decoder import ONNXWhisperDecoder
from evaluation.hailo.hailo_sdk_encoder import HailoSdkWhisperEncoder
from evaluation.hailo.hailo_sdk_decoder import HailoSdkWhisperDecoder
from evaluation.faster_whisper.fw_encoder import FasterWhisperEncoder
from evaluation.faster_whisper.fw_decoder import FasterWhisperDecoder
from evaluation.faster_whisper.fw_transcriber import FasterWhisperTranscriber
import os


def get_backend_from_file_extension(model_path):
    root, ext = os.path.splitext(model_path)
    ext = ext[1:]
    if ext == "onnx":
        backend = "onnx"
    elif ext == "har":
        backend = "hailo"
    else:
        raise ValueError(f"Unsupported model extension: {ext}")

    return backend


def get_encoder(model_path, target="native"):
    """Returns the appropriate encoder based on the backend."""
    backend = get_backend_from_file_extension(model_path)
    if backend == "onnx":
        if target != "native":
            print(f"Selected target {target} for encoder will be ignored when using ONNXRuntime")
        return ONNXWhisperEncoder(model_path)
    elif backend == "hailo":
        print(f"Encoder target: {target}")
        return HailoSdkWhisperEncoder(model_path, target)
    else:
        raise ValueError(f"Unsupported encoder backend: {backend}")


def get_decoder(model_path, variant="tiny", target="native"):
    """Returns the appropriate decoder based on the backend."""
    backend = get_backend_from_file_extension(model_path)
    if backend == "onnx":
        if target != "native":
            print(f"Selected target {target} for decoder will be ignored when using ONNXRuntime")
        return ONNXWhisperDecoder(model_path, variant)
    elif backend == "hailo":
        print(f"Decoder target: {target}")
        return HailoSdkWhisperDecoder(model_path, variant, target)
    else:
        raise ValueError(f"Unsupported decoder backend: {backend}")


def get_faster_whisper_encoder(model_size="base", device="cpu", compute_type="int8"):
    """
    Returns a faster-whisper encoder wrapper.

    Args:
        model_size: Whisper model size (tiny, base, small, medium, large, or distil-*)
        device: Device to run on ("cpu" or "cuda")
        compute_type: Computation type ("int8", "float16", "float32")

    Returns:
        FasterWhisperEncoder instance
    """
    return FasterWhisperEncoder(model_size, device, compute_type)


def get_faster_whisper_decoder(model_size="base", device="cpu", compute_type="int8"):
    """
    Returns a faster-whisper decoder wrapper.

    Args:
        model_size: Whisper model size (tiny, base, small, medium, large, or distil-*)
        device: Device to run on ("cpu" or "cuda")
        compute_type: Computation type ("int8", "float16", "float32")

    Returns:
        FasterWhisperDecoder instance
    """
    return FasterWhisperDecoder(model_size, device, compute_type)


def get_faster_whisper_transcriber(
    model_size="base",
    device="cpu",
    compute_type="int8",
    download_root=None
):
    """
    Returns a faster-whisper transcriber for high-performance transcription.

    This is the recommended way to use faster-whisper as it provides:
    - 4-6x faster decoding via CTranslate2
    - Built-in VAD support
    - Streaming output
    - Support for all model sizes including distil-*

    Args:
        model_size: Whisper model size. Options:
            - Standard: tiny, base, small, medium, large, large-v2, large-v3
            - English-only: tiny.en, base.en, small.en, medium.en
            - Distil: distil-small.en, distil-medium.en, distil-large-v2, distil-large-v3
        device: Device to run on ("cpu" or "cuda")
        compute_type: Computation type:
            - "int8": Best for CPU (recommended for Raspberry Pi)
            - "float16": Good for GPU
            - "float32": Highest precision
        download_root: Optional path to download models to

    Returns:
        FasterWhisperTranscriber instance
    """
    return FasterWhisperTranscriber(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        download_root=download_root
    )
