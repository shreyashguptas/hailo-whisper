# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hailo Whisper is a toolkit for converting and deploying OpenAI's Whisper speech recognition model on Hailo-8 and Hailo-10H AI accelerators. It handles the complete pipeline: PyTorch → ONNX export → Hailo format (HAR/HEF) conversion → evaluation with WER metrics.

**Supported Whisper variants:** tiny, tiny.en, base, base.en

## Common Commands

### Setup
```bash
python3 setup.py
source whisper_env/bin/activate
```

### Export Model to ONNX
```bash
python3 -m export.export_whisper_model --variant tiny
```

### Convert to Hailo Format
```bash
# Encoder
python3 -m conversion.convert_whisper_encoder ONNX_ENCODER_PATH --variant tiny --hw-arch hailo8

# Decoder (requires calibration set first)
python3 -m conversion.create_decoder_calib_set --encoder ENCODER_HAR --decoder DECODER_ONNX --variant tiny
python3 -m conversion.convert_whisper_decoder ONNX_DECODER_PATH --variant tiny --hw-arch hailo8
```

### Evaluation
```bash
# Original ONNX/Hailo backend
python3 -m evaluation.testbench --variant tiny --encoder encoder.onnx --decoder decoder.onnx

# faster-whisper backend (4-6x faster, recommended)
python3 -m evaluation.testbench --backend faster-whisper --model small audio.wav

# faster-whisper with VAD and streaming
python3 -m evaluation.testbench --backend faster-whisper --model base --vad --stream audio.wav

# Live transcription from microphone
python3 -m evaluation.live_transcribe --model small --min-silence 0.3

# Batch evaluation with WER
python3 -m evaluation.evaluation --encoder encoder.har --decoder decoder.har --variant tiny --num-samples 100
```

## Architecture

### Pipeline Flow
```
PyTorch Whisper → ONNX (encoder + decoder) → Calibration → HAR → HEF
```

### Key Directories
- `export/` - PyTorch to ONNX export
- `conversion/` - ONNX to Hailo format (HAR/HEF) conversion with calibration
- `evaluation/` - Inference and WER evaluation with pluggable backends (ONNX Runtime, Hailo)
- `common/` - Shared audio preprocessing, Mel spectrogram generation, utilities
- `third_party/whisper/` - Git submodule of OpenAI Whisper (patched for Hailo compatibility)

### Backend System
The evaluation code uses a factory pattern (`evaluation/base/whisper_factory.py`) to create encoder/decoder instances:
- `evaluation/onnxruntime/` - ONNX Runtime backend
- `evaluation/hailo/` - Hailo SDK backend
- `evaluation/faster_whisper/` - faster-whisper backend (CTranslate2-based, 4-6x faster)

### faster-whisper Backend
The faster-whisper backend provides significant performance improvements:
- **4-6x faster** decoding via CTranslate2 optimization
- **All model sizes**: tiny, base, small, medium, large, distil-*
- **Built-in VAD**: Voice Activity Detection for intelligent chunking
- **Streaming**: Word-level timestamps, real-time output
- **CPU optimized**: INT8 quantization for Raspberry Pi

### Audio Processing
- 16kHz sample rate, 80-channel Mel spectrograms
- Configurable input lengths: 10s for tiny, 5s for base variants
- LibriSpeech dev-clean dataset used for calibration (in `audio/`)

## Dependencies

- Python 3.10 or 3.11
- Hailo Dataflow Compiler (v3.x for Hailo-8/8L, v5.x for Hailo-10H)
- ffmpeg, libportaudio2
- Key packages: torch, onnx, onnxruntime, transformers, jiwer
