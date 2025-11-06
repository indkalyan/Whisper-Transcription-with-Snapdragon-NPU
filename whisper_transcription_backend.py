#!/usr/bin/env python3
import os
import sys
import logging
from argparse import Namespace
from pathlib import Path
import numpy as np
import librosa
from qai_hub_models.models._shared.hf_whisper.app import HfWhisperApp
from qai_hub_models.utils.onnx_torch_wrapper import OnnxModelTorchWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WhisperTranscriber:

    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load and preprocess audio file
        
        Args:
            audio_path: Path to audio file (MP3, WAV, etc.)
            
        Returns:
            Preprocessed audio array
        """
        logger.info(f"Loading audio: {audio_path}")

        try:
            # Load audio file and resample to 16kHz
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            logger.info(f"  Duration: {len(audio) / self.sample_rate:.2f}s")
            logger.info(f"  Sample rate: {sr} Hz")

            return audio
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise

    def audio_to_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert audio to mel spectrogram (Whisper preprocessing)
        
        Args:
            audio: Audio waveform
            
        Returns:
            Mel spectrogram
        """
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize
        log_mel_spec = (log_mel_spec + 40) / 40  # Approximate normalization

        return log_mel_spec.astype(np.float32)


def main():
    """Main CLI interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description="German Audio Transcription using Whisper on Snapdragon X Elite"
    )
    parser.add_argument(
        '--audio_path',
         default='C:\\Users\\indka\\Music\\pons\\c05.mp3',
        # default='C:\\Users\\indka\\Music\\Story-kaspar',


        help='Path to audio file or directory containing MP3 files'
    )
    parser.add_argument(
        '--encoder_path',
        default='C:\\Users\\indka\\PycharmProjects\\ai-hub-apps\\build\\whisper_base_float\\precompiled\\qualcomm-snapdragon-x-elite\\HfWhisperEncoder\\model.onnx',
        help='Path to encoder ONNX model'
    )
    parser.add_argument(
        '--decoder_path',
        default='C:\\Users\\indka\\PycharmProjects\\ai-hub-apps\\build\\whisper_base_float\\precompiled\\qualcomm-snapdragon-x-elite\\HfWhisperDecoder\\model.onnx',
        help='Path to decoder ONNX model'
    )
    parser.add_argument(
        '--output-dir',
        default='C:\\Users\\indka\\Music\\pons',
        help='Output directory for transcriptions'
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large", "large-v3-turbo"],
        help="Size of the model being run, corresponding to a specific model checkpoint on huggingface.",
    )

    args = parser.parse_args()

    print("Loading model...")
    app = HfWhisperApp(
        OnnxModelTorchWrapper.OnNPU(args.encoder_path),
        OnnxModelTorchWrapper.OnNPU(args.decoder_path),
        f"openai/whisper-{args.model_size}",
    )

    # Get audio files
    audio_path = Path(args.audio_path)

    if audio_path.is_file():
        # Single file
        audio_files = [str(audio_path)]
    elif audio_path.is_dir():
        # Directory - find all MP3 files
        audio_files = list(audio_path.glob("*.mp3"))
        audio_files = [str(f) for f in audio_files]

        if not audio_files:
            logger.error(f"❌ No MP3 files found in {audio_path}")
            sys.exit(1)
    else:
        logger.error(f"❌ Invalid path: {audio_path}")
        sys.exit(1)

    logger.info(f"Found {len(audio_files)} audio file(s)")

    for audio_file in audio_files:
        logger.info(f"\nProcessing: {audio_file}")
        transcription = app.transcribe(audio_file)
        # write transcription to file
        writeTranscriptionToFile(args, audio_file, transcription)


def writeTranscriptionToFile(args: Namespace, audio_file: str, transcription: str):
    audio_name = Path(audio_file).stem
    output_file = Path(args.output_dir) / f"{audio_name}_transcript.txt"
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(transcription)
    logger.info(f"  ✓ Saved to: {output_file}")


if __name__ == "__main__":
    main()
