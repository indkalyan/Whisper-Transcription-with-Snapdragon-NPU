"""
Real-time Speech Translation: English to German
Two approaches: SeamlessM4T (end-to-end) and Whisper+NLLB+TTS pipeline
"""

import torch
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import queue
import threading

# =============================================================================
# APPROACH 1: SeamlessM4T v2 (End-to-End Speech-to-Speech)
# =============================================================================

def setup_seamless_m4t():
    """Initialize SeamlessM4T v2 model"""
    from transformers import AutoProcessor, SeamlessM4Tv2Model
    
    print("Loading SeamlessM4T v2 model...")
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
    
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    return processor, model, device

def translate_audio_seamless(audio_array, sample_rate, processor, model, device):
    """
    Translate English speech to German speech using SeamlessM4T
    
    Args:
        audio_array: numpy array of audio samples
        sample_rate: sample rate (typically 16000)
        processor: SeamlessM4T processor
        model: SeamlessM4T model
        device: cuda or cpu
    
    Returns:
        translated_audio: numpy array of German speech
        sample_rate: output sample rate
    """
    # Process input audio
    audio_inputs = processor(
        audios=audio_array, 
        sampling_rate=sample_rate, 
        return_tensors="pt"
    ).to(device)
    
    # Generate German speech directly from English speech
    # src_lang="eng", tgt_lang="deu" for English to German
    output_tokens = model.generate(
        **audio_inputs,
        tgt_lang="deu",  # German
        generate_speech=True
    )
    
    # Extract audio waveform
    translated_audio = output_tokens[0].cpu().numpy().squeeze()
    
    return translated_audio, model.config.sampling_rate


# =============================================================================
# APPROACH 2: Whisper + NLLB + TTS Pipeline (More Control)
# =============================================================================

def setup_pipeline():
    """Initialize Whisper, NLLB, and TTS models"""
    from transformers import (
        AutoModelForSpeechSeq2Seq, 
        AutoProcessor as WhisperProcessor,
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        VitsModel,
        VitsTokenizer
    )
    
    print("Loading Whisper model...")
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3")
    
    print("Loading NLLB translation model...")
    nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    
    print("Loading German TTS model...")
    tts_tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-deu")
    tts_model = VitsModel.from_pretrained("facebook/mms-tts-deu")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model = whisper_model.to(device)
    nllb_model = nllb_model.to(device)
    tts_model = tts_model.to(device)
    
    return {
        "whisper_processor": whisper_processor,
        "whisper_model": whisper_model,
        "nllb_tokenizer": nllb_tokenizer,
        "nllb_model": nllb_model,
        "tts_tokenizer": tts_tokenizer,
        "tts_model": tts_model,
        "device": device
    }

def translate_audio_pipeline(audio_array, sample_rate, models):
    """
    Translate English speech to German speech using pipeline
    
    Steps:
    1. Speech to Text (Whisper)
    2. Text Translation (NLLB)
    3. Text to Speech (MMS-TTS)
    """
    device = models["device"]
    
    # Step 1: Speech to Text (English)
    print("Transcribing English speech...")
    inputs = models["whisper_processor"](
        audio_array, 
        sampling_rate=sample_rate, 
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        predicted_ids = models["whisper_model"].generate(**inputs)
    
    english_text = models["whisper_processor"].batch_decode(
        predicted_ids, 
        skip_special_tokens=True
    )[0]
    print(f"English: {english_text}")
    
    # Step 2: Translate to German
    print("Translating to German...")
    inputs = models["nllb_tokenizer"](
        english_text, 
        return_tensors="pt"
    ).to(device)

    if hasattr(models["nllb_tokenizer"], "src_lang"):
        models["nllb_tokenizer"].src_lang = "eng_Latn"
    
    translated_tokens = models["nllb_model"].generate(
        **inputs,
        forced_bos_token_id=models["nllb_tokenizer"].convert_tokens_to_ids("deu_Latn"),
        max_length=200
    )
    
    german_text = models["nllb_tokenizer"].batch_decode(
        translated_tokens, 
        skip_special_tokens=True
    )[0]
    print(f"German: {german_text}")
    
    # Step 3: Text to Speech (German)
    print("Generating German speech...")
    tts_inputs = models["tts_tokenizer"](german_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = models["tts_model"](**tts_inputs)
    
    german_audio = output.waveform[0].cpu().numpy()
    
    return german_audio, models["tts_model"].config.sampling_rate, english_text, german_text


# =============================================================================
# Audio Recording and Playback
# =============================================================================

def record_audio(duration=5, sample_rate=16000):
    """Record audio from microphone"""
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(
        int(duration * sample_rate), 
        samplerate=sample_rate, 
        channels=1,
        dtype='float32'
    )
    sd.wait()
    print("Recording finished!")
    return audio.squeeze()

def play_audio(audio_array, sample_rate):
    """Play audio through speakers"""
    print("Playing translated audio...")
    sd.play(audio_array, sample_rate)
    sd.wait()
    print("Playback finished!")

def save_audio(audio_array, sample_rate, filename):
    """Save audio to file"""
    # Convert to int16 for WAV format
    audio_int16 = np.int16(audio_array / np.max(np.abs(audio_array)) * 32767)
    write(filename, sample_rate, audio_int16)
    print(f"Audio saved to {filename}")


# =============================================================================
# Real-time Streaming (Advanced)
# =============================================================================

class RealtimeTranslator:
    """Real-time audio translation with streaming"""
    
    def __init__(self, approach="seamless", chunk_duration=3):
        self.approach = approach
        self.chunk_duration = chunk_duration
        self.sample_rate = 16000
        self.audio_queue = queue.Queue()
        self.is_running = False
        
        if approach == "seamless":
            self.processor, self.model, self.device = setup_seamless_m4t()
        else:
            self.models = setup_pipeline()
    
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio stream"""
        if status:
            print(status)
        self.audio_queue.put(indata.copy())
    
    def process_stream(self):
        """Process audio chunks from queue"""
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        buffer = []
        
        while self.is_running:
            try:
                data = self.audio_queue.get(timeout=0.1)
                buffer.extend(data.squeeze())
                
                if len(buffer) >= chunk_samples:
                    audio_chunk = np.array(buffer[:chunk_samples])
                    buffer = buffer[chunk_samples:]
                    
                    # Translate chunk
                    if self.approach == "seamless":
                        translated, sr = translate_audio_seamless(
                            audio_chunk, 
                            self.sample_rate,
                            self.processor,
                            self.model,
                            self.device
                        )
                    else:
                        translated, sr, eng_txt, deu_txt = translate_audio_pipeline(
                            audio_chunk,
                            self.sample_rate,
                            self.models
                        )
                        print(f"{eng_txt} → {deu_txt}")
                    
                    # Play translated audio
                    play_audio(translated, sr)
                    
            except queue.Empty:
                continue
    
    def start(self):
        """Start real-time translation"""
        self.is_running = True
        
        # Start processing thread
        process_thread = threading.Thread(target=self.process_stream)
        process_thread.start()
        
        # Start audio stream
        with sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            dtype='float32'
        ):
            print("Real-time translation started. Press Ctrl+C to stop...")
            try:
                while self.is_running:
                    sd.sleep(100)
            except KeyboardInterrupt:
                print("\nStopping...")
                self.is_running = False
        
        process_thread.join()


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=== Real-time Speech Translation: English → German ===\n")
    
    # Choose approach
    approach = "pipeline"
    # approach = "seamless"
    
    # OPTION 1: Simple one-shot translation
    # print("OPTION 1: Record and translate")
    # audio = record_audio(duration=5, sample_rate=16000)
    #
    # if approach == "seamless":
    #     processor, model, device = setup_seamless_m4t()
    #     translated_audio, sr = translate_audio_seamless(
    #         audio, 16000, processor, model, device
    #     )
    # else:
    #     models = setup_pipeline()
    #     translated_audio, sr, eng_text, deu_text = translate_audio_pipeline(
    #         audio, 16000, models
    #     )
    #
    # play_audio(translated_audio, sr)
    # save_audio(translated_audio, sr, "german_output.wav")
    
    # OPTION 2: Real-time streaming (uncomment to use)
    print("\nOPTION 2: Real-time streaming translation")
    translator = RealtimeTranslator(approach=approach, chunk_duration=3)
    translator.start()