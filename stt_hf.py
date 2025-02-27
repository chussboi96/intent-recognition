import sounddevice as sd
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import queue
import threading
import sys
from typing import Optional

class VoiceProcessor:
    def __init__(
        self,
        model_id="openai/whisper-large-v3",
        device="cpu",
        silence_seconds=2,
        not_interrupt_words=None,
        logger=None,
    ):
        self.logger = logger
        self.device = device
        self.not_interrupt_words = not_interrupt_words or ["um", "uh", "hmm"]
        
        # Initialize recording parameters
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = np.float32
        self.recording_duration = 10  # Fixed 10 seconds recording
        
        # Initialize Whisper model
        device_str = "cuda:0" if device == "cuda" else "cpu"
        try:
            print("Loading Whisper model...")
            self.processor = WhisperProcessor.from_pretrained(model_id)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
            self.model = self.model.to(device_str)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error initializing voice processor: {e}")
            raise

    def _log_event(self, event: str, details: str, further: str = ""):
        if self.logger:
            self.logger.info(event, extra={"details": details, "further": further})

    def transcribe(self, audio: np.ndarray) -> str:
        try:
            # Ensure audio is correct shape and type
            audio = audio.flatten().astype(np.float32)
            
            # Process through Whisper with proper attention mask
            inputs = self.processor(
                audio,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                return_attention_mask=True  # Explicitly request attention mask
            )
            
            # Move inputs to the correct device
            input_features = inputs.input_features.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device) if hasattr(inputs, 'attention_mask') else None
            
            # Generate with attention mask
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    attention_mask=attention_mask
                )
                transcription = self.processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True
                )[0]
            
            return transcription.strip()
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

    def listen(self) -> str:
        """Record audio for 10 seconds and transcribe"""
        try:
            # Calculate total samples needed
            total_samples = int(self.sample_rate * self.recording_duration)
            
            # Initialize array to store the recording
            recording = np.zeros((total_samples, self.channels), dtype=self.dtype)
            
            # Create a variable to track the current position in the recording
            current_sample = 0
            
            def callback(indata, frame_count, time_info, status):
                nonlocal current_sample
                if status:
                    print(f"Status: {status}")
                remaining = total_samples - current_sample
                if remaining > 0:
                    samples_to_write = min(remaining, len(indata))
                    recording[current_sample:current_sample + samples_to_write] = indata[:samples_to_write]
                    current_sample += samples_to_write

            # Start recording
            with sd.InputStream(
                channels=self.channels,
                samplerate=self.sample_rate,
                dtype=self.dtype,
                callback=callback,
                blocksize=int(self.sample_rate * 0.5)  # 0.5 second blocks
            ):
                print("\nRecording...")
                sd.sleep(int(self.recording_duration * 1000))  # Convert to milliseconds
            
            print("Recording complete.")
            
            # Check if we got any audio
            if np.all(recording == 0):
                print("No audio detected")
                return ""
                
            # Check if we got some actual sound (not just silence)
            if np.max(np.abs(recording)) < 0.01:
                print("Only silence detected")
                return ""
            
            # Transcribe
            self._log_event("transcribing", "STT")
            text = self.transcribe(recording)
            self._log_event("transcribed", "STT", text)
            
            return text.strip()
            
        except Exception as e:
            print(f"Error in listen: {e}")
            return ""