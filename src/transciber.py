import torch
import numpy as np
from typing import Dict
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class Transcriber:
    """ Transcriber class for speech recognition """
    def __init__(self, config: TranscriberConfig):
        self.config = config
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Model object
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.config.model,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=self.config.low_cpu_mem_usage,
            use_safetensors=self.config.use_safetensors
        ).to(self.config.device)

        # Spectrogram extractor object
        self.processor = AutoProcessor.from_pretrained(self.config.model)

        # Pipeline object
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.config.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.config.device,
        )

    def transcribe(self, audio_data: Dict[np.ndarray, int]) -> str:
        """
          Transcribe the given audio data to text.

          Parameters:
            audio_data (bytes): Raw audio data for transcription.

          Returns:
            str: Transcribed text from the audio data.
        """
        result = self.pipeline(audio_data, 
                               return_timestamps=True, 
                               generate_kwargs={"language": self.config.language})
        return result["text"]