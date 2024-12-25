from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pydantic import BaseModel, Field, ConfigDict
from config import TranscriberConfig
from typing import Dict
from tqdm import tqdm
import numpy as np
import torch

class AudioData(BaseModel):
    array: np.ndarray = Field(..., arbitrary_types_allowed=True) 
    sample_rate: int
    
    # Allowing np.ndarray for the array field
    model_config = ConfigDict(arbitrary_types_allowed=True)

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

    def transcribe(self, data: AudioData) -> str:
        """
          Transcribe the given audio data to text.

          Parameters:
            audio_data (bytes): Raw audio data for transcription.

          Returns:
            str: Transcribed text from the audio data.
        """

        # Split the audio data into chunks
        chunk_duration = 30
        num_samples_per_chunk = chunk_duration * data.sample_rate
        chunks = [
            data.array[i:i + num_samples_per_chunk] 
            for i in range(0, len(data.array), num_samples_per_chunk)
        ]

        # Initialize the progress bar
        progress_bar = tqdm(total=len(chunks), desc="Transcribing", unit="chunk")

        # Process each chunk
        transcriptions = []
        for chunk in chunks:
            result = self.pipeline(
                chunk, 
                return_timestamps=True, 
                generate_kwargs={"language": self.config.language}
            )
            transcriptions.append(result["text"])
            progress_bar.update(1)

        # Concatenate the transcriptions
        text = " ".join(transcriptions)
        return text