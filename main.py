import json
from config import TranscriberConfig
from transcriber import Transcriber
import soundfile as sf 

with open("config.json", "r") as f:
    config = json.load(f)

try:
    transcriber_config = TranscriberConfig(**config)
    transcriber = Transcriber(transcriber_config)
except Exception as e:
    print(f"Error loading config: {e}")
    exit(1)

if __name__ == "__main__":
    data, sampling_rate = librosa.load("test.m4a")
    result = transcriber.transcribe(audio_path)
    
    with open("test.txt", "w") as f:
        f.write(result) # write the result to a file

