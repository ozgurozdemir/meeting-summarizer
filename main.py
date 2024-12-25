from transcriber import Transcriber, AudioData
from config import TranscriberConfig
import librosa
import json

with open("config.json", "r") as f:
    config = json.load(f)

try:
    transcriber_config = TranscriberConfig(**config)
    transcriber = Transcriber(transcriber_config)
except Exception as e:
    print(f"Error loading config: {e}")
    exit(1)

if __name__ == "__main__":
    waveform, sampling_rate = librosa.load("test.m4a")

    data = AudioData = {
        "array": waveform,
        "sample_rate": sampling_rate
    }
    result = transcriber.transcribe(data)
    
    with open("test.txt", "w") as f:
        f.write(result) # write the result to a file

