from pydantic import BaseModel

class TranscriberConfig(BaseModel):
    model: str
    device: str
    low_cpu_mem_usage: bool = True
    use_safetensors: bool = True
    language: str = "english"