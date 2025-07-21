## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.

from dataclasses import dataclass

@dataclass
class Config:
    device: str
    dataset: str
    compress_model: str
    converter_model: str
    decoder_model: str
    embed_len: int
    write: bool
    segment_length: int
    stage: int = 1
    use_lora: bool = False
    lora_r: int = 64
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    adapter_model: str = None
    compressor_gradient_checkpoint: bool = False
    decoder_gradient_checkpoint: bool = False

    def __str__(self):
        return (
            f"--------------------------------------------------\n"
            f"Config:\n"
            f"Device: {self.device}\n"
            f"Dataset Path: {self.dataset}\n"
            f"Compress Model Path: {self.compress_model}\n"
            f"Converter Model Path: {self.converter_model}\n"
            f"Decoder Model Path: {self.decoder_model}\n"
            f"Compress Ratio: {256 // self.embed_len}\n"
            f"Write Enabled: {self.write}\n"
            f"Segment_length: {self.segment_length}\n"
            f"Use lora: {self.use_lora}\n"
            f"Lora R: {self.lora_r}\n" if self.use_lora else ""
            f"Lora Alpha: {self.lora_alpha}\n" if self.use_lora else ""
            f"Lora Dropout: {self.lora_dropout}\n" if self.use_lora else ""
            f"Adapter Model: {self.adapter_model}\n"
            f"Compressor Gradient Checkpoint: {self.compressor_gradient_checkpoint}\n"
            f"Decoder Gradient Checkpoint: {self.decoder_gradient_checkpoint}\n"
            f"--------------------------------------------------\n"
        )