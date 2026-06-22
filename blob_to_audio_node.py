
import wave
from io import BytesIO
import torch
import numpy as np


class BlobToAudioNode:
    """
    Decode binary audio data (WAV bytes) to ComfyUI AUDIO type.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bytes": ("BYTES",),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "decode"
    OUTPUT_NODE = False
    CATEGORY = "RequestNode/Utils"

    def decode(self, bytes: bytes):
        w = wave.open(BytesIO(bytes), "rb")
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
        w.close()

        if sampwidth == 2:
            audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
        elif sampwidth == 4:
            audio_np = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483647.0
        elif sampwidth == 1:
            audio_np = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")

        if n_channels > 1:
            audio_np = audio_np.reshape(-1, n_channels)
        else:
            audio_np = audio_np.reshape(-1, 1)

        # ComfyUI AUDIO format: [batch, channels, samples]
        waveform = torch.from_numpy(audio_np)  # [samples, channels]
        waveform = waveform.permute(1, 0).unsqueeze(0)  # [1, channels, samples]
        audio = {"waveform": waveform, "sample_rate": framerate}
        return (audio,)
