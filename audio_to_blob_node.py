import io
import torch
import numpy as np


class AudioToBlobNode:
    """Convert ComfyUI AUDIO type to WAV binary (BYTES).

    ComfyUI AUDIO = {"waveform": Tensor [batch, samples, channels], "sample_rate": int}
    Outputs raw WAV bytes suitable for HTTP POST as application/octet-stream.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("BYTES",)
    RETURN_NAMES = ("wav_bytes",)
    FUNCTION = "convert"
    CATEGORY = "RequestNode/Converters"

    def convert(self, audio):
        waveform = audio["waveform"]  # [batch, channels, samples]
        sample_rate = audio["sample_rate"]

        # ComfyUI AUDIO format: [batch, channels, samples]
        w = waveform
        if w.dim() == 3:
            w = w.squeeze(0)  # [channels, samples]
        # soundfile.write expects [samples, channels]
        w = w.T  # [samples, channels]

        # Convert to int16 numpy
        w_np = w.cpu().float().numpy()
        w_np = np.clip(w_np, -1.0, 1.0)
        w_int16 = (w_np * 32767).astype(np.int16)

        buffer = io.BytesIO()
        import soundfile as sf
        sf.write(buffer, w_int16, sample_rate, format="WAV")
        buffer.seek(0)
        return (buffer.getvalue(),)


NODE_CLASS_MAPPINGS = {
    "AudioToBlobNode": AudioToBlobNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioToBlobNode": "Audio to Blob (WAV)",
}
