from __future__ import annotations
import numpy as np
from PIL import Image
from src import config
import torch
from transformers import CLIPModel,CLIPProcessor



class CLIPEncoder:
    def __init__(self, model_name:str = config.MODEL_NAME, device:str|None = None):
        self._torch = torch
        torch.set_num_threads(1)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        self.dim = self.model.config.projection_dim


    def encode_image(self, image):

        image = image.convert("RGB")
        inputs = self.processor(images = image, return_tensors = "pt")
        inputs = {k:v.to(self.device) for k,v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model.get_image_features(**inputs)
            vector = outputs.image_embeds if hasattr(outputs, "image_embeds") else outputs
            if hasattr(vector, "pooler_output"):
                vector = vector.pooler_output

        vector = vector/vector.norm(p = 2, dim = -1, keepdim = True).clamp_min(1e-12)
        return vector.cpu().numpy().astype(np.float32)
    

    def encode_text(self, text):
        if not text or not text.strip():
                raise ValueError("Enter a non-empty search description.")

        inputs = self.processor(text = text, padding = True, truncation = True, return_tensors = "pt")
        inputs = {k:v.to(self.device) for k,v in inputs.items()}

        with torch.inference_mode():
                outputs = self.model.get_text_features(**inputs)
                vector = outputs.text_embeds if hasattr(outputs, "text_embeds") else outputs
                if hasattr(vector, "pooler_output"):
                    vector = vector.pooler_output

        vector = vector/vector.norm(p = 2, dim = -1, keepdim = True).clamp_min(1e-12)
        return vector.cpu().numpy().astype(np.float32)

    def encode_joint(self, image=None, text=None, alpha:float = config.DEFAULT_ALPHA,):
        parts = []
        if image is not None:
            parts.append(alpha * self.encode_image(image))
        if text and text.strip():
            parts.append((1 - alpha) * self.encode_text(config.TEXT_PROMPT.format(text.strip())))
        if not parts:
            raise ValueError("Give an image, a text, or both.")
        v = np.sum(parts, axis=0)
        return (v / np.linalg.norm(v)).astype(np.float32)

