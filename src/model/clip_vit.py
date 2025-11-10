import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from loguru import logger


class VisionEncoder:
    def __init__(
            self,
            device: torch.device,
            model_name: str = "openai/clip-vit-large-patch14",
            unfreeze_layers: int = 0,
            only_use_processor: bool = False):
        if not only_use_processor:
            self.model = CLIPModel.from_pretrained(model_name).to(device)
            self._unfreeze_clip_layers(num_layers=unfreeze_layers)

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = device

    def path_to_tensor(self, image_paths: str | list[str]):
        """
        Convert image(s) to tensor(s) using CLIP processor.
        
        Returns:
            A BatchEncoding dict containing pixel_values.
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        images = [Image.open(path).convert("RGB") for path in image_paths]

        image_inputs = self.processor(images=images, return_tensors="pt")
        image_inputs = image_inputs.to(self.device)

        return image_inputs
    
    def encode(self, image_input) -> torch.Tensor:
        """
        Encode a batch of images using the CLIP vision model.
        """
        outputs = self.model.vision_model(**image_input)
        image_features = outputs.last_hidden_state
        image_features = image_features[:, 1:, :]  # Remove CLS token

        return image_features # (batch_size, num_patches - 1, hidden_dim)
    
    def _unfreeze_clip_layers(self, num_layers: int):
        """
        Unfreeze the last `num_layers` transformer layers of the CLIP vision model.
        """
        for param in self.model.parameters():
            param.requires_grad = False

        vision_layers = self.model.vision_model.encoder.layers

        for i, block in enumerate(reversed(vision_layers)):
            if i < num_layers:
                for param in block.parameters():
                    param.requires_grad = True

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Unfroze {num_layers} layers of CLIP vision model. Trainable parameters: {trainable_params}")