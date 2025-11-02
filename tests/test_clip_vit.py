from src.model.clip_vit import VisionEncoder
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

def main():
    model_name = "openai/clip-vit-large-patch14"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_embedding = model.get_image_features(**inputs)

    with torch.no_grad():
        vision_outputs = model.vision_model(**inputs)
        path_embeddings = vision_outputs.last_hidden_state

    print("Image embedding shape:", image_embedding.shape)
    print("Path embeddings shape:", path_embeddings.shape)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")