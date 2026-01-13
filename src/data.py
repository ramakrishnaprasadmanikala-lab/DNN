import os
import json
import requests
from PIL import Image
from io import BytesIO

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ========== Image transform ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ============================================
# Helper: Load image from URL as a tensor
# ============================================

def load_image_from_url(url):
    """
    Downloads image -> converts to tensor.
    If fails, return zero tensor.
    """
    try:
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return transform(img)
    except Exception:
        return torch.zeros(3, 224, 224)


# ============================================
# Dataset class — loads images + captions
# ============================================

class URLImageDataset(Dataset):

    def __init__(self, json_file, stoi_dict, max_len=20):
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Dataset not found: {json_file}")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Some JSONs nest inside "images"
        self.items = data.get("images", data)
        self.stoi = stoi_dict
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def encode_caption(self, text):
        """
        Encodes caption text to padded tensor.
        """
        words = text.lower().split()
        token_ids = [self.stoi["<bos>"]]

        for w in words:
            token_ids.append(self.stoi.get(w, self.stoi["<unk>"]))

        token_ids.append(self.stoi["<eos>"])
        token_ids = token_ids[:self.max_len]
        token_ids += [self.stoi["<pad>"]] * (self.max_len - len(token_ids))

        return torch.tensor(token_ids)

    def __getitem__(self, idx):
        item = self.items[idx]
        image_url = item.get("url_o", None)
        caption = item.get("text", "") or item.get("title", "")

        img_tensor = load_image_from_url(image_url)
        caption_tensor = self.encode_caption(caption)

        return {
            "images": img_tensor,
            "tokens": caption_tensor
        }
