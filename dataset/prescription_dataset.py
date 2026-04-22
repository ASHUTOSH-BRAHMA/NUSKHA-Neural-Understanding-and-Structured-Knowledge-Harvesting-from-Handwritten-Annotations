import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class PrescriptionDataset(Dataset):
    def __init__(
        self,
        image_dir,
        annotation_file,
        processor,
        enhancer=None,
        max_length=512
    ):
        self.image_dir = image_dir
        self.processor = processor
        self.enhancer = enhancer
        self.max_length = max_length

        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

        with open(annotation_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # --------------------------------------------------
        # 1. LOAD IMAGE
        # --------------------------------------------------
        image_name = sample.get("fileName") or sample.get("image_id")
        if image_name is None:
            raise KeyError(f"Index {idx} missing image reference")

        image_path = os.path.join(self.image_dir, image_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        # Apply enhancer ONCE
        if self.enhancer:
            image = self.enhancer(image)

        # HARD SAFETY CHECK
        if not isinstance(image, Image.Image):
            raise ValueError(
                f"Enhancer returned invalid type at idx={idx}: {type(image)}"
            )

        image = image.convert("RGB")

        # --------------------------------------------------
        # 2. SOAP EXTRACTION
        # --------------------------------------------------
        if "SOAP" in sample:
            soap = sample["SOAP"]
            target_dict = {
                "subjective": soap.get("Subjective", ""),
                "objective": soap.get("Objective", ""),
                "assessment": soap.get("Assessment", ""),
                "plan": soap.get("Plan", "")
            }
        else:
            target_dict = {
                "subjective": sample.get("subjective", ""),
                "objective": sample.get("objective", ""),
                "assessment": sample.get("assessment", ""),
                "plan": sample.get("plan", "")
            }

        target_text = json.dumps(target_dict, ensure_ascii=False)
        # GLM-OCR requires an explicit image token in the prompt so image tokens align with features.
        if "<|image|>" not in target_text:
            target_text = "<|image|>\n" + target_text

        return {
            "image": image,
            "text": target_text
        }
