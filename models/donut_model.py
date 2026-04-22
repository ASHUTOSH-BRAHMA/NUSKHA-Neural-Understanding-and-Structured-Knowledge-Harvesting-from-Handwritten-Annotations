import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

class MedicalOCRSystem(nn.Module):
    def __init__(self, model_id="zai-org/GLM-OCR"):
        super().__init__()
        print(f"Loading {model_id}... (Ensure Developer Mode is ON for Windows)")
        
        # 1. Use AutoProcessor (Bundles image + text logic)
        # trust_remote_code=True is required for GLM-OCR's custom architecture
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        
        # 2. Use AutoModelForImageTextToText (The 2026 standard for VLMs)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16, # Uses half-memory, high accuracy
            device_map="auto"           # Automatically uses GPU if available
        )

    def forward(self, images, text_prompts, labels=None):
        """
        Standard forward pass for training.
        """
        inputs = self.processor(
            text=text_prompts, 
            images=images, 
            return_tensors="pt", 
            padding=True
        ).to(self.model.device)
        
        if labels is not None:
            inputs["labels"] = labels

        return self.model(**inputs)

    def extract_as_json(self, image_path):
        """
        Inference method to get structured JSON from a prescription.
        """
        image = Image.open(image_path).convert("RGB")
        
        # GLM-OCR uses specific instruction triggers
        prompt = "Information Extraction: Extract medicine, dosage, and frequency as JSON."
        
        # Format the multimodal input
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0 # Keep it precise for medical data
            )
        
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

# --- Training Setup Example ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ocr_model = MedicalOCRSystem()
    
    # Example Inference for one of your uploaded files
    try:
        print("\n--- Testing Extraction on 51.jpg ---")
        result = ocr_model.extract_as_json("51.jpg")
        print(result)
    except FileNotFoundError:
        print("Image 51.jpg not found in the local directory.")