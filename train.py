import torch
import os
import time
from torch.utils.data import DataLoader
from functools import partial
from torch import optim
from transformers import get_scheduler
from models.donut_model import MedicalOCRSystem 
from dataset.prescription_dataset import PrescriptionDataset

def collate_fn(batch, processor, max_len):
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]

    inputs = processor(
        images=images,
        text=texts,
        return_tensors="pt",
        padding="longest",
        truncation=False,
    )

    input_ids = inputs["input_ids"]
    # If any sample exceeds max_len, drop it to avoid image-token mismatch from truncation.
    # Re-run processor on the filtered batch to keep image tokens aligned.
    keep_indices = [i for i in range(input_ids.size(0)) if input_ids.size(1) <= max_len]
    if len(keep_indices) != input_ids.size(0):
        if len(keep_indices) == 0:
            return None
        images = [images[i] for i in keep_indices]
        texts = [texts[i] for i in keep_indices]
        inputs = processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding="longest",
            truncation=False,
        )
        input_ids = inputs["input_ids"]
    labels = input_ids.clone()
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100

    batch_out = {
        "pixel_values": inputs["pixel_values"],
        "labels": labels,
    }
    if "image_grid_thw" in inputs:
        batch_out["image_grid_thw"] = inputs["image_grid_thw"]

    return batch_out

# --- GPU / dtype ---
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    device = "cuda"
    dtype = torch.bfloat16
else:
    print("WARNING: Training on CPU. This will be extremely slow.")
    device = "cpu"
    dtype = torch.float32

# --- MODEL & PROCESSOR ---
model_wrapper = MedicalOCRSystem()
model = model_wrapper.model.to(device, dtype=dtype)
processor = model_wrapper.processor
# Cap max length to reduce extremely slow steps; override via env if needed.
# Example: set TOKENIZER_MAX_LEN=4096 to restore the previous behavior.
model_max = getattr(model.config, "max_position_embeddings", None)
env_max = int(os.getenv("TOKENIZER_MAX_LEN", "1024"))
max_len = min(env_max, model_max) if model_max is not None else env_max
processor.tokenizer.model_max_length = max_len
print(f"Tokenizer max length: {max_len}")
collate = partial(collate_fn, processor=processor, max_len=max_len)

# Vision config for grid_thw (fallback values match GLM-OCR defaults)
vision_config = model.config.vision_config
patch_size = getattr(vision_config, "patch_size", 14)
spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2)
print(f"Vision config → patch_size={patch_size}, spatial_merge_size={spatial_merge_size}")

if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    print(f"Set pad_token_id to eos_token_id: {processor.tokenizer.pad_token_id}")

# --- DATASET (now returns image_grid_thw) ---
dataset = PrescriptionDataset(
    image_dir="data/images",
    annotation_file="data/annotations.json",
    processor=processor
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    collate_fn=collate,
)

optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
max_steps_per_epoch = int(os.getenv("MAX_STEPS_PER_EPOCH", "0"))

# --- TRAINING LOOP ---
model.train()
for epoch in range(15):
    epoch_loss = 0.0
    for step, batch in enumerate(loader):
        if batch is None:
            print("Skipped batch: sequence length exceeds max_len")
            continue
        t0 = time.perf_counter()
        pixel_values = batch["pixel_values"].to(device, dtype=dtype)
        labels = batch["labels"].to(device)
        image_grid_thw = batch.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device)

        # Decoder input_ids (your original logic)
        input_ids = labels.clone()
        input_ids[input_ids == -100] = processor.tokenizer.pad_token_id

        # Forward pass – now includes image_grid_thw
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            image_grid_thw=image_grid_thw,
        )
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()

        if step % 2 == 0:
            dt = time.perf_counter() - t0
            print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f} | {dt:.2f}s")

        if max_steps_per_epoch and (step + 1) >= max_steps_per_epoch:
            print(f"Max steps per epoch reached: {max_steps_per_epoch}")
            break

    print(f"--- Epoch {epoch+1} Complete | Avg Loss: {epoch_loss/len(loader):.4f} ---")

# --- SAVE ---
os.makedirs("checkpoints/glm_medical_ocr", exist_ok=True)
model.save_pretrained("checkpoints/glm_medical_ocr")
processor.save_pretrained("checkpoints/glm_medical_ocr")
