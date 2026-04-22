# NUSKHA

**Neural Understanding and Structured Knowledge Harvesting from Handwritten Annotations in Medical Prescriptions**

NUSKHA is a multimodal medical OCR project for recognizing handwritten and scanned Indian medical prescriptions. It combines image preprocessing, prescription-specific dataset loading, and vision-language OCR modeling to convert unstructured prescription images into structured clinical data.

The project focuses on extracting useful healthcare information such as medicines, dosage instructions, diagnosis, treatment plan, and SOAP-style clinical notes from prescription images.

## Project Overview

Indian medical prescriptions often contain mixed handwriting, abbreviations, printed headers, handwritten annotations, regional medicine naming styles, and inconsistent formatting. Generic OCR systems usually struggle with this type of document because they only extract raw text and do not understand the medical structure behind it.

NUSKHA addresses this by using a multimodal OCR pipeline. The model receives both the prescription image and a text prompt, then learns to produce structured medical information in JSON format.

## Key Features

- Recognizes handwritten and scanned medical prescriptions.
- Uses image enhancement to improve prescription readability.
- Supports multimodal OCR through a vision-language model.
- Extracts prescription content into structured JSON.
- Handles SOAP-style clinical annotation fields.
- Designed with Indian prescription formats and handwriting variation in mind.
- Provides a training pipeline for prescription image and annotation pairs.

## Repository Structure

```text
ocr/
+-- data/
|   +-- annotations.json
|   +-- images/
|       +-- 1.jpg
|       +-- 2.jpg
|       +-- ...
+-- dataset/
|   +-- prescription_dataset.py
+-- models/
|   +-- donut_model.py
|   +-- enhancement.py
+-- inference.py
+-- requirements.txt
+-- train.py
+-- README.md
```

## OCR Pipeline

The NUSKHA pipeline follows these main stages:

1. **Prescription Image Collection**

   Prescription images are stored inside `data/images/`. Each image corresponds to an entry in `data/annotations.json`.

2. **Annotation Loading**

   The dataset loader in `dataset/prescription_dataset.py` reads the JSON annotation file and maps every prescription image to its target structured output.

3. **Image Preprocessing**

   `models/enhancement.py` contains an image enhancement module. It applies preprocessing operations such as grayscale conversion, noise reduction, adaptive thresholding, normalization, and tensor/array preparation.

4. **Target Formatting**

   The dataset converts medical fields into a JSON-style target format. Current supported fields include:

   ```json
   {
     "subjective": "",
     "objective": "",
     "assessment": "",
     "plan": ""
   }
   ```

5. **Multimodal Input Preparation**

   The training collate function uses the model processor to prepare image tensors, text tokens, labels, and multimodal image-grid metadata. For GLM-OCR, the prompt must include the `<|image|>` token so image features align correctly with image tokens.

6. **Model Training**

   `train.py` loads the medical OCR model, prepares the dataset, creates batches, and fine-tunes the model to map prescription images to structured medical text.

7. **Structured Output Generation**

   During inference, the trained model can generate structured text or JSON-like output from a new prescription image.

## Model

The current implementation uses a GLM-OCR based multimodal model through Hugging Face Transformers:

```python
AutoProcessor.from_pretrained("zai-org/GLM-OCR", trust_remote_code=True)
AutoModelForImageTextToText.from_pretrained("zai-org/GLM-OCR", trust_remote_code=True)
```

The model processes prescription images together with text prompts and generates medical information as structured output.

## Dataset Format

The annotation file is expected at:

```text
data/annotations.json
```

Each annotation entry should contain an image reference and medical fields. The dataset currently supports image keys such as:

```json
{
  "fileName": "1.jpg"
}
```

or:

```json
{
  "image_id": "1.jpg"
}
```

For SOAP-based entries, the loader supports:

```json
{
  "SOAP": {
    "Subjective": "",
    "Objective": "",
    "Assessment": "",
    "Plan": ""
  }
}
```

For direct fields, the loader supports:

```json
{
  "subjective": "",
  "objective": "",
  "assessment": "",
  "plan": ""
}
```

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, install the PyTorch version that matches your CUDA setup from the official PyTorch installation instructions.

## Training

Run training from inside the `ocr/` directory:

```bash
python train.py
```

Training performs the following steps:

- Loads the GLM-OCR processor and model.
- Reads prescription images from `data/images/`.
- Reads annotations from `data/annotations.json`.
- Converts target medical fields into JSON text.
- Prepares multimodal inputs with image tokens and image features.
- Fine-tunes the model using AdamW optimization.
- Saves the trained model and processor into:

```text
checkpoints/glm_medical_ocr/
```

## Inference

The project includes `inference.py` as an inference entry point. The intended inference flow is:

1. Load the trained model checkpoint.
2. Load a prescription image.
3. Apply optional image enhancement.
4. Send the image and prompt to the OCR model.
5. Decode the generated output.
6. Convert the result into structured JSON if required.

Example expected output:

```json
{
  "medicine": "Paracetamol",
  "dosage": "500 mg",
  "frequency": "Twice daily",
  "diagnosis": "Fever",
  "plan": "Follow up if symptoms persist"
}
```

## Current Status

The current project includes:

- Prescription image dataset structure.
- JSON annotation loading.
- Image enhancement module.
- GLM-OCR model wrapper.
- Training script.
- Initial inference script.
- Handling for multimodal image-token alignment issues.

## Future Improvements

- Improve inference script into a complete CLI.
- Add validation metrics such as CER, WER, and JSON field accuracy.
- Add medicine-name normalization.
- Add confidence scoring for extracted fields.
- Add frontend integration for uploading and viewing prescription results.
- Add support for multilingual Indian prescriptions.
- Export predictions directly into EHR-friendly JSON format.

## Project Description

NUSKHA is a multimodal medical OCR system that recognizes handwritten annotations in Indian prescriptions. It uses image enhancement and vision-language modeling to extract medicines, dosage, diagnosis, and SOAP notes into structured JSON for faster prescription digitization and healthcare record management.
