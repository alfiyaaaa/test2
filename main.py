import io
from PIL import Image
import torch
import base64
from io import BytesIO

from fastapi import FastAPI, HTTPException

from transformers import ViTForImageClassification, ViTImageProcessor

app = FastAPI()

device_idx = 0
device = torch.device("cuda:{}".format(device_idx) if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
model_path = "./model/"
model = ViTForImageClassification.from_pretrained(model_path).to(device)
tokenizer = ViTImageProcessor.from_pretrained(model_path)

# Define the prediction function
def predict(image_file):
    image = Image.open(io.BytesIO(image_file))
    inputs = tokenizer(images=image, return_tensors="pt", padding=True)
    inputs = inputs.to(device)
    outputs = model(**inputs)
    predicted_class_idx = torch.argmax(outputs.logits).item()
    predicted_class = model.config.id2label[predicted_class_idx]
    print(f"Max GPU memory allocated: {torch.cuda.max_memory_allocated(device) / 1024 / 1024:.2f} MB")
    return predicted_class

# Define the FastAPI app route
@app.post("/predict")
async def make_prediction(image_data: str):
    try:
        decoded_image = base64.b64decode(image_data)
        predicted_class = predict(decoded_image)
        return predicted_class
    except:
        raise HTTPException(status_code=400, detail="Invalid image data")
