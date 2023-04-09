# Pokémon Classifier

# Intro

A fine-tuned version of ViT-base on a collected set of Pokémon images. You can read more about the model [here](https://medium.com/@imjeffhi4/tutorial-using-vision-transformer-vit-to-create-a-pok%C3%A9mon-classifier-cb3f26ff2c20). 

# Using the model

```python
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import torch

# Loading in Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ViTForImageClassification.from_pretrained( "imjeffhi/pokemon_classifier").to(device)
feature_extractor = ViTFeatureExtractor.from_pretrained('imjeffhi/pokemon_classifier')

# Caling the model on a test image
img = Image.open('test.jpg')
extracted = feature_extractor(images=img, return_tensors='pt').to(device)
predicted_id = model(**extracted).logits.argmax(-1).item()
predicted_pokemon = model.config.id2label[predicted_id]
```