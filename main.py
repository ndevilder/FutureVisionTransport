from fastapi import FastAPI, APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from keras.saving import load_model
from keras.metrics import MeanIoU
from PIL import Image
import numpy as np
import io
import uvicorn
import tensorflow as tf
from keras.saving import register_keras_serializable
import os
import time
from fastapi.staticfiles import StaticFiles


@register_keras_serializable(package="MyMetrics")
class CustomMeanIoU(MeanIoU):
    def __init__(self, num_classes, name=None, dtype=None):
        super().__init__(num_classes=num_classes, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)


app = FastAPI()

router = APIRouter()

model_path = "app/model/mini_unet_hd_complete.keras"
model = load_model(model_path, custom_objects={'CustomMeanIoU': CustomMeanIoU})

def convert_mask_to_color(mask):
    # Exemple de palette de couleurs pour 8 classes
    # Tu peux ajuster selon tes besoins
    palette = {
        0: (128, 128, 128),  # 'void': gray
        1: (0, 0, 255),      # 'flat': blue
        2: (255, 165, 0),    # 'construction': orange
        3: (0, 128, 0),      # 'object': green
        4: (128, 0, 128),    # 'nature': purple
        5: (0, 255, 255),    # 'sky': cyan
        6: (255, 0, 255),    # 'human': magenta
        7: (255, 0, 0),      # 'vehicle': red
    }

    # Créer une image RGB vide
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)

    # Remplir l'image avec les couleurs de la palette
    for class_id, color in palette.items():
        color_mask[mask == class_id] = color

    return Image.fromarray(color_mask)


@router.post("/predict/")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()

    # Charger l'image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Redimensionner l'image à 256x256 pour le modèle
    image_resized = image.resize((256, 256))
    image_array = np.array(image_resized) / 255.0  # Normaliser
    image_array = np.expand_dims(image_array, axis=0)  # Ajouter la dimension batch

    # Prédire le masque
    prediction = model.predict(image_array)
    predicted_classes = np.argmax(prediction, axis=-1)[0]

    # Convertir en image couleur
    color_mask = convert_mask_to_color(predicted_classes)
    
    # Redimensionner le masque coloré à la taille de l'image d'origine
    color_mask_image = Image.fromarray(color_mask)
    color_mask_image = color_mask_image.resize(original_size, Image.NEAREST)

    # Sauvegarder le masque final
    mask_filename = f"mask_{int(time.time())}.png"
    mask_path = os.path.join("app", "data", "predictions", mask_filename)
    color_mask_image.save(mask_path

    processing_time = f"{(time.time() - start_time):.2f} s"

    # Retourner le chemin du masque et le temps de traitement
    return JSONResponse(content={
        "mask_path": f"app/data/predictions/{mask_filename}",
        "processing_time": processing_time
    })
    


app.mount("/app/data", StaticFiles(directory="app/data"), name="data")
app.include_router(router)