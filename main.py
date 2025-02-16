from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from keras.saving import load_model
from keras.metrics import MeanIoU
from PIL import Image
import numpy as np
import io
import uvicorn
import tensorflow as tf
import time
from keras.saving import register_keras_serializable

@register_keras_serializable(package="MyMetrics")
class CustomMeanIoU(MeanIoU):
    def __init__(self, num_classes, name='mean_iou', dtype=None):
        super().__init__(num_classes=num_classes, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

# Initialiser l'application FastAPI
app = FastAPI()
    
# Charger le modèle
model_path = "app/model/mini_unet_hd_complete.keras"
model = load_model(model_path, custom_objects={'MyMetrics>CustomMeanIoU': CustomMeanIoU})

# Définir la palette de couleurs
PALETTE = np.array([
    [0, 0, 0],        # void
    [128, 64, 128],   # flat
    [70, 70, 70],     # construction
    [190, 153, 153],  # object
    [107, 142, 35],   # nature
    [70, 130, 180],   # sky
    [220, 20, 60],    # human
    [0, 0, 142]       # vehicle
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        start_time = time.time()  # DÉBUT TEMPS DE TRAITEMENT

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image = image.resize((256, 256))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array)
        predicted_classes = np.argmax(prediction, axis=-1)[0]

        mask_colored = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(8):
            mask_colored[predicted_classes == i] = PALETTE[i]

        mask_path = 'static/predicted_mask.png'
        Image.fromarray(mask_colored).save(mask_path)

        end_time = time.time()  # FIN TEMPS DE TRAITEMENT
        processing_time = end_time - start_time

        return JSONResponse(content={
            "message": "Prediction réussie",
            "mask_path": "static/predicted_mask.png",
            "processing_time": f"{processing_time:.2f} secondes"
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)