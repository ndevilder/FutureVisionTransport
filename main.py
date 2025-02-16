from fastapi import FastAPI, UploadFile, File
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

model_path = "app/model/mini_unet_hd_complete.keras"
model = load_model(model_path, custom_objects={'CustomMeanIoU': CustomMeanIoU})


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        start_time = time.time()
        contents = await file.read()
        image_filename = file.filename

        mask_filename = image_filename.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
        mask_path = f"app/data/masks/{mask_filename}"

        if not os.path.exists(mask_path):
            return JSONResponse(status_code=404, content={"error": f"Masque non trouvé : {mask_path}"})

        mask = Image.open(mask_path)
        mask = np.array(mask)

        prediction = model.predict(np.expand_dims(mask, axis=(0, -1)))

        predicted_mask_image = Image.fromarray(np.argmax(prediction[0], axis=-1).astype(np.uint8))
        output_path = "app/data/predicted_mask.png"
        predicted_mask_image.save(output_path)

        return {"mask_path": output_path, "processing_time": f"{time.time() - start_time:.2f}s"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


app.mount("/app/data", StaticFiles(directory="app/data"), name="data")
