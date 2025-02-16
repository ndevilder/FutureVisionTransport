import streamlit as st
import requests
from PIL import Image
import os
import io



API_URL = os.getenv("API_URL", "http://13.61.19.31:8000/predict/")

st.title("Segmentation d'Images - Démo U-Net")

# Lister les images du dossier 'data'
data_dir = "app/data"
image_files = [f for f in os.listdir(data_dir) if f.endswith(('png', 'jpg', 'jpeg'))]

selected_image = st.selectbox("Choisissez une image dans le dossier 'data'", image_files)

if selected_image:
    image_path = os.path.join(data_dir, selected_image)
    image = Image.open(image_path)
    st.image(image, caption=f"Image choisie : {selected_image}", use_column_width=True)

    if st.button("Prédire la segmentation"):
        with st.spinner("Prédiction en cours..."):
            with open(image_path, 'rb') as img_file:
                files = {"file": img_file}
                response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                result = response.json()
                mask_path = result.get("mask_path")
                processing_time = result.get("processing_time")

                if mask_path:
                    mask_response = requests.get(f"http://13.61.19.31:8000/{mask_path}")
                    if mask_response.status_code == 200:
                        mask_image = Image.open(io.BytesIO(mask_response.content))

                        col1, col2 = st.columns(2)
                        col1.image(image, caption="Image Originale", use_column_width=True)
                        col2.image(mask_image, caption="Masque Prédit", use_column_width=True)

                        st.success(f"Prédiction réussie en {processing_time}")
                    else:
                        st.error(f"Erreur lors du chargement du masque : {mask_response.status_code}")
                else:
                    st.error("Aucun chemin de masque renvoyé par l'API.")
            else:
                st.error(f"Erreur de l'API : {response.status_code} - {response.text}")
