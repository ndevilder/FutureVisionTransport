import streamlit as st
import requests
from PIL import Image
import os
import io

API_URL = os.getenv("API_URL", "API_URL = os.getenv("API_URL", "http://13.61.19.31/:8000/predict/")

st.title("Segmentation d'Images - Démo U-Net")

data_dir = "app/data/images"
masks_dir = "app/data/masks"

image_files = [f for f in os.listdir(data_dir) if f.endswith('_leftImg8bit.png')]

selected_image = st.selectbox("Choisissez une image dans le dossier 'data/images'", image_files)

if selected_image:
    # CHEMINS
    image_path = os.path.join(data_dir, selected_image)
    mask_filename = selected_image.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
    mask_path = os.path.join(masks_dir, mask_filename)

    # AFFICHER IMAGE ET MASQUE PRÉTRAITÉ
    image = Image.open(image_path)
    mask_image = Image.open(mask_path) if os.path.exists(mask_path) else None

    col1, col2 = st.columns(2)
    col1.image(image, caption=f"Image : {selected_image}", use_container_width=True)
    if mask_image:
        col2.image(mask_image, caption=f"Masque prétraité : {mask_filename}", use_container_width=True)
    else:
        col2.write("⚠️ Masque introuvable")

    # PRÉDICTION
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
                    mask_response = requests.get(f"http://localhost:8000/{mask_path}")
                    if mask_response.status_code == 200:
                        predicted_mask_image = Image.open(io.BytesIO(mask_response.content))

                        # AFFICHER LES 3 IMAGES CÔTE À CÔTE
                        st.subheader("Résultats de la segmentation")
                        col1, col2, col3 = st.columns(3)
                        col1.image(image, caption="Image Originale", use_container_width=True)
                        col2.image(mask_image, caption="Masque Prétraité", use_container_width=True)
                        col3.image(predicted_mask_image, caption="Masque Prédit", use_container_width=True)

                        st.success(f"Prédiction réussie en {processing_time}")
                    else:
                        st.error(f"Erreur lors du chargement du masque prédit : {mask_response.status_code}")
                else:
                    st.error("Aucun chemin de masque renvoyé par l'API.")
            else:
                st.error(f"Erreur de l'API : {response.status_code} - {response.text}")
