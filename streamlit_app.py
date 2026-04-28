import streamlit as st 
import tensorflow as tf
import requests
from io import BytesIO
from PIL import Image

# =========================
# Load Model
# =========================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "mustafa_cnn_model.h5"
    )

model = load_model()

# =========================
# Class Names
# =========================

classnames = [
    'WITH MASK',
    'WITHOUT MASK'
]

# =========================
# Image Preprocessing
# =========================

def preprocess_image(img):

    img = img.convert('RGB')

    img = img.resize((224,224))

    img = tf.keras.preprocessing.image.img_to_array(img)

    img = img / 255.0

    img = tf.expand_dims(img, axis=0)

    return img

# =========================
# UI Title
# =========================

st.title("Face Mask Detection using CNN")

st.write(
"""
Upload an image or paste an image URL  
to detect whether a face mask is present.
"""
)

# =========================
# Input Method
# =========================

option = st.radio(
    "Choose input method:",
    ("Upload Image", "Image URL")
)

image_data = None

# Upload

if option == "Upload Image":

    file = st.file_uploader(
        "Upload an image...",
        type=["jpg","jpeg","png"]
    )

    if file:

        image_data = Image.open(file)

# URL

else:

    url = st.text_input("Paste image URL:")

    if url:

        try:

            response = requests.get(url)

            image_data = Image.open(
                BytesIO(response.content)
            )

        except:

            st.error("Invalid URL.")

# =========================
# Prediction
# =========================

if image_data:

    st.image(
        image_data,
        caption="Input Image",
        use_container_width=True
    )

    img = preprocess_image(image_data)

    with st.spinner("Analyzing image..."):

        prediction = model.predict(img)

    prob = prediction[0][0]

    threshold = 0.5

    if prob >= threshold:

        st.error(
            f"Result: WITHOUT MASK "
            f"(Confidence: {prob*100:.2f}%)"
        )

    else:

        st.success(
            f"Result: WITH MASK "
            f"(Confidence: {(1-prob)*100:.2f}%)"
        )

else:

    st.info(
        "Upload or provide an image to begin."
    )
