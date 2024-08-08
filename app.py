import streamlit as st
from PIL import Image
import pickle
import numpy as np
import base64

# Function to add background image from a file
def add_background_image(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url(data:image/png;base64,{encoded});
        background-size: cover;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Add your background image from the local file in the repo
add_background_image('rice_farmer.jpeg')

# Load the pickled model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Add custom CSS to change the text color to light blue
st.markdown(
    """
    <style>
    .light-blue-text {
        color: #ADD8E6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Rice Image Classification')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying the image")

    # Preprocess the image to match the model's expected input
    image = image.resize((128, 128))  # Resize to the size your model expects
    image = np.array(image) / 255.0  # Normalize the image
    image = image.reshape((1, 128, 128, 3))  # Reshape to add batch dimension

    # Make prediction
    prediction = model.predict(image)
    class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']  # Replace with your actual class names
    predicted_class = class_names[np.argmax(prediction)]

    st.markdown(f'<p class="light-blue-text">Predicted class: {predicted_class}</p>', unsafe_allow_html=True)
    
    # Conditional messages for each rice type
    if predicted_class == 'Ipsala':
        st.markdown('<p class="light-blue-text">Ipsala Rice is grown in the Ipsala Plain, one of the most fertile rice fields in Turkey. This type of rice is known for its large grains and flavor. It is especially preferred for making pilaf and stands out with its ability to remain grainy during cooking.</p>', unsafe_allow_html=True)
    elif predicted_class == 'Jasmine':
        st.markdown('<p class="light-blue-text">Jasmine Rice is an aromatic type of rice widely used especially in Thai cuisine. It attracts attention with its long grains and distinctive fragrance. It is known for its soft and light texture when cooked.</p>', unsafe_allow_html=True)
    elif predicted_class == 'Basmati':
        st.markdown('<p class="light-blue-text">Basmati Rice is known for its long and fine grains. This rice, which is widely used especially in Indian and Middle Eastern cuisine, attracts attention with its pleasant smell and light structure. It is frequently preferred in pilafs with its grains that elongate when cooked.</p>', unsafe_allow_html=True)
    elif predicted_class == 'Arborio':
        st.markdown('<p class="light-blue-text">Arborio Rice is a short-grain rice named after Arborio in Italy\'s Po Valley. Known for its high amylopectin content, it has a creamy texture ideal for risotto. The grains are short, fat, and roundish. When cooked, Arborio rice absorbs liquids well, making it perfect for creamy dishes like risotto and rice pudding. It’s sautéed in butter or oil before adding broth to achieve its signature texture. Nutritionally, it’s a good source of carbohydrates and some protein.</p>', unsafe_allow_html=True)
    elif predicted_class == 'Karacadag':
        st.markdown('<p class="light-blue-text">Karacadag Rice is a local variety of rice grown in the Karacadag region of Turkey. It is known for its high quality and is often preferred for its unique taste and aroma.</p>', unsafe_allow_html=True)
