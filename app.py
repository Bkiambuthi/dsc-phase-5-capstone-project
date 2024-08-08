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
        color: black; /* Set all text color to black */
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Add your background image from the local file in the repo
add_background_image('rice_farmer.jpeg')

# Load the pickled model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Apply global CSS to set all text color to black
st.markdown(
    """
    <style>
    .stApp {{
        color: black;
    }}
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

    st.write(f'Predicted class: {predicted_class}')
    
    # Conditional messages for each rice type
    if predicted_class == 'Ipsala':
        st.write("Ipsala Rice is grown in the Ipsala Plain, one of the most fertile rice fields in Turkey. This type of rice is known for its large grains and flavor. It is especially preferred for making pilaf and stands out with its ability to remain grainy during cooking.")
    elif predicted_class == 'Jasmine':
        st.write("Jasmine Rice is an aromatic type of rice widely used especially in Thai cuisine. It attracts attention with its long grains and distinctive fragrance. It is known for its soft and light texture when cooked.")
    elif predicted_class == 'Basmati':
        st.write("Basmati Rice is known for its long and fine grains. This rice, which is widely used especially in Indian and Middle Eastern cuisine, attracts attention with its pleasant smell and light structure. It is frequently preferred in pilafs with its grains that elongate when cooked.")
    elif predicted_class == 'Arborio':
        st.write("Arborio Rice is a short-grain rice named after Arborio in Italy's Po Valley. Known for its high amylopectin content, it has a creamy texture ideal for risotto. The grains are short, fat, and roundish. When cooked, Arborio rice absorbs liquids well, making it perfect for creamy dishes like risotto and rice pudding. It’s sautéed in butter or oil before adding broth to achieve its signature texture. Nutritionally, it’s a good source of carbohydrates and some protein.")
    elif predicted_class == 'Karacadag':
        st.write("Karacadag Rice is a local variety of rice grown in the Karacadag region of Turkey. It is known for its high quality and is often preferred for its unique taste and aroma.")
