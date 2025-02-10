import streamlit as st
import random
import numpy as np
from PIL import Image, ImageDraw, ImageOps

# Function to add label to an image or create a blank image with a label
def add_label_to_image(image, label, size=(300, 300)):
    if image is None:
        # Create a blank image if no image is uploaded
        img = Image.new("RGB", size, color="gray")
    else:
        img = image.copy()
        img = img.resize(size)
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), label, fill="white")
    return img

# Streamlit UI
st.title("Image Processor with Random Outputs")

# Drag-and-drop file upload
uploaded_image = st.file_uploader(
    "Drag and drop an image here (PNG, JPG, JPEG)",
    type=[],
)

image = None
if uploaded_image:
    try:
        # Load the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    except Exception as e:
        st.error(f"Error loading the image: {e}")

# Submit button
if st.button("Submit"):
    # Generate a random number
    random_number = random.randint(1, 100)
    st.subheader(f"Random Number: {random_number}")

    # Generate 12 labeled images
    st.subheader("Generated Images:")
    cols = st.columns(6)
    images = []
    for i in range(1, 13):
        label = f"Layer-{i}" if i <= 6 else f"Image-{i-6}"
        labeled_image = add_label_to_image(image, label)
        images.append((label, labeled_image))

    # Display 12 images
    for idx, (label, img) in enumerate(images):
        cols[idx % 6].image(img, caption=label, use_column_width=True)

    # Generate 12 random numbers for bar chart
    random_numbers = np.random.randint(1, 101, 12)

    # Display bar chart
    st.subheader("Bar Chart of Random Numbers:")
    st.bar_chart(random_numbers)

# If no image is uploaded, show blank images when the button is clicked
elif not uploaded_image:
    st.info("No image uploaded. Blank images will be generated when submitted.")


# modified_vit_model
