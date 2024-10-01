import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import io

# Initialize the BLIP model and processor for image captioning
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize the sentence transformer model for embedding generation
embedding_model = SentenceTransformer('all-mpnet-base-v2')

def generate_detailed_image_description(img):
    try:
        # Process the image using BlipProcessor
        inputs = caption_processor(images=img, return_tensors="pt", padding=True)
        
        # Generate description
        out = caption_model.generate(
            **inputs,
            min_length=15,
            max_length=150,
            num_beams=5,
            repetition_penalty=2.5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        
        # Decode and return the description
        description = caption_processor.decode(out[0], skip_special_tokens=True)
        return description
    
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def get_text_embedding(text):
    return embedding_model.encode(text)

def compute_cosine_similarity(text1, text2):
    embedding1 = get_text_embedding(text1)
    embedding2 = get_text_embedding(text2)
    similarity_score = cosine_similarity([embedding1], [embedding2])
    return similarity_score[0][0]

def plot_images_with_descriptions(img1, img2, desc1, desc2, similarity_score):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot images
    axes[0].imshow(img1)
    axes[0].set_title("Image 1")
    axes[0].axis('off')
    axes[1].imshow(img2)
    axes[1].set_title("Image 2")
    axes[1].axis('off')

    # Add descriptions as subtitles
    fig.suptitle(f"Similarity Score: {similarity_score:.2f}", fontsize=16)
    axes[0].set_xlabel(desc1, fontsize=12)
    axes[1].set_xlabel(desc2, fontsize=12)

    st.pyplot(fig)

# Streamlit app
st.title("Image Similarity Application")

# Upload images
uploaded_image1 = st.file_uploader("Choose the first image...", type=["jpg", "jpeg", "png"])
uploaded_image2 = st.file_uploader("Choose the second image...", type=["jpg", "jpeg", "png"])

if uploaded_image1 and uploaded_image2:
    # Open the images
    img1 = Image.open(uploaded_image1).convert("RGB")
    img2 = Image.open(uploaded_image2).convert("RGB")
    
    # Generate descriptions
    desc1 = generate_detailed_image_description(img1)
    desc2 = generate_detailed_image_description(img2)

    if desc1 and desc2:
        st.write("Description of Image 1:", desc1)
        st.write("Description of Image 2:", desc2)
        
        # Compute cosine similarity
        similarity_score = compute_cosine_similarity(desc1, desc2)
        st.write(f"Similarity Score: {similarity_score:.2f}")
        
        # Plot images with descriptions and similarity score
        plot_images_with_descriptions(img1, img2, desc1, desc2, similarity_score)
    else:
        st.error("Error generating descriptions for one or both images.")
