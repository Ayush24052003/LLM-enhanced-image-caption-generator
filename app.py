import streamlit as st
import numpy as np
import pickle
import together
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import os
from dotenv import load_dotenv
load_dotenv()

# Loading VGG16 model for feature extraction
vgg_model = VGG16(weights="imagenet")
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Loading my trained model
model = tf.keras.models.load_model('mymodel.h5')

# Loading the tokenizer
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)
    
# Set custom web page title
st.set_page_config(page_title="Caption Generator App", page_icon="📷")

# Streamlit app
st.title("Image Caption Generator")
st.markdown(
    "Upload an image, and it will generate a caption for it."
)

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Process uploaded image
if uploaded_image is not None:
    st.subheader("Uploaded Image")
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Generated Caption")
    # Display loading spinner while processing
    with st.spinner("Generating caption..."):
        # Loading image
        image = load_img(uploaded_image, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        # Extract features using VGG16
        image_features = vgg_model.predict(image, verbose=0)

        # Max caption length
        max_caption_length = 34
        
        # Define function to get word from index
        def get_word_from_index(index, tokenizer):
            return next(
                (word for word, idx in tokenizer.word_index.items() if idx == index), None
        )

        # Generate caption using the model
        def predict_caption(model, image_features, tokenizer, max_caption_length):
            caption = "startseq"
            for _ in range(max_caption_length):
                sequence = tokenizer.texts_to_sequences([caption])[0]
                sequence = pad_sequences([sequence], maxlen=max_caption_length)
                yhat = model.predict([image_features, sequence], verbose=0)
                predicted_index = np.argmax(yhat)
                predicted_word = get_word_from_index(predicted_index, tokenizer)
                caption += " " + predicted_word
                if predicted_word is None or predicted_word == "endseq":
                    break
            return caption

        # Generate caption
        generated_caption = predict_caption(model, image_features, tokenizer, max_caption_length)

        # Initialize Together client
        together.api_key = os.getenv("TOGETHER_API_KEY")


        def refine_caption_with_llm(captions, target_lang="english"):
            captions_text = "\n".join([f"{i + 1}. {cap}" for i, cap in enumerate(captions)])

            prompt = f"""You are an expert creative writer specializing in image captions.
        I will give you {len(captions)} automatically generated captions for the same image.

        Your tasks are:
        1. Understand the meaning across all captions.
        2. Merge the best parts into one natural, polite, and elegant caption.
        3. Keep it concise (1–2 sentences maximum).
        4. Do not mention that these captions were given by a model.
        5. Output only the final caption in {target_lang}.

        Here are the generated captions:
        {captions_text}
        """

            response = together.Complete.create(
                model="mistral-7b-instruct",
                prompt=prompt,
                max_tokens=60,
                temperature=0.7,
                top_p=0.9
            )

            return response['output']['choices'][0]['text'].strip()

        # Remove start and end tokens
        generated_caption = generated_caption.replace("startseq", "").replace("endseq", "").strip()

        # Refine with Together AI (Mistral)
        refined_caption = refine_caption_with_llm([generated_caption], target_lang="english")

        # Display raw caption
        st.subheader("Raw Caption (Model Output)")
        st.write(generated_caption)

        # Display refined caption
        st.subheader("Refined Caption (Mistral)")
        st.markdown(
            f'<div style="border-left: 6px solid #4CAF50; padding: 5px 20px; margin-top: 20px;">'
            f'<p style="font-style: italic;">“{refined_caption}”</p>'
            f'</div>',
            unsafe_allow_html=True
        )