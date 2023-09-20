import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import rawpy
import imageio
import os


# Function to get the class label from class index
def get_class_label(class_index):
    class_labels = {
        0: "a",
        1: "b",
        2: "c",
        3: "jj"
    }
    return class_labels.get(class_index, "Unknown")


# Function to get description based on confidence
def get_confidence_description(confidence):
    if confidence >= 90:
        return "Very confident"
    elif confidence >= 70:
        return "Confident"
    elif confidence >= 50:
        return "Moderately confident"
    else:
        return "Less confident"


# Function to convert a JPEG image to a simulated DNG
def convert_jpeg_to_dng(jpeg_path, dng_path):
    with rawpy.imread(jpeg_path) as raw:
        rgb = raw.postprocess()
        imageio.imsave(dng_path, rgb)


st.set_page_config(page_title="Areca.ai")


def main():
    st.title("Areca Nut Classification")

    # Upload the image (JPEG or DNG)
    uploaded_file = st.file_uploader("Upload Image (JPEG or DNG)", type=["jpg", "dng"])

    # Check if an image is provided
    if uploaded_file is not None:
        # Check the file extension
        file_extension = os.path.splitext(uploaded_file.name)[-1].lower()

        if file_extension == ".jpg":
            # Convert the JPEG to a simulated DNG
            jpeg_path = "uploaded.jpg"
            dng_path = "uploaded.dng"
            with open(jpeg_path, "wb") as f:
                f.write(uploaded_file.read())
            convert_jpeg_to_dng(jpeg_path, dng_path)

            # Display the converted DNG image
            st.image(dng_path, caption="Converted DNG Image", use_column_width=True)

        # Preprocess the image for classification (you can add your classification code here)
        if file_extension in [".jpg", ".dng"]:
            # Load the image
            if file_extension == ".jpg":
                image_path = dng_path  # Use the converted DNG for classification
            else:
                image_path = uploaded_file
            new_image = load_img(image_path, target_size=(380, 380))
            new_image_array = img_to_array(new_image)
            new_image_array = np.expand_dims(new_image_array, axis=0)
            new_image_array = new_image_array / 255.0

            # Load the MobilenetV2 models
            model_paths = [
                r"mobilenetv2_model.h5",
                #r"C:\Users\aniru\PycharmProjects\areca nut model fp\mobilenetv2_model (1).h5",
                r"custom_model.h5",
                #r"custom_model (2).h5",
                #"fine_tuned_model.h5",
                #"trained_model_01.h5",
                #"trained_model.h5",
               # r"C:\Users\aniru\PycharmProjects\areca nut model fp\resnet50_model.h5",
                #r"C:\Users\aniru\PycharmProjects\areca nut model fp\efficientnetb4_model.h5",
                #r"C:\Users\aniru\PycharmProjects\areca nut model fp\vgg16_model.h5",

            ]

            model_results = []

            for model_path in model_paths:
                with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
                    tf.compat.v1.keras.backend.set_session(sess)
                    model = load_model(model_path)
                    predictions = model.predict(new_image_array)
                    predicted_class_index = np.argmax(predictions)
                    predicted_class = get_class_label(predicted_class_index)
                    confidence = np.max(predictions) * 100
                    model_results.append({
                        'Model Name': model_path,
                        'Predicted Class': predicted_class,
                        'Probability': confidence
                    })

            # Display the classification results for each model
            st.write("Model Comparisons:")
            for result in model_results:
                st.write(f"Model: {result['Model Name']}")
                st.write("Predicted Class:", result['Predicted Class'])
                st.write("Confidence:", result['Probability'])
                st.write("Confidence Level:", get_confidence_description(result['Probability']))

            # Calculate the final prediction based on the model with the highest confidence
            final_prediction = max(model_results, key=lambda x: x['Probability'])
            st.write("Final Prediction:")
            st.write("Predicted Class:", final_prediction['Predicted Class'])
            st.write("Confidence:", final_prediction['Probability'])
            st.write("Confidence Level:", get_confidence_description(final_prediction['Probability']))


if __name__ == "__main__":
    main()