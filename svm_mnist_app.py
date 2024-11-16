import streamlit as st
import joblib  # For loading scikit-learn models
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler

# Load the SVM model
def load_model():
    model = joblib.load('svm_mnist_model.pkl')  # Load the trained SVM model
    return model

# Preprocess the image for prediction
def preprocess_image(image):
    # Convert the image to grayscale (if it's not already)
    image = image.convert("L")
    
    # Resize the image to 28x28 pixels, as required by the MNIST model
    image = image.resize((28, 28))
    
    # Convert the image to a numpy array and normalize the pixel values
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    
    # Flatten the image to a 1D array (28x28 -> 784)
    image_array = image_array.flatten()
    
    # Apply scaling if the model was trained with StandardScaler
    scaler = StandardScaler()
    image_array = scaler.fit_transform(image_array.reshape(-1, 1)).flatten()
    
    return image_array

# Prediction function for SVM model
def predict(image, model):
    image_array = preprocess_image(image)
    
    # Reshape the image for the model input (SVM expects 2D input)
    image_array = image_array.reshape(1, -1)  # SVM expects (n_samples, n_features)
    
    # Make prediction
    prediction = model.predict(image_array)
    return prediction[0]

def main():
    st.title('MNIST Digit Classifier (SVM)')
    st.write('Upload an image of a handwritten digit (0-9) and I will predict it.')
    
    # File uploader widget for uploading an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Open the image file
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)  
        st.write("")
        
        # Load the trained SVM model
        model = load_model()
        
        # Make a prediction
        prediction = predict(image, model)
        
        st.write(f"Prediction: {prediction}")
        
# Run the app
if __name__ == "__main__":
    main()
