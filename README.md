 

🖋️ MNIST Handwritten Digit Classifier 🖋️
==========================================

This is a simple web application that uses a **Support Vector Machine (SVM)** model to classify handwritten digits (0-9) based on images uploaded by users. The app is built using **Streamlit** and works by preprocessing the uploaded images and predicting the digit using a pre-trained SVM model.

🔍 Features:
------------

*   🖼️ Upload images of handwritten digits in `.jpg`, `.png`, or `.jpeg` format.
*   ⚙️ Preprocessing steps such as resizing, grayscale conversion, and thresholding are applied.
*   🤖 The SVM model predicts the digit and displays the result to the user.
*   🌐 Built using **Streamlit** for a simple web interface.

🚀 How to Run the App:
----------------------

1.  Clone the repository: `git clone https://github.com/your-username/mnist-digit-classifier.git`
2.  Install the necessary libraries:
    
        pip install -r requirements.txt
    
3.  Ensure the pre-trained model file `svm_mnist_model.pkl` is in the project folder (if not, train the model and save it).
4.  Run the Streamlit app:
    
        streamlit run app.py
    
5.  Open the app in your browser and start uploading handwritten digit images for prediction!

📂 File Structure:
------------------

*   **app.py** - Main Streamlit app file with UI and logic for prediction.
*   **svm\_mnist\_model.pkl** - Pre-trained SVM model for digit classification (ensure you have this file).
*   **requirements.txt** - List of dependencies for the app (e.g., Streamlit, scikit-learn, Pillow).
*   **README.md** - This file 😊.

📈 How It Works:
----------------

The app processes the uploaded image by performing the following steps:

1.  Converts the image to **grayscale** to match the input format of the MNIST dataset.
2.  Resizes the image to **28x28 pixels**, the size expected by the model.
3.  Uses **SVM classifier** to predict the digit based on the processed image.
4.  Displays the predicted digit on the app.

🛠️ Technologies Used:
----------------------

*   **Streamlit** - For creating the web interface.
*   **Scikit-learn (SVM)** - For the machine learning model (SVM classifier) used to predict the digits.
*   **Pillow** - For image preprocessing tasks such as resizing, grayscale conversion, and binarization.
*   **Joblib** - For saving and loading the trained model.

📊 Model Performance:
---------------------

The SVM model is trained on the MNIST dataset, achieving an accuracy of around **92%**. However, performance can vary based on the quality and preprocessing of the input image. Ensure the image is clearly visible for better results!

📝 Troubleshooting:
-------------------

*   If you encounter the error **“module 'PIL.Image' has no attribute 'ANTIALIAS'”**, update your image resizing code to use `Image.Resampling.LANCZOS` (modern replacement for `ANTIALIAS`).
*   Ensure the model file `svm_mnist_model.pkl` is correctly placed in the project directory for the app to load the pre-trained model.

📥 Contributions:
-----------------

Feel free to open issues or submit pull requests if you have suggestions for improvements or new features. All contributions are welcome! 🙌

📄 License:
-----------

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

Web app live link:
-----------

https://handwritten-digit-recongition-using-svm-ajscvpmbz2sbzjdgwwyyzp.streamlit.app/


Made with ❤️ by Akash Anandani 👨‍💻
