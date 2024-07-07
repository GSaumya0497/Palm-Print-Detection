# Palm-Print-Detection

## Overview

This project implements a palm print detection system using Python. The goal is to identify and verify palm prints from images using machine learning techniques. This README provides instructions on setting up the environment, running the code, and evaluating the results.

## Features

- **Image Preprocessing**: Convert images to grayscale and resize them.
- **Feature Extraction**: Extract features from palm print images for classification.
- **Model Training**: Train machine learning models such as SVM, ANN, and others.
- **Evaluation**: Evaluate model performance with accuracy, precision, recall, and confusion matrix.

## Requirements

- Python 3.x
- Required libraries: `numpy`, `opencv-python`, `scikit-learn`, `matplotlib`, `pandas`, `PIL` (or `Pillow`)

You can install the required libraries using `pip`:

```bash
pip install numpy opencv-python scikit-learn matplotlib pandas Pillow
```

## Dataset

- **Source**: You need a dataset of palm print images. You can use publicly available datasets like CASIA or PolyU or create your own dataset.
- **Format**: Images should be in `.jpg` or `.png` format.
- **Structure**: Organize images into directories, e.g., `positive` (for palm prints) and `negative` (for non-palm prints).

## Setup

1. **Download the Dataset**: Obtain the palm print dataset and place it in the appropriate directory structure.

2. **Prepare the Dataset**: Use the provided scripts to preprocess and organize the dataset.

## Usage

1. **Compress the Folder (if needed)**

   Compress your dataset folder into a zip file if required for uploading or sharing.

   ```python
   import shutil

   def compress_folder(folder_path, output_zip_path):
       shutil.make_archive(output_zip_path, 'zip', folder_path)

   # Example usage
   folder_path = 'path_to_your_folder'
   output_zip_path = 'path_to_your_output_zip_file'
   compress_folder(folder_path, output_zip_path)
   ```

2. **Run the Code**

   Run the following script to preprocess the images, train the model, and evaluate the results:

   ```python
   import cv2
   import numpy as np
   import os
   from sklearn.svm import SVC
   from sklearn.preprocessing import StandardScaler
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import classification_report, confusion_matrix
   from PIL import Image

   def preprocess_image(image_path, size=(128, 128)):
       image = Image.open(image_path).convert('L')  # Convert to grayscale
       image = image.resize(size)
       return np.array(image).flatten()  # Flatten the image to a 1D array

   def load_image_dataset(folder_path):
       data = []
       labels = []
       for filename in os.listdir(folder_path):
           if filename.endswith('.jpg') or filename.endswith('.png'):
               image_path = os.path.join(folder_path, filename)
               image = preprocess_image(image_path)
               if image is not None:
                   data.append(image)
                   label = 1 if 'positive' in folder_path else 0
                   labels.append(label)
       return np.array(data), np.array(labels)

   # Load dataset
   positive_folder = 'path_to_positive_folder'
   negative_folder = 'path_to_negative_folder'
   X, y = load_image_dataset(positive_folder) + load_image_dataset(negative_folder)

   # Split data
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

   # Normalize features
   scaler = StandardScaler()
   X_train = scaler.fit_transform(X_train)
   X_test = scaler.transform(X_test)

   # Train SVM model
   clf = SVC(kernel='linear')
   clf.fit(X_train, y_train)

   # Evaluate model
   y_pred = clf.predict(X_test)
   print("Classification Report:")
   print(classification_report(y_test, y_pred))
   print("Confusion Matrix:")
   print(confusion_matrix(y_test, y_pred))
   ```

3. **Visualize Results**

   Use the following code to visualize some test results:

   ```python
   import matplotlib.pyplot as plt

   def plot_images(images, labels, predictions=None, num_images=10):
       plt.figure(figsize=(10, 10))
       for i in range(num_images):
           plt.subplot(1, num_images, i + 1)
           plt.imshow(images[i].reshape(128, 128), cmap='gray')
           plt.title(f"Label: {labels[i]}")
           if predictions is not None:
               plt.xlabel(f"Pred: {predictions[i]}")
       plt.show()

   plot_images(X_test, y_test, predictions=y_pred)
   ```

## Notes

- Ensure you have the correct dataset paths and folder structure.
- Adjust preprocessing and model parameters as needed based on your dataset and requirements.

## Contributing

Feel free to contribute to this project by submitting issues or pull requests. Your contributions are welcome!

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For any questions or comments, please contact Saumya at saumya.cse001@gmail.com.
