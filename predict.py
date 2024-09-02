import os
import numpy as np
from skimage import io
from data import preprocess_image, create_patches, classes
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model = load_model('best_glnet_model.h5')

test_path = '/content/drive/MyDrive/bach-breast-cancer-histology-images/ICIAR2018_BACH_Challenge_TestDataset/Photos'

def process_image(image_path):
    image = io.imread(image_path)
    image = preprocess_image(image)
    patches = create_patches(image)
    return np.array(patches)

def predict_class(image_path):
    patches = process_image(image_path)
    predictions = model.predict([patches, patches])
    avg_prediction = np.mean(predictions, axis=0)
    predicted_class = np.argmax(avg_prediction)
    confidence = avg_prediction[predicted_class]
    return classes[predicted_class], confidence

test_results = []
for image_name in os.listdir(test_path):
    if image_name.endswith('.tif'):
        image_path = os.path.join(test_path, image_name)
        predicted_class, confidence = predict_class(image_path)
        test_results.append({
            'image': image_name,
            'predicted_class': predicted_class,
            'confidence': confidence
        })

for result in test_results:
    print(f"Image: {result['image']}")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.
    print(f"Confidence: {result['confidence']:.4f}")
    print("="*50)

def visualize_prediction(image_path):
    patches = process_image(image_path)
    predictions = model.predict([patches, patches])
    
    plt.figure(figsize=(12, 6))
    plt.suptitle(f"Predictions for {os.path.basename(image_path)}")

    for i, patch in enumerate(patches):
        plt.subplot(3, 4, i+1)
        plt.imshow(patch)
        predicted_class = np.argmax(predictions[i])
        confidence = predictions[i][predicted_class]
        plt.title(f"{classes[predicted_class]} ({confidence:.2f})")
        plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

example_image_path = os.path.join(test_path, 'sample_image.tif')
visualize_prediction(example_image_path)
