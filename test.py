import os
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

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
    print(f"Confidence: {result['confidence']:.4f}")
    print()

def plot_predictions(num_images=5):
    plt.figure(figsize=(20, 4*num_images))
    for i, result in enumerate(test_results[:num_images]):
        image_path = os.path.join(test_path, result['image'])
        image = io.imread(image_path)

        plt.subplot(num_images, 2, 2*i+1)
        plt.imshow(image)
        plt.title(f"Image: {result['image']}")
        plt.axis('off')

        plt.subplot(num_images, 2, 2*i+2)
        plt.bar(classes, model.predict([process_image(image_path), process_image(image_path)]).mean(axis=0))
        plt.title(f"Predicted: {result['predicted_class']} (Confidence: {result['confidence']:.4f})")
        plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()

plot_predictions()

correct_predictions = sum(1 for result in test_results if result['predicted_class'] == result['image'].split('_')[0])
accuracy = correct_predictions / len(test_results)
print(f"Overall Accuracy: {accuracy:.4f}")

print('Coded by Majid khan \n follow on kaggle:"https://www.kaggle.com/majid0110" ')