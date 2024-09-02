import os
import numpy as np
from skimage import io, filters, exposure

base_path = '/kaggle/input/bach-breast-cancer-histology-images/ICIAR2018_BACH_Challenge/ICIAR2018_BACH_Challenge/Photos'
classes = ['Benign', 'InSitu', 'Invasive', 'Normal']

def preprocess_image(image):
    image = exposure.adjust_gamma(image, 0.8)
    image = filters.unsharp_mask(image, radius=5, amount=2)
    image = filters.median(image, footprint=np.ones((3, 3, 1)))
    return image

def create_patches(image, patch_size=(250, 200), num_patches=(3, 4)):
    h, w = image.shape[:2]
    patches = []
    for i in range(num_patches[0]):
        for j in range(num_patches[1]):
            y = int(i * h / num_patches[0])
            x = int(j * w / num_patches[1])
            patch = image[y:y+patch_size[0], x:x+patch_size[1]]
            if patch.shape[:2] == patch_size:
                patches.append(patch)
    return patches

def load_data(base_path=base_path, classes=classes):
    images = []
    labels = []
    for idx, class_name in enumerate(classes):
        class_path = os.path.join(base_path, class_name)
        for img_name in os.listdir(class_path):
            if img_name.endswith('.tif'):
                img_path = os.path.join(class_path, img_name)
                image = io.imread(img_path)
                image = preprocess_image(image)
                patches = create_patches(image)
                images.extend(patches)
                labels.extend([idx] * len(patches))
    return np.array(images), np.array(labels)
