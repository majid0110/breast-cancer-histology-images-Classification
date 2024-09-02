import os
from data import load_data, base_path, classes
from model import compile_glnet_model, GLNETDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

X, y = load_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_generator = GLNETDataGenerator(X_train, y_train, batch_size=32)
test_generator = GLNETDataGenerator(X_test, y_test, batch_size=32, shuffle=False)

input_shape = (250, 200, 3)
glnet_model = compile_glnet_model(input_shape)

checkpoint = ModelCheckpoint('best_glnet_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = glnet_model.fit(
    train_generator,
    epochs=100,
    validation_data=test_generator,
    callbacks=[checkpoint, early_stopping]
)

test_loss, test_accuracy = glnet_model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy:.4f}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
