import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Conv2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.utils import Sequence

input_shape = (250, 200, 3)

def global_feature_extractor(input_shape):
    base_model = tf.keras.applications.ResNet101(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    return Model(inputs=base_model.input, outputs=x)

def local_feature_extractor(input_shape):
    base_model = tf.keras.applications.ResNet101(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    return Model(inputs=base_model.input, outputs=x)

def build_glnet_model(input_shape, num_classes=4):
    global_input = Input(shape=input_shape)
    global_features = global_feature_extractor(input_shape)(global_input)

    local_input = Input(shape=input_shape)
    local_features = local_feature_extractor(input_shape)(local_input)

    combined_features = tf.keras.layers.Concatenate()([global_features, local_features])

    x = Dense(512, activation='relu')(combined_features)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[global_input, local_input], outputs=output)

    return model

class GLNETDataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size=32, shuffle=True):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_x = self.x[indexes]
        batch_y = self.y[indexes]
        return [batch_x, batch_x], batch_y  # Return same input for both global and local

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.x))
        if self.shuffle:
            np.random.shuffle(self.indexes)

def compile_glnet_model(input_shape):
    model = build_glnet_model(input_shape)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
