import tensorflow as tf
from keras.applications.vgg16 import VGG16
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Model
# from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
import visualkeras

from ModelInterface import ModelInterface


class CNNKeras(ModelInterface):
    def __init__(self, train_ds_path, val_ds_path, batch_size, image_size, epochs, model_save_path, model):
        self.BATCH_SIZE = batch_size
        self.IMAGE_SIZE = image_size
        self.EPOCHS = epochs
        self.train_ds_path = train_ds_path
        self.val_ds_path = val_ds_path
        self.model_save_path = model_save_path
        self.train_dataset = 0
        self.val_dataset = 0
        self.train_dataset1 = 0
        self.val_dataset1 = 0
        self.class_names = 0
        self.data_augmentation = 0
        self.num_classes = 0
        self.model = model

    def read_dataset(self):
        self.train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            self.train_ds_path,
            shuffle=True,
            image_size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
            batch_size=self.BATCH_SIZE
        )
        self.val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            self.val_ds_path,
            shuffle=True,
            image_size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
            batch_size=self.BATCH_SIZE
        )
        self.class_names = self.train_dataset.class_names
        self.num_classes = len(self.class_names)
        # self.train_dataset = self.train_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        # self.val_dataset = self.val_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    def augment_data(self):
        self.data_augmentation = tf.keras.Sequential(
            [
                layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical",
                                                             input_shape=(self.IMAGE_SIZE,
                                                                          self.IMAGE_SIZE,
                                                                          3)),
                layers.experimental.preprocessing.RandomRotation(0.1),
                layers.experimental.preprocessing.RandomZoom(0.1),
            ]
        )

    def compile_model(self):

        # self.model = Sequential([
        #     self.data_augmentation,
        #     layers.Rescaling(1. / 255, input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3)),
        #     layers.Conv2D(16, 3, padding='same', activation='relu'),
        #     layers.MaxPooling2D(),
        #     layers.Conv2D(32, 3, padding='same', activation='relu'),
        #     layers.MaxPooling2D(),
        #     layers.Conv2D(64, 3, padding='same', activation='relu'),
        #     layers.MaxPooling2D(),
        #     layers.Dropout(0.2),
        #     layers.Flatten(),
        #     layers.Dense(128, activation='relu'),
        #     layers.Dense(1, activation='sigmoid')
        # ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                           metrics=['accuracy',
                                    tf.keras.metrics.Precision(name='precision'),
                                    tf.keras.metrics.Recall(name='recall'),
                                    tf.keras.metrics.SpecificityAtSensitivity(0.5, name='specificity_at_sensitivity')])
        self.model.summary()
        visualkeras.layered_view(self.model, legend=True, to_file="output.png").show()

    def train(self):
        history = self.model.fit(
            self.train_dataset,
            epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            validation_data=self.val_dataset,
            verbose=1,
        )
        return history

    def save_model(self):
        self.model.save(self.model_save_path)
