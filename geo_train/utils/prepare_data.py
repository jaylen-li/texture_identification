import tensorflow as tf
import config
import cv2
import numpy as np
from PIL import Image

def lab_preprocessing(image):
    image = np.array(image)
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    lab_image = (lab_image * 255).astype(np.uint8)
    return Image.fromarray(lab_image)

def get_datasets(set):
    train_dir = f"D:/project_geo/code_test_combined/5folds_with_test_updated_combined_ver3/set_{set}/train"
    val_dir = f"D:/project_geo/code_test_combined/5folds_with_test_updated_combined_ver3/set_{set}/val"
    test_dir = "D:/project_geo/code_test_combined/5folds_with_test_updated_combined_ver3/test"

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        # rescale=1.0 / 255.0  
    )

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(config.image_height, config.image_width),
                                                        color_mode="rgb",
                                                        batch_size=config.BATCH_SIZE,
                                                        seed=1,
                                                        shuffle=True,  # Still shuffling the data
                                                        class_mode="categorical")

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        # rescale=1.0 / 255.0
    )

    valid_generator = valid_datagen.flow_from_directory(val_dir,
                                                        target_size=(config.image_height, config.image_width),
                                                        color_mode="rgb",
                                                        batch_size=config.BATCH_SIZE,
                                                        seed=1,
                                                        shuffle=True,
                                                        class_mode="categorical")

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        # rescale=1.0 / 255.0
    )

    test_generator = test_datagen.flow_from_directory(test_dir,
                                                      target_size=(config.image_height, config.image_width),
                                                      color_mode="rgb",
                                                      batch_size=config.BATCH_SIZE,
                                                      seed=1,
                                                      shuffle=False,  # No shuffle for testing
                                                      class_mode="categorical")

    train_num = train_generator.samples
    valid_num = valid_generator.samples
    test_num = test_generator.samples

    return train_generator, \
           valid_generator, \
           test_generator, \
           train_num, valid_num, test_num

def get_datasets_autosplit():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        validation_split=0.2  
        # rescale=1.0 / 255.0
    )

    train_generator = train_datagen.flow_from_directory(config.train_dir,
                                                        target_size=(config.image_height, config.image_width),
                                                        color_mode="rgb",
                                                        batch_size=config.BATCH_SIZE,
                                                        seed=1,
                                                        shuffle=True,
                                                        class_mode="categorical",
                                                        subset='training')

    valid_generator = train_datagen.flow_from_directory(config.train_dir,
                                                        target_size=(config.image_height, config.image_width),
                                                        color_mode="rgb",
                                                        batch_size=config.BATCH_SIZE,
                                                        seed=7,
                                                        shuffle=True,
                                                        class_mode="categorical",
                                                        subset='validation')

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        # rescale=1.0 / 255.0
    )

    test_generator = test_datagen.flow_from_directory(config.test_dir,
                                                      target_size=(config.image_height, config.image_width),
                                                      color_mode="rgb",
                                                      batch_size=config.BATCH_SIZE,
                                                      seed=1,
                                                      shuffle=False,
                                                      class_mode="categorical")

    train_num = train_generator.samples
    valid_num = valid_generator.samples
    test_num = test_generator.samples

    return train_generator, \
           valid_generator, \
           test_generator, \
           train_num, valid_num, test_num
