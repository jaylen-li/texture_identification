#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Author https://github.com/boguss1225
Reference : https://github.com/calmisential/TensorFlow2.0_Image_Classification
"""
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import pandas as pd
import config
from utils.evaluate import eval_model
from utils.prepare_data import get_datasets, get_datasets_autosplit
from utils.pretrained_models import pretrained_model
import matplotlib.pyplot as plt


# ## Select GPU

# In[ ]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# os.environ["CUDA_HOME"]


# In[ ]:


# GPU settings
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# In[ ]:


gpus


# ## Select Model

# In[ ]:


available_models=["Xception",
                  "EfficientNetB0", "EfficientNetB1", "EfficientNetB2",
                  "EfficientNetB3", "EfficientNetB4", "EfficientNetB5",
                  "EfficientNetB6", "EfficientNetB7",
                  "EfficientNetV2B0", "EfficientNetV2B1",
                  "EfficientNetV2B2", "EfficientNetV2B3",
                  "EfficientNetV2S", "EfficientNetV2M", "EfficientNetV2L",
                  "VGG16","VGG19",
                  "DenseNet121", "DenseNet169", "DenseNet201",
                  "NASNetLarge","NASNetMobile",
                  "InceptionV3","InceptionResNetV2"
                  ]

def get_model():
    model = pretrained_model(model_name="EfficientNetB0",
                            load_weight=None)

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                  metrics=['accuracy', # add more metrics if you want
                            tf.keras.metrics.AUC(),
                            tf.keras.metrics.Precision(),
                            tf.keras.metrics.Recall(),
                            ])
    return model


# ## Load Data

# In[ ]:


# Load Data with manual data split
# train_generator, valid_generator, test_generator, \
# train_num, valid_num, test_num = get_datasets()

# Load Data with auto split
#train_generator, valid_generator, test_generator, \
#train_num, valid_num, test_num = get_datasets_autosplit()


# In[ ]:


"""
Apply image data augmentation in 'utils.prepare_data.py' manually
"""
config.image_height


# ## Visualize the data after augmentaion

# In[ ]:


"""
Here are the first 9 images in the training dataset. 
Label 0 :1_thick "Beggiatoa 1 (thick mat)"
Label 1 :2_patchy "Beggiatoa 2 (patchy)"
Label 2 :3_thick "Beggiatoa 3 (thin film)"
Label 3 :3_thin "Beggiatoa 3 (thin film)"
Label 4 :3_very_thin "Beggiatoa 3 (thin film)"
Label 5 :4_thin_worm "Worm 1 (Ophryotroca shieldsii - thin colony)"
Label 6 :5_big_worm "Worm 2 (Schistomeringos lovenii - thick worms)"
Label 7 :6_dark "Background"
Label 8 :6_number "Background"
Label 9 :6_poll "Background"
Label 10 :6_soil "Background"
Label 11 :6_steel "Background"
"""
if config.BATCH_SIZE > 9 :
    range_val = 9
else :
    range_val = config.BATCH_SIZE

plt.figure(figsize=(10, 10))
for i in range(range_val):
    ax = plt.subplot(3, 3, i + 1)
    img, label = next(train_generator) #img, label = train_generator.next()
    plt.imshow(img[0].astype("uint8"))
    plt.title(label[0].argmax())
    plt.axis("off")


# ## Callbacks

# In[ ]:


"""
Callbacks
"""

# tensorboard = tf.keras.callbacks.TensorBoard(log_dir='log')

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=config.model_dir + config.model_save_name + "_best.weights.h5",  # Change extension to .weights.h5
    #filepath=config.model_dir+config.model_save_name+"_best.keras",
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="loss",
    patience=20,
    restore_best_weights=True
)

callbacks = [
#     tensorboard, 
    model_checkpoint_callback, 
    early_stop_callback
]


# In[ ]:


model = get_model()
# tf.keras.utils.plot_model(model, show_shapes=True)


# ## Train the model

# In[ ]:


# mkdir for model save path
if not os.path.exists(config.model_dir):
    os.makedirs(config.model_dir)

print("Start training:",config.train_dir)
print("Model:", config.model_save_name)

history = model.fit(train_generator,
                    epochs=config.EPOCHS,
                    steps_per_epoch=train_num // config.BATCH_SIZE,
                    validation_data=train_generator,
                    validation_steps=valid_num // config.BATCH_SIZE,
                    callbacks=callbacks)


# ## Save Model

# In[ ]:


# save the whole model
model.save(config.model_dir+config.model_save_name+"_last.h5")


# In[ ]:


hist_df = pd.DataFrame(history.history)
with open(config.model_dir+"train_history.csv", mode='w') as f:
    hist_df.to_csv(f)


# ## Evaluate Model

# In[ ]:


# Evaluation
eval_model(model)


# ## Inference Example

# In[ ]:


import numpy as np
def test_single_image(img_dir, model):
    img_raw = tf.io.read_file(img_dir)
    img_tensor = tf.image.decode_jpeg(img_raw, channels=config.channels)
    img_tensor = tf.image.resize(img_tensor, [config.image_height, config.image_width])

    img_numpy = img_tensor.numpy()
    img_numpy = (np.expand_dims(img_numpy, 0))
    img_tensor = tf.convert_to_tensor(img_numpy, tf.float32)

#     img_tensor = img_tensor / 255.0 # uncomment if model included rescale preprocessing layer
    prob = model(tf.image.resize(img_tensor,[config.image_width,config.image_height]))

    
    probability = np.max(prob)

    classification = np.argmax(prob)
    return classification, probability


# In[ ]:


# detect samples w last model
last_model = model
print(config.test_image_path)
classification_result, probability = test_single_image(config.test_image_path, last_model)
print("class : ",classification_result+1,"of",probability,"%")


# In[ ]:


# detect samples w best model
# best_model = tf.keras.models.load_model(config.model_dir+config.model_save_name+"_best.h5")
best_model = get_model()
best_model.load_weights(config.model_dir+config.model_save_name+"_best.keras")
print(config.test_image_path)
classification_result, probability = test_single_image(config.test_image_path, best_model)
print("class : ",classification_result+1,"of",probability,"%")


# In[ ]:


eval_model(best_model)


# In[ ]:




