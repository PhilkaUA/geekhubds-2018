#!/usr/bin/env python
# coding: utf-8

# **GeekHub 2018-2019 DL**
# 
# **Home work 13 Neural Networks**
# 
# Kaggle competion: https://www.kaggle.com/c/geekhub-ds-2019-challenge

# Kaggle for colab

# In[1]:


# import json
# import kaggle

# !mkdir .kaggle
# token = {"username":"philkaua","key":""}
# with open('/content/.kaggle/kaggle.json', 'w') as file:
#     json.dump(token, file)
# !cp /content/.kaggle/kaggle.json ~/.kaggle/kaggle.json
# !kaggle config set -n path -v{/content}
# !chmod 600 /root/.kaggle/kaggle.json


# In[6]:


# Import numpy, keras submodules and some other modules

import cv2
import imageio
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
import tensorflow as tf
import random

#from google.colab import drive
from glob import glob
from imgaug import augmenters as iaa
from keras import layers, initializers, optimizers, regularizers
from keras import datasets, models, callbacks, applications, utils
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.ERROR)


# Load data from repository

# In[7]:


#!kaggle competitions download -c geekhub-ds-2019-challenge
# !wget https://www.dropbox.com/s/ljcgvyjo95ogjs8/train_labels.csv
# !wget https://www.dropbox.com/s/bn4hedoc1q2tptc/train.zip
# !unzip -q "train.zip"
# !wget https://www.dropbox.com/s/n87ike3hk8ybkio/test.zip
# !unzip -q "test.zip"


# File operations - load data

# In[8]:


#Train and Test labels

# File operations
PATH_TO_DATA = os.path.abspath(os.curdir)
path_to_files_train = 'train/*'
path_to_files_test = 'test/*'

train_csv_file = pd.read_csv('train_labels.csv')['Category']

file_train = [file for file in glob(path_to_files_train)]
file_test = [file for file in glob(path_to_files_test)]

# Classes
class_names = np.unique(train_csv_file)
class_dict = {class_name:ind for ind, class_name in enumerate(class_names)}

# labels
train_labels = train_csv_file.map(class_dict).values

print(f'Classes:{class_dict}\n')
print(f'Train labels:{train_labels.shape}\n')


for cls, label in class_dict.items():
    print(f'{cls} => {len(np.where(train_labels == label)[0])}')


# Preprocessing

# In[18]:


get_ipython().run_cell_magic('time', '', '# load images\nim_size_1 = 128\nim_size_2 = 128\n\nlabled_images = np.array([image.img_to_array( image.load_img(file, target_size=(im_size_1, im_size_2)) )\\\n                          for file in sorted(file_train)])\ntest_images = np.array([image.img_to_array( image.load_img(file, target_size=(im_size_1, im_size_2)) )\\\n                        for file in sorted(file_test)])')


# In[13]:


# Scalling image
labled_images_scaled = labled_images / 255.
test_labled_images_scaled = test_images / 255.

# To binary image
labled_images_scaled_binary = np.array([ rgb2gray(imgs) for imgs in labled_images_scaled])
test_labled_images_scaled_binary = np.array([ rgb2gray(imgs) for imgs in labled_images_scaled])

# For colour use
labled_images_scaled_binary = labled_images_scaled
test_labled_images_scaled_binary = test_labled_images_scaled

print(f'Images scaled:{labled_images_scaled.shape}')
print(f'Images scaled binary:{labled_images_scaled_binary.shape}')


# Splitting data train, validation, test

# In[15]:


X_train, X_test_all, y_train, y_test_all = train_test_split(labled_images_scaled_binary,                                                            train_labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test_all, y_test_all, test_size=0.3, random_state=42)


# Image example

# In[16]:


print("Train shape:", X_train.shape)
print("Unique values in labels:", np.unique(train_labels))
print("Test shape:", X_test.shape)
print("Train images min:", np.min(X_train), "max:", np.max(X_train))

print("First image:")
plt.figure()
plt.imshow(X_train[0], cmap=plt.cm.binary)
plt.grid(False)
plt.show()

print("Class for this image:", class_names[y_train[0]])


# In[17]:


plt.figure(figsize=(15,15))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
plt.show()


# **MODELs**

# Danse layers

# In[21]:


def getModel(input_shape):
    inp = layers.Input(input_shape)
    flatten = layers.Flatten()(inp) # (batch, width, height) -> (batch, length)
    fc = layers.Dense(75, activation='relu')(flatten)
    fc = layers.Dense(50, activation='relu')(flatten)
    out = layers.Dense(5, activation='softmax')(fc)
    return models.Model(inp, out)
  
model = getModel((im_size_1, im_size_2, 3))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          batch_size=15,
          epochs=10)


# In[22]:


model.evaluate(X_test, y_test)


# In[34]:


model.summary()


# CNN - 3 layers Conv + Dense

# In[25]:


def getModel(input_shape):
    
    inp = layers.Input(input_shape)
  
    conv = layers.Conv2D(32, 3, padding='same')(inp) #filters=32, kernel size=3
    conv = layers.Activation('relu')(conv)
    conv = layers.Conv2D(32, 3, padding='same')(conv)
    conv = layers.Activation('relu')(conv)
    pool = layers.MaxPool2D()(conv) #default pool_size=2
  
    conv = layers.Conv2D(64, 3, padding='same')(pool)
    conv = layers.Activation('relu')(conv)
    conv = layers.Conv2D(64, 3, padding='same')(conv)
    conv = layers.Activation('relu')(conv)
    pool = layers.MaxPool2D()(conv) 
   
    conv = layers.Conv2D(128, 3, padding='same')(pool) 
    conv = layers.Activation('relu')(conv)
    conv = layers.Conv2D(128, 3, padding='same')(conv)
    conv = layers.Activation('relu')(conv)
  
    pool = layers.GlobalMaxPool2D()(conv)  
    fc = layers.Dense(100)(pool)
    fc = layers.Activation('relu')(fc)
    fc = layers.Dense(40, activation='relu')(fc)
    fc = layers.Dense(5)(fc)
    out = layers.Activation('softmax')(fc)
    
    return models.Model(inp, out)

model = getModel((im_size_1, im_size_2, 3))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          batch_size=15,
          epochs=15
         )


# In[26]:


model.evaluate(X_test, y_test)


# CNN - 3 layers Conv + Dropouts + Dense

# In[31]:


def getModel(input_shape):
    
    inp = layers.Input(input_shape)
  
    conv = layers.Conv2D(32, (3, 3), padding='same')(inp) #filters=32, kernel size=3
    conv = layers.Activation('relu')(conv)
    pool = layers.MaxPool2D(pool_size=(2, 2))(conv) #default pool_size=2
    
    conv = layers.Conv2D(32, (3, 3), padding='same')(pool)
    conv = layers.Activation('relu')(conv)
    pool = layers.MaxPool2D(pool_size=(2, 2))(conv)
    
    conv = layers.Conv2D(64, (3, 3), padding='same')(pool)
    conv = layers.Activation('relu')(conv)
    pool = layers.MaxPool2D(pool_size=(2, 2))(conv) 
   
    #pool = layers.GlobalMaxPool2D()(pool)  
            
    flatten = layers.Flatten()(pool)
        
    fc = layers.Dense(40)(flatten)
    fc = layers.Activation('relu')(fc)
    fc = layers.Dropout(0.1)(fc)
    fc = layers.Dense(5)(fc)
    out = layers.Activation('softmax')(fc)
    
    return models.Model(inp, out)

model = getModel((im_size_1, im_size_2, 3))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          batch_size=15,
          epochs=15
         )


# In[32]:


model.evaluate(X_test, y_test)


# In[36]:


#model.summary()


# CNN - 3 layers Conv + Dropouts + Dense + Callacks

# In[37]:


def getModel(input_shape):
    inp = layers.Input(input_shape)
  
    conv = layers.Conv2D(32, 3, padding='same')(inp) #filters=32, kernel size=3
    conv = layers.Activation('relu')(conv)
    conv = layers.Conv2D(32, 3, padding='same')(conv)
    conv = layers.Activation('relu')(conv)
    pool = layers.MaxPool2D()(conv) #default pool_size=2
  
    conv = layers.Conv2D(64, 3, padding='same')(pool)
    conv = layers.Activation('relu')(conv)
    conv = layers.Conv2D(64, 3, padding='same')(conv)
    conv = layers.Activation('relu')(conv)
    pool = layers.MaxPool2D()(conv) 
  
    conv = layers.Conv2D(128, 3, padding='same')(pool) 
    conv = layers.Activation('relu')(conv)
    conv = layers.Conv2D(128, 3, padding='same')(conv)
    conv = layers.Activation('relu')(conv)
  
    pool = layers.GlobalMaxPool2D()(conv)
    pool = layers.Dropout(0.2)(pool)
    fc = layers.Dense(40)(pool)
    fc = layers.Activation('relu')(fc)
    fc = layers.Dropout(0.1)(fc)
    fc = layers.Dense(5)(fc)
    out = layers.Activation('softmax')(fc)   
    return models.Model(inp, out)
  
model = getModel((im_size_1, im_size_2, 3))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train.reshape(-1, im_size_1, im_size_2, 3), y_train,
          validation_data=(X_val.reshape(-1, im_size_1, im_size_2, 3), y_val),
          batch_size=15,
          epochs=35,
          callbacks=[
              callbacks.ModelCheckpoint('weights.h5', verbose=1, save_best_only=True, save_weights_only=True),
              callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1),
              callbacks.EarlyStopping(patience=4, verbose=1)
          ]
         )

model.load_weights('weights.h5') 


# In[38]:


model.evaluate(X_test, y_test)


# CNN - BatchNormalization, Callacks , Multi outputmodel

# In[40]:


regularization = regularizers.L1L2(0, 1e-4)

def res_block(x):
    count = int(x.shape[-1])
  
    conv = layers.Conv2D(count, 3, padding='same', kernel_regularizer=regularization)(x)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation('relu')(conv)
    conv = layers.Conv2D(count, 3, padding='same', kernel_regularizer=regularization)(conv)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation('relu')(conv)
  
    return layers.add([conv, x])

def strided_res_block(x, filters):
    conv = layers.Conv2D(filters, 3, padding='same', strides=(2,2), kernel_regularizer=regularization)(x)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation('relu')(conv)
    conv = layers.Conv2D(filters, 3, padding='same', kernel_regularizer=regularization)(conv)
    conv = layers.BatchNormalization()(conv)
  
    shortcut = layers.Conv2D(filters, 1, padding='same', strides=(2,2), kernel_regularizer=regularization)(x)
    shortcut = layers.BatchNormalization()(shortcut)
  
    s = layers.add([conv, shortcut])
    s = layers.Activation('relu')(s)
  
    return s

def getModel(input_shape):
    inp = layers.Input(input_shape)
  
    conv = layers.Conv2D(32, 3, padding='same', strides=(2,2), kernel_regularizer=regularization)(inp)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation('relu')(conv)
    
    conv = res_block(conv)
    conv = res_block(conv)
  
    conv = strided_res_block(conv, 64)
    conv = res_block(conv)
  
    out2 = layers.GlobalAveragePooling2D()(conv)
    out2 = layers.Dropout(0.2)(out2)
    out2 = layers.Dense(5, activation='softmax', name='aux_out')(out2)
  
    conv = strided_res_block(conv, 64)
    conv = res_block(conv)
  
    conv = strided_res_block(conv, 128)
    conv = res_block(conv)   
  
    pool = layers.GlobalAveragePooling2D()(conv)
    pool = layers.Dropout(0.1)(pool)
    fc = layers.Dense(50)(pool)
    fc = layers.Activation('relu')(fc)
    fc = layers.Dense(5)(fc)
    out = layers.Activation('softmax', name='out')(fc)
   
    return models.Model(inputs=inp, outputs=[out, out2])
 
model = getModel((im_size_1, im_size_2, 3))


model.compile(optimizer=optimizers.Adam(1e-3), 
              loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy'],
              loss_weights=[1, 0.5],
              metrics=['accuracy'])

model.fit(X_train.reshape(-1, im_size_1, im_size_2, 3), [y_train, y_train],
          validation_data=(X_val.reshape(-1, im_size_1, im_size_2, 3), [y_val, y_val]),
          batch_size=15,
          epochs=25,
          callbacks=[
              callbacks.ModelCheckpoint('weights.h5', verbose=1, 
                                        save_best_only=True, save_weights_only=True,
                                        monitor='val_out_loss'),
              callbacks.ReduceLROnPlateau(patience=2, verbose=1, monitor='val_out_loss'),
              callbacks.EarlyStopping(patience=5, verbose=1, monitor='val_out_loss')
          ]
         )

model.load_weights('weights.h5') 


# In[41]:


# predicting for submission images
predict = model.predict(test_labled_images_scaled_binary)[0]
np.argmax(predict, axis=-1)


# ResNet50

# In[ ]:


model = applications.ResNet50(weights=None, input_shape=(im_size_1, im_size_2, 3), classes=5)

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          batch_size=16,
          epochs=15,
          callbacks=[
              callbacks.ModelCheckpoint('weights.h5', verbose=1, save_best_only=True, save_weights_only=True),
              callbacks.ReduceLROnPlateau(patience=2, verbose=1),
              callbacks.EarlyStopping(patience=4, verbose=1)
          ])


# In[ ]:


model.evaluate(X_test, y_test)


# ResNet50 + frozen Dense layer + preload weights

# In[ ]:


X_train = X_train*2-1
X_val = X_val*2-1
X_test = X_test*2-1


# In[44]:


#pretrained model without top layer
base_model = applications.ResNet50(include_top=False, weights='imagenet', input_shape=(im_size_1, im_size_2, 3),                                   pooling='avg')
#fix weights
base_model.trainable = False

#and our model become simple
inp = layers.Input((im_size_1, im_size_2, 3))
resnet = base_model(inp)

fc = layers.Dense(5)(resnet)
fc = layers.Activation('softmax')(fc)

model = models.Model(inp, fc)

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#train only dense layer on top
model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          batch_size=16,
          epochs=1,
          callbacks=[
              callbacks.ModelCheckpoint('weights.h5', verbose=1, save_best_only=True, save_weights_only=True),
              callbacks.ReduceLROnPlateau(patience=2, verbose=1),
              callbacks.EarlyStopping(patience=4, verbose=1)
          ]
         )


#unfreeze all weights and train 
base_model.trainable = True
model.compile(optimizer=optimizers.Adam(1e-4), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          batch_size=16,
          epochs=30,
          callbacks=[
              callbacks.ModelCheckpoint('weights.h5', verbose=1, save_best_only=True, save_weights_only=True),
              callbacks.ReduceLROnPlateau(patience=2, verbose=1),
              callbacks.EarlyStopping(patience=4, verbose=1)
          ]
         )


# In[45]:


model.evaluate(X_test, y_test)


# ResNet50 + frozen Dense layer + preload weights + Augment images

# In[ ]:


class AugmentedSequence(utils.Sequence):
    def __init__(self, X, y, batch_size):
        self.X = np.array(X)
        self.y = np.array(y)
        self.batch_size = batch_size
    
        #for shuffling
        self.ids = np.random.permutation(range(len(X)))
    
        #for augmentation
        self.seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(scale=(0.8, 1.2), translate_percent=(-0.2, 0.2), rotate=(-15, 15))
            ])
    
    def __len__(self):
        return int(np.ceil(len(self.X)/float(self.batch_size)))
  
    def __getitem__(self, index):
        start = index * self.batch_size
        end = np.minimum((index + 1) * self.batch_size, len(self.X))
    
        ids = self.ids[start:end]
    
        batchX = self.X[ids]
        batchy = self.y[ids]
    
        batchX = self.seq.augment_images(batchX)
    
        return np.array(batchX), np.array(batchy)
  
    def on_epoch_end(self):    
        self.ids = np.random.permutation(range(len(self.X)))


# Model 1

# In[13]:


#pretrained model without top layer
base_model = applications.ResNet50(include_top=False, weights='imagenet',                                   input_shape=(im_size_1, im_size_2, 3), pooling='avg')
#fix weights
base_model.trainable = False

#and our model become simple
inp = layers.Input((im_size_1, im_size_2, 3))
resnet = base_model(inp)
fc = layers.Dense(100)(resnet)
fc = layers.Dense(5)(fc)
fc = layers.Activation('softmax')(fc)

model = models.Model(inp, fc)

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

generator = AugmentedSequence(X_train, y_train, 16)

#train only dense layer on top
model.fit_generator(generator,
          validation_data=(X_val, y_val),
          epochs=1,
          callbacks=[
              callbacks.ModelCheckpoint('weights.h5', verbose=1, save_best_only=True, save_weights_only=True),
              callbacks.ReduceLROnPlateau(patience=2, verbose=1),
              callbacks.EarlyStopping(patience=4, verbose=1)
          ])

#unfreeze all weights and train 
base_model.trainable = True
model.compile(optimizer=optimizers.Adam(1e-4), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(generator,
          validation_data=(X_val, y_val),
          epochs=30,
          callbacks=[
              callbacks.ModelCheckpoint('weights.h5', verbose=1, save_best_only=True, save_weights_only=True),
              callbacks.ReduceLROnPlateau(patience=2, verbose=1),
              callbacks.EarlyStopping(patience=4, verbose=1)
          ])


# In[14]:


model.evaluate(X_test, y_test)


# Model 2 (modified frozen layers)

# In[20]:


#pretrained model without top layer
base_model = applications.ResNet50(include_top=False, weights='imagenet',                                   input_shape=(im_size_1, im_size_2, 3), pooling='avg')
#fix weights
base_model.trainable = False

#and our model become simple
inp = layers.Input((im_size_1, im_size_2, 3))
resnet = base_model(inp)

fc = layers.Dense(128)(resnet)
fc = layers.Activation('relu')(fc)
fc = layers.Dropout(0.1)(fc)

fc = layers.Dense(64)(fc)
fc = layers.Activation('relu')(fc)
fc = layers.Dropout(0.1)(fc)

fc = layers.Dense(5)(fc)
fc = layers.Activation('softmax')(fc)

model = models.Model(inp, fc)

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

generator = AugmentedSequence(X_train, y_train, 16)

#train only dense layer on top
model.fit_generator(generator,
          validation_data=(X_val, y_val),
          epochs=10,
          callbacks=[
              callbacks.ModelCheckpoint('weights.h5', verbose=1, save_best_only=True, save_weights_only=True),
              callbacks.ReduceLROnPlateau(patience=2, verbose=1),
              callbacks.EarlyStopping(patience=4, verbose=1)
          ])

#unfreeze all weights and train 
base_model.trainable = True
model.compile(optimizer=optimizers.Adam(1e-4), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(generator,
          validation_data=(X_val, y_val),
          epochs=30,
          callbacks=[
              callbacks.ModelCheckpoint('weights.h5', verbose=1, save_best_only=True, save_weights_only=True),
              callbacks.ReduceLROnPlateau(patience=2, verbose=1),
              callbacks.EarlyStopping(patience=4, verbose=1)
          ])


# In[21]:


model.evaluate(X_test, y_test)


# Predictions and submission

# In[35]:


# Predioction
predict = model.predict(test_labled_images_scaled_binary)
labels_predict = np.argmax(predict, axis=-1)
# from sklearn.metrics import accuracy_score
# print(accuracy_score(labels_predict, y_test))


# ID and Categorial labels
labels_id = range(3026,3026+len(labels_predict))
class_dict_submission = {v:k for k, v in class_dict.items()}
labels_predict_kategorial = pd.Series(labels_predict).map(class_dict_submission).values

# Save to csv
sub = pd.DataFrame({'Id':labels_id, 'Category':labels_predict_kategorial})
sub = sub[['Id','Category']]
sub.to_csv('submission.csv', index=False)

#!kaggle competitions submit -c geekhub-ds-2019-challenge -f submission.csv -m "11 try ResNet50 + 200 sizes + frozen dense layer(mod)+Augmented"


# In[ ]:




