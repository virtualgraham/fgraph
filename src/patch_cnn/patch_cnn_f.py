# this is a test to simply recreate vgg16 to see if everything works 

import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ZeroPadding2D, Lambda, Dropout, Dense, Input, Concatenate, ReLU, Cropping2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.applications.vgg16 import preprocess_input

import cv2 as cv
import numpy as np
from tensorflow.keras.utils import to_categorical
import os



# L2/HardNet Architecture (with color)?

la01 = Conv2D(64, (3, 3), activation='relu', padding='same')
la02 = Conv2D(64, (3, 3), activation='relu', padding='same')
la03 = MaxPooling2D((2, 2), strides=(2, 2))

la04 = Conv2D(128, (3, 3), activation='relu', padding='same')
la05 = Conv2D(128, (3, 3), activation='relu', padding='same')
la06 = MaxPooling2D((2, 2), strides=(2, 2))

la07 = Conv2D(256, (3, 3), activation='relu', padding='same')
la08 = Conv2D(256, (3, 3), activation='relu', padding='same')
la09 = Conv2D(256, (3, 3), activation='relu', padding='same')
la10 = MaxPooling2D((2, 2), strides=(2, 2))

la11 = Conv2D(512, (3, 3), activation='relu', padding='same')
la12 = Conv2D(512, (3, 3), activation='relu', padding='same')
la13 = Conv2D(512, (3, 3), activation='relu', padding='same')
la14 = MaxPooling2D((2, 2), strides=(2, 2))

la15 = Conv2D(512, (3, 3), activation='relu', padding='same')
la16 = Conv2D(512, (3, 3), activation='relu', padding='same')
la17 = Conv2D(512, (3, 3), activation='relu', padding='same')
la18 = MaxPooling2D((2, 2), strides=(2, 2))

la19 = Flatten(name='flatten')

# Inputs are a 7 x 7 grid of 32px x 32px patches 
# That is a 224px 224px image, same as many common CNNs trained on ImageNet

input = Input(shape=(224,224,3))



op00 = la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(input)))))))))))))))))))


# Fully connected layers from VGG16

x = Dense(4096, activation='relu', name='fc1')(op00)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(1000, activation='softmax', name='predictions')(x)

model = Model(input, x)

model.compile(loss='categorical_crossentropy', optimizer = "rmsprop", metrics=['accuracy'])

print(model.summary())


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



import tensorflow_datasets as tfds

# Fetch the dataset directly
imagenet = tfds.image.Imagenet2012()
## or by string name
#imagenet = tfds.builder('imagenet2012')

# Describe the dataset with DatasetInfo
C = imagenet.info.features['label'].num_classes
Ntrain = imagenet.info.splits['train'].num_examples
Nvalidation = imagenet.info.splits['validation'].num_examples
Nbatch = 32
assert C == 1000
assert Ntrain == 1281167
assert Nvalidation == 50000

# Download the data, prepare it, and write it to disk
imagenet.download_and_prepare()

# Load data from disk as tf.data.Datasets
datasets = imagenet.as_dataset()
train_dataset, validation_dataset = datasets['train'], datasets['validation']

# print("type(train_dataset).__name__",type(train_dataset).__name__)
# print("type(validation_dataset).__name__", type(validation_dataset).__name__)
# assert isinstance(train_dataset, tf.data.PrefetchDataset)
# assert isinstance(validation_dataset, tf.data.PrefetchDataset)

def imagenet_generator(dataset, batch_size=32, num_classes=1000, is_training=False):
  images = np.zeros((batch_size, 224, 224, 3))
  labels = np.zeros((batch_size, num_classes))
  while True:
    count = 0 
    for sample in tfds.as_numpy(dataset):
      image = sample["image"]
      label = sample["label"]
    
      images[count%batch_size] = preprocess_input(np.expand_dims(cv.resize(image, (224, 224)), 0))
      labels[count%batch_size] = np.expand_dims(to_categorical(label, num_classes=num_classes), 0)
      
      count += 1
      if (count%batch_size == 0):
        yield images, labels

# Infer on ImageNet
# labels = np.zeros((Nvalidation))
# pred_labels = np.zeros((Nvalidation, C))
# pred_labels_new = np.zeros((Nvalidation, C))
  
# score = model.evaluate_generator(imagenet_generator(validation_dataset,batch_size=32), 
#                                   steps= Nvalidation // Nbatch, 
#                                   verbose=1)
# print("Evaluation Result of Original Model on ImageNet2012: " + str(score))
  
# Train on ImageNet
checkpoint_path = "patchnet/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
os.makedirs(checkpoint_dir, exist_ok=True)

cp_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 1-epoch
    period=1)
    
csv_logger = keras.callbacks.CSVLogger('patchnet_training.csv')
  
print("Starting to train patchnet...")
epochs = 20

model.fit_generator(imagenet_generator(train_dataset, batch_size=Nbatch, is_training=True),
                        steps_per_epoch= Ntrain // Nbatch,
                        epochs = epochs,
                        validation_data = imagenet_generator(validation_dataset, batch_size=Nbatch),
                        validation_steps = Nvalidation // Nbatch,
                        verbose = 1,
                        callbacks = [cp_callback, csv_logger])
  
model.save("patchnet.h5")