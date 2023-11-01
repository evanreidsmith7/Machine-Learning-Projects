#************************************************************************************
# Evan Smith
# ML – HW#5
# Filename: hw5-dcnn.py
# Due: , 2023
#
# Objective:
#Develop your own CNN model to classify all classes.
#• Provide the training and test confusion matrices.
#• Provide the test accuracy, precision, recall, and F1-scores to a text file.
#• Provide the Loss and Accuracy curves for training and validation (you can use a single
#plot for these four curves)
#• Expected results: High 90’s for training, validation, and testing without
#overfitting/underfitting
#*************************************************************************************
print("Importing libraries...")
# Import libraries
#*************************************************************************************

import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

print("Importing done \n Prepocessing now...")


original_neg_dir = 'Dataset/Concrete Crack Images for Classi/Negative/'
original_pos_dir = 'Dataset/Concrete Crack Images for Classi/Positive/'



train_dir = 'Dataset/MyDataset/Train'
validation_dir = 'Dataset/MyDataset/Val'
test_dir = 'Dataset/MyDataset/Test'



test_neg_dir = 'Dataset/MyDataset/Test/Neg'
test_pos_dir = 'Dataset/MyDataset/Test/Pos'

val_neg_dir = 'Dataset/MyDataset/Val/Neg'
val_pos_dir = 'Dataset/MyDataset/Val/Pos'

train_neg_dir = 'Dataset/MyDataset/Train/Neg'
train_pos_dir = 'Dataset/MyDataset/Train/Pos'




i = 0
for filename in os.listdir(original_neg_dir):
   if i < 1000: # copy neg images 0-999 to neg dir (1000 images)
      shutil.copyfile(original_neg_dir + filename, train_neg_dir + '/' + filename)
   elif i < 1500: # copy neg images 1000-1499 to neg dir (500 images)
      shutil.copyfile(original_neg_dir + filename, val_neg_dir + '/' + filename)
   elif i < 2000: # copy neeg images 1500-1999 to neg dir (500 images)
      shutil.copyfile(original_neg_dir + filename, test_neg_dir + '/' + filename)
   else:
      break
   i += 1

   i = 0
for filename in os.listdir(original_pos_dir):
   if i < 1000:
      shutil.copyfile(original_pos_dir + filename, train_pos_dir + '/' + filename)
   elif i < 1500:
      shutil.copyfile(original_pos_dir + filename, val_pos_dir + '/' + filename)
   elif i < 2000:
      shutil.copyfile(original_pos_dir + filename, test_pos_dir + '/' + filename)
   else:
      break
   i += 1




print('total training pos images:', len(os.listdir(train_pos_dir)))
print('total training neg images:', len(os.listdir(train_neg_dir)))
print('total validation pos images:', len(os.listdir(val_pos_dir)))
print('total validation neg images:', len(os.listdir(val_neg_dir)))
print('total test pos images:', len(os.listdir(test_pos_dir)))
print('total test neg images:', len(os.listdir(test_neg_dir)))

#*************************************************************************************
print("Preprocessing done \n Creating model now...")
# Create model
#*************************************************************************************

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(227, 227),
                                                    batch_size=32,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(227, 227),
                                                        batch_size=32,
                                                        class_mode='binary')


model = models.Sequential()
model.add(layers.Conv2D(32,(3, 3),activation='relu',input_shape=(227,227,3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#*************************************************************************************
print("Model created \n Compiling model now...")
# Compile model
#*************************************************************************************

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
history = model.fit_generator(train_generator,
                              steps_per_epoch=50, #number of gradient step before next epoch
                              epochs=30,
                              validation_data=validation_generator,
                              validation_steps=50)
# how many batches to draw from the validation generator for evaluation
model.save('pos_neg_cracks.h5')

#*************************************************************************************
print("Model compiled \n Evaluating model now...")
# Evaluate model
#*************************************************************************************