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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
import tensorflow as tf

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("No GPU available. Please install GPU version of TensorFlow.")

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
#*************************************************************************************
# functions
#*************************************************************************************

def plot_confusion_matrix(y_labels, y_preds, save_location, title):
   print("Plotting " + title + " confusion matrix...")
   confusion_data = confusion_matrix(y_labels, y_preds)
   plt.figure(figsize=(8, 6))
   sns.heatmap(confusion_data, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 16})
   plt.xlabel("Predicted")
   plt.ylabel("True")
   plt.title(title)
   plt.savefig(save_location)
   #plt.show()
   plt.close()
   print("done...")

#*************************************************************************************

def plot_loss_acc(history, save_location):
   print("Plotting loss and accuracy...")
   acc = history.history['acc']
   val_acc = history.history['val_acc']
   loss = history.history['loss']
   val_loss = history.history['val_loss']
   epochs = range(1, len(acc) + 1)
   # Create subplots
   plt.figure(figsize=(12, 6))

   # Subplot for accuracy
   plt.subplot(1, 2, 1)
   plt.plot(epochs, acc, 'bo', label='Training acc')
   plt.plot(epochs, val_acc, 'b', label='Validation acc')
   plt.title('Training and validation accuracy')
   plt.legend()

   # Subplot for loss
   plt.subplot(1, 2, 2)
   plt.plot(epochs, loss, 'bo', label='Training loss')
   plt.plot(epochs, val_loss, 'b', label='Validation loss')
   plt.title('Training and validation loss')
   plt.legend()

   plt.savefig(save_location)
   plt.close()
   print("done...")

#*************************************************************************************

def write_metrics(y_test, y_test_pred, y_train, y_train_pred, txt_location):
   # Calculate and print the accuracy
   test_accuracy = accuracy_score(y_test, y_test_pred)
   print("test Accuracy: %.2f%%" % (test_accuracy * 100.0))
   train_accuracy = accuracy_score(y_train, y_train_pred)
   print("train Accuracy: %.2f%%" % (train_accuracy * 100.0))
   print("\n\n\n")
   classification_report_str = classification_report(y_test, y_test_pred, target_names=['Negative', 'Positive'])
   print(classification_report_str)
   with open(txt_location, 'w') as file:
      file.write("test Accuracy: %.2f%%" % (test_accuracy * 100.0))
      file.write("train Accuracy: %.2f%%" % (train_accuracy * 100.0))
      file.write("\n\n\n")
      file.write(classification_report_str)

#*************************************************************************************
print("Importing done \n Prepocessing now...")
# preprocessing
#*************************************************************************************

train_confusion_matrix_save_location = 'Results/train_confusion_matrixa.png'
test_confusion_matrix_save_location = 'Results/test_confusion_matrixa.png'
txt_location = 'Results/metricsa.txt'
loss_acc_plot_location = 'Results/loss_acc_plota.png'

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

train_index = 10000
val_index = 15000
test_index = 20000

i = 0
for filename in os.listdir(original_neg_dir):
   if i < train_index: # copy neg images 0-9999 to neg dir (10,000 images)
      shutil.copyfile(original_neg_dir + filename, train_neg_dir + '/' + filename)
   elif i < val_index: # copy neg images 10000-14999 to neg dir (5,000 images)
      shutil.copyfile(original_neg_dir + filename, val_neg_dir + '/' + filename)
   elif i < test_index: # copy neeg images 15000-19999 to neg dir (5,000 images)
      shutil.copyfile(original_neg_dir + filename, test_neg_dir + '/' + filename)
   else:
      break
   i += 1

j = 0
for filename in os.listdir(original_pos_dir):
   if j < train_index:
      shutil.copyfile(original_pos_dir + filename, train_pos_dir + '/' + filename)
   elif j < val_index:
      shutil.copyfile(original_pos_dir + filename, val_pos_dir + '/' + filename)
   elif j < test_index:
      shutil.copyfile(original_pos_dir + filename, test_pos_dir + '/' + filename)
   else:
      break
   j += 1

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

# Define the data generators for training, validation, and testing
train_datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(227, 227), batch_size=32, class_mode='binary')

validation_datagen = ImageDataGenerator(rescale=1.0/255)
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(227, 227), batch_size=32, class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(227, 227), batch_size=32, class_mode='binary')



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

'''
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1e-4), metrics=['acc'])
history = model.fit_generator(train_generator,
                              steps_per_epoch=50, #number of gradient step before next epoch
                              epochs=30,
                              validation_data=validation_generator,
                              validation_steps=50)
'''
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1e-4), metrics=['acc'])

#*************************************************************************************
print("Model compliled \n fitting model now...")
# Compile model
#*************************************************************************************

history = model.fit(train_generator,
                    steps_per_epoch=100, #number of gradient step before next epoch
                    epochs=5,
                    validation_data=validation_generator,
                    validation_steps=50)
                    # how many batches to draw from the validation generator for evaluation
model.save('pos_neg_cracks.h5')

#*************************************************************************************
print("Model fitted \n making predictions with model now...")
# Evaluate model
#*************************************************************************************
train_generator.reset()
test_generator.reset()
validation_generator.reset()

y_train = train_generator.classes
y_test = test_generator.classes

train_predictions = model.predict(train_generator, steps=len(train_generator), verbose=1)
y_train_pred = [1 * (x[0]>=0.5) for x in train_predictions]

test_predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)
y_test_pred = [1 * (x[0]>=0.5) for x in test_predictions]
#*************************************************************************************
print("Model predictions done \n gathering metrics from model now...")
# Evaluate model
#*************************************************************************************

plot_confusion_matrix(y_train, y_train_pred, train_confusion_matrix_save_location, 'Train Confusion Matrix')
plot_confusion_matrix(y_test, y_test_pred, test_confusion_matrix_save_location, 'Test Confusion Matrix')

plot_loss_acc(history, loss_acc_plot_location)

write_metrics(y_test, y_test_pred, y_train, y_train_pred, txt_location)