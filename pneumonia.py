
# Convolutional Neural Network
​
#-----------------------------------------------------------
# Part 1 - Building the CNN
#-----------------------------------------------------------
​
#-----------------------------------------------------------
# Importing the Keras libraries and packages
#-----------------------------------------------------------
import keras
from keras.layers import Dense, Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.models import Sequential
from keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import numpy as np
from keras.preprocessing import image

#-----------------------------------------------------------
# Initialising the CNN
#-----------------------------------------------------------
classifier = Sequential()
​
#-----------------------------------------------------------
# Step 1 - Convolution
#-----------------------------------------------------------
classifier.add(Conv2D(64, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
​
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(216, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
​
#-----------------------------------------------------------
# Step 3 - Flattening
#-----------------------------------------------------------
classifier.add(Flatten())
​
#-----------------------------------------------------------
# Step 4 - Full connection
#-----------------------------------------------------------
classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
print(model.summary())
​
#-----------------------------------------------------------
# Compiling the CNN
#-----------------------------------------------------------
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
​
#-----------------------------------------------------------
# Part 2 - Fitting the CNN to the images
#-----------------------------------------------------------
​
from keras.preprocessing.image import ImageDataGenerator
​
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.3,
                                   zoom_range = 0.3,
                                   horizontal_flip = True,
                                   vertical_flip=True)
​
​
val_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
​
​
training_set = train_datagen.flow_from_directory('dataset/train',
                                                 target_size = (64, 64),
                                                 batch_size = 64,
                                                 class_mode = 'binary')
​
val_set = val_datagen.flow_from_directory('dataset/val',
                                            target_size = (64, 64),
                                            batch_size = 64,
                                            class_mode = 'binary')
​
history=classifier.fit_generator(training_set,
                         steps_per_epoch = 3516/64,
                         epochs = 45,
                         validation_data = val_set,
                         validation_steps = 1170/64)
​


test_set = test_datagen.flow_from_directory('dataset/test',
                                            target_size = (64, 64),
                                            batch_size = 64,
                                            class_mode = 'binary',shuffle=False)
​
pred=classifier.predict_generator(test_set,steps=1170/64)
​
predictedClasses = np.where(pred>0.5, 1, 0)
true_classes = test_set.classes
class_labels = list(test_set.class_indices.keys()) 
cm = metrics.confusion_matrix(true_classes, predictedClasses)
report = metrics.classification_report(true_classes, predictedClasses, target_names=class_labels)
print(report)

#-----------------------------------------------------------
# Part 3 : Check for overfitting or underfitting 
#-----------------------------------------------------------

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

# Retrieve a list of list results on training and test data
# sets for each training epoch

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc, 'r', label="Training Accuracy")
plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
plt.title('Training and validation accuracy')
plt.ylabel('Accuracy Value')
plt.xlabel('Epoch')
plt.legend(loc='best')
plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss, 'r', label="Training Loss")
plt.plot(epochs, val_loss, 'b', label="Validation Loss")
plt.title('Training and validation loss')
plt.ylabel('Loss Value')
plt.xlabel('Epoch')
plt.legend(loc='best')
plt.figure()

#-----------------------------------------------------------
# Part 4 : Checking results for a new image : 
#-----------------------------------------------------------

img = image.load_img('test_image_normal.jpeg')
imgplot = plt.imshow(img)
print('The image that you have entered for testing :\n')
plt.show()
print('\n')
print('ACTUAL OBSERVATION : normal \n')
print('----- Testing on the trained model ----- \n')
print("MODEL's OBSERVATION :")
test_image = image.load_img('test_image_pneu.jpeg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict_on_batch(test_image)
print(result)
if result[0][0] == 0:
 print("NORMAL")
else:
 print("PNEUMONIA PRESENT")

#-----------------------------------------------------------
#END
#-----------------------------------------------------------
