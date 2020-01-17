import numpy as np
from keras.layers import Dense, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from K_REPORT_MODELS import R50, VGG



"""LOAD DATA"""
# path to folder with data
path = "C:/Users/luciensc/Desktop/sipakmed_formatted/" #"/cluster/home/luciensc/sipakmed_formatted/"
# expected data structure inside folder: train, test, val. in each folder: one folder for each class,
# comprising its respective slide images.


# specify image data generator with data augmentation (train_datagen) resp. without (no_DA_IDG)
train_datagen = ImageDataGenerator(featurewise_center=False,
                                   rotation_range = 5, fill_mode="nearest",
                                   zoom_range=[1/1.0, 1/1.0], width_shift_range=0.0, height_shift_range=0.0, # occasionally out of range
                                   horizontal_flip = True, vertical_flip=True,
                                   brightness_range=[0.5, 1.3], channel_shift_range=20)

no_DA_IDG = ImageDataGenerator()

# in training set: use data augmentation image data generator, for validation and test: no data augmentation.
training_set = train_datagen.flow_from_directory(path+"train/",
                                                target_size=(224, 224), # typical imagenet dimensions
                                                color_mode='rgb',
                                                batch_size=32,
                                                class_mode='categorical', shuffle=True)



validation_set = no_DA_IDG.flow_from_directory(path+"val/",
                                                target_size=(224, 224),
                                                color_mode='rgb',
                                                class_mode='categorical',
                                                shuffle=True)



test_set_V2 = no_DA_IDG.flow_from_directory(path+"test/",
                                                target_size=(224, 224),
                                                color_mode='rgb',
                                                batch_size=1,
                                                class_mode='categorical',
                                                shuffle=False)

###############################################################################################
"""CREATE & TRAIN MODELS"""

model_r50 = R50()

""" The model was trained with a Adam with a learning rate of 1e-3 for 50 epochs.
Then the training was continued for 50 epochs at a lower learning
rate.
The weights loaded at the end of the block result from a good run (based on validation score) following the described
training regime.
"""
epochs = 50
opt = Adam(learning_rate=1e-3)
model_r50.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
model_r50.fit_generator(generator=training_set, epochs=epochs, validation_data=validation_set, verbose=2)

opt = Adam(learning_rate=1e-5)
model_r50.fit_generator(generator=training_set, epochs=epochs, validation_data=validation_set, verbose=2)
# save model weights
model_r50.save_weights("K_R50_T2.h5")

"""EVALUATE PERFORMANCE ON THE TEST SET"""

y_test = test_set_V2.classes
pred = np.argmax(model_r50.predict_generator(test_set_V2, steps = test_set_V2.n), axis=1)
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))










