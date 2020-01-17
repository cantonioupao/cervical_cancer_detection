import numpy as np
from keras.layers import Dense, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from K_REPORT_MODELS import R50, VGG


"""LOAD DATA"""
# path to folder with data
path = "/cluster/home/luciensc/sipakmed_formatted/"
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

training_set_V2 = no_DA_IDG.flow_from_directory(path+"train/", ### TO USE FOR FEATURE EXTRACTION
                                                target_size=(224, 224), # typical imagenet dimensions
                                                color_mode='rgb',
                                                batch_size=1,
                                                class_mode='categorical', shuffle=False)

validation_set = no_DA_IDG.flow_from_directory(path+"val/",
                                                target_size=(224, 224),
                                                color_mode='rgb',
                                                class_mode='categorical',
                                                shuffle=True)

validation_set_V2 = no_DA_IDG.flow_from_directory(path+"val/",
                                                target_size=(224, 224),
                                                color_mode='rgb',
                                                batch_size=1,
                                                class_mode='categorical',
                                                shuffle=False)

test_set_V2 = no_DA_IDG.flow_from_directory(path+"test/",
                                                target_size=(224, 224),
                                                color_mode='rgb',
                                                batch_size=1,
                                                class_mode='categorical',
                                                shuffle=False)

###############################################################################################
"""CREATE MODELS"""
model_r50 = R50()
model_vgg = VGG()

"""LOADING WEIGHTS FROM PREVIOUSLY TRAINED MODELS (see files K_REPORT_MAIN_R50 resp. K_REPORT_MAIN_VGG)"""
model_r50.load_weights("K_R50_T2.h5")
model_vgg.load_weights("K_VGG_T2.h5")




"""MODEL COMBINATION"""
# creating the model extracting features from the last layer before the softmax layer.
vgg_extractor = Model(inputs=model_vgg.input, outputs=model_vgg.get_layer("dense_1024").output)
r50_extractor = Model(inputs=model_r50.input, outputs=model_r50.get_layer("dense_1024").output)


# using the model extractor to generate feature arrays with 1024 features from each model extractor.
y_train = to_categorical(training_set_V2.classes)
X_train_m1 = vgg_extractor.predict_generator(training_set_V2, steps = training_set_V2.n)
X_train_m2 = r50_extractor.predict_generator(training_set_V2, steps = training_set_V2.n)
X_train = np.concatenate([X_train_m1, X_train_m2], axis=1)

y_val = to_categorical(validation_set_V2.classes)
X_val_m1 = vgg_extractor.predict_generator(validation_set_V2, steps = validation_set_V2.n)
X_val_m2 = r50_extractor.predict_generator(validation_set_V2, steps = validation_set_V2.n)
X_val = np.concatenate([X_val_m1, X_val_m2], axis=1)

y_test = to_categorical(test_set_V2.classes)
X_test_m1 = vgg_extractor.predict_generator(test_set_V2, steps = test_set_V2.n)
X_test_m2 = r50_extractor.predict_generator(test_set_V2, steps = test_set_V2.n)
X_test = np.concatenate([X_test_m1, X_test_m2], axis=1)


"""TRAIN & TEST FEATURE EXTRACTION MODEL"""
# the feature arrays are read into a sequential model directly connecting them to the softmax layer, with
# some dropout and batch normalization in between.

# set a seed found to produce a run with consistently high validation accuracy (~.93 +)
# the specified seed will probably have to be adapted with different train-validation split
# and loaded model weights of R50 and VGG
np.random.seed(668)

opt = Adam(learning_rate=1e-3)
model = Sequential()
model.add(Dropout(0.75, input_shape=(2048,)))
model.add(BatchNormalization())
model.add(Dense(5, activation="softmax"))
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])


epochs = 200
model.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=2, shuffle=True, validation_data=(X_val, y_val))

"""EVALUATE MODEL ON TEST DATA"""
y_test = np.argmax(y_test, axis=1)
pred = np.argmax(model.predict(X_test), axis=1)
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

