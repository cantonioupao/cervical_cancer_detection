### MODELS
import numpy as np
from keras.layers import Dense, GlobalMaxPool2D, BatchNormalization, Dropout
from keras.applications import ResNet50, VGG16
from keras.models import Model, Sequential


"""
Create a new feature extractor based on the pretrained ResNet50 model:
The model is frozen from the bottom layer to activation layer 24. 
Activation layer 25 is then pooled with GlobalMaxPool2D, connected to a 
Dense layer with 1024 neurons, and then to the softmax layer.
"""
def R50(seed = None):
    np.random.seed(seed)
    base_model = ResNet50(weights="imagenet", include_top=False)
    for layer in base_model.layers[:86]:
        layer.trainable = False
    for layer in base_model.layers[86:]:
        layer.trainable = True
    x = (base_model.get_layer("activation_25")).output
    mx = GlobalMaxPool2D()(x)
    x = BatchNormalization()(mx)
    x = Dropout(.5)(x)
    x = Dense(1024, activation='relu', name="dense_1024")(x)
    x = BatchNormalization()(x)
    x = Dropout(.5)(x)
    preds = Dense(5,activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=preds)
    return model

"""
Create a new feature extractor based on the pretrained VGG16 model:
The model is frozen from the bottom layer up to block4 conv layer 2. 
Conv layer 3 from block 4 is then pooled with GlobalMaxPool2D, and
connected to a Dense layer with 1024 neurons, and then to the softmax layer.
"""
def VGG(seed = None):
    np.random.seed(seed)
    vgg16 = VGG16(weights="imagenet", include_top=False)
    for layer in vgg16.layers[:13]:
        layer.trainable = False
    for layer in vgg16.layers[13:]:
        layer.trainable = True
    y = (vgg16.get_layer("block4_conv3")).output
    mx_y = GlobalMaxPool2D()(y)
    x = BatchNormalization()(mx_y)
    x = Dropout(.5)(x)
    x = Dense(1024, activation='relu', name="dense_1024")(x)
    x = BatchNormalization()(x)
    x = Dropout(.5)(x)
    preds = Dense(5,activation='softmax')(x)
    model = Model(inputs=vgg16.input,outputs=preds)
    return model