from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras-layers import Dense, Conv2D, Flatten ,Dropout, MaxPoooling
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplolib.pyplot as plt



URL = https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8451588&tag=1     
URL_description = http://www.cs.uoi.gr/~marina/SIPAKMED/Description_of_Features.pdf #link of the dataset
pathZIP= tf.keras.utils.get_file("cervicalcancer.zip", origin= URL, extract = True)
dataset_directory = os.path.join(os.path.dirname(pathZIP))  #create the path of the main dataset directory
train_dir= os.path.join(dataset_directory,  "train") #create another directory but dont create the folder
val_dir = os.path.join(dataset_directory,  "validation")


#Training directories for various classification classes
cancerous_cells_tdir =os.path.join(train_dir, "cancerous")  
semicancerous_cells_tdir =os.path.join(train_dir,"semi")

#Validation directories for various classification classes
cancerous_cells_vdir =os.path.join(train_dir, "cancerous")
semicancerous_cells_vdir =os.path.join(train_dir,"semi")

#load all the training classes with their directories in classes arrays

classes_t_dir = {cancelous_cells_tdir, semicancerous_cells_tdir}
classes_v_dir =  {cancelous_cells_vdir, semicancerous_cells_vdir}

#Check the length of each dataset division
#set the lengths of training and validation to 0
training_pics_number =0
validation_pics_number =0
for i in length(classes_v_dir):
	training_pics_number = training_pics_number + len(os.listdir(classes_t_dir[i]))
	validation_pics_number = validation_pics_number + len(os.listdir(classes_v_dir[i]))


print("The total number of images is:", training_pics_numbers)
print("The total number of valdiation images is", validation_pics_numbers)




#training parameters
batch_size = 128
epochs =50 
i_h =150
i_w = 150


#create the image generators
train_image_gen =ImageDataGenerator(rescale= 1.0/255) #generator that rescales the images values form 0-255 to just values in the range 0.0-1.0 as it is more preferable for input values for NN
val_image_gen =ImageDataGenerator(rescale= 1.0/255) #generator that rescales the images values form 0-255 to just values in the range 0.0-1.0 as it is more preferable for input values for NN

#implement the generators to our dataset
train_gen = train_image_generator.flow_from_directory(batch_size=batch_size,directory=train_dir, target_size= (i_h, i_w), class_mode = categorical) #it can either be binary ,categorical and sparse
val_gen = val_image_gen.flow_from_directory(batch_size=batch_size, directory=val_dir,target_size= (i_h, i_w), class_mode = categorical) ;



#return a batch of the dataset
samples_img , labels =next(train_gen) ;


# This function will plot images in the form of a grid with 5 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(5, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        #ax.axis('off')
    plt.tight_layout()
    plt.show()


#print now 25 images 
plotImages(samples_img[:25])


#create the NN model
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(i_h, i_w ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
]);  #using a "same" padding it means we are not really using padding

#complile the model
model.compile(optimizer='adam',loss="categorical_crossentropy",metrics=['accuracy']);

#model summary
model.summary()


#source
#https://www.tensorflow.org/tutorials/images/classification

