from fastai import *
from fastai.vision import *
import numpy as np
import torch
import os
import glob
import shutil
from sklearn.metrics import *


#in order to use the inference we need to export the model using model.export()

#then we need to just load the learner.Everything is loaded automatically without having to load any arguments except 
#the location of the folder where the model was saved
print("Give me 5 minutes");

path_folder= Path("/cluster/home/cantoniou/deep_project/sipakmedFormat")
model_path = path_folder/'models'/'TTL'
learn = load_learner(model_path) #it requires the path for the export.pkl file
test_dataset_path = path_folder/'test'             

all_test_files_labels=[]
predictions=[]
labels = []
index = 0 
mistakes = 0

for file in os.listdir(test_dataset_path):            #for any file inside the root directory 
	classes_path = os.path.join(test_dataset_path, file)  #fSo for every folder class we create a class directory
	class_files = [name for name in glob.glob(os.path.join(classes_path,'*.bmp'))]
	for i in range(len(class_files)):
		labels.append(index)
		img = open_image(class_files[i])
		category, preds,la = learn.predict(img)
		if(preds.item()!=index):
			mistakes=mistakes+1
	all_test_files_labels = all_test_files_labels+class_files
	index=index+1

#print(mistakes)	
#print(labels)

for i in range(len(all_test_files_labels)):
	img = open_image(all_test_files_labels[i])
	category, preds,la = learn.predict(img)
	predictions.append(preds.item())

	


#You can now get the predictions on any image via learn.predict.
#imgPath = "dyskeratotic.bmp"
#img = open_image(imgPath) #just for one image
#print(learn.predict(img)) #get the predictions only for one image
#You can also do inference on a larger set of data by adding a test set. This is done by passing an ItemList to load_learner.

#learn = load_learner(model_path, test=ImageList.from_folder(path_folder/'test'))
#preds,labels = learn.get_preds(ds_type=DatasetType.Test)
#predictions = preds.argmax(dim=-1) #small trick to get the actual ground truth
#print(labels)
#label = labels.argmax(dim=-1)
#class_predictions = [data.classes[int(x)] for x in predictions]
torch.set_printoptions(threshold=10000)
#print(preds) #print the whole prediction tensor
#print("/n")
#print(label)
#print(type(label))
#print(type(preds))
#print(label[3].item()) #testing a random element of the tensor 
#print(preds.shape)#get the shape of the tensor
#shape_preds = preds.shape ; #get the size-shape of the prediction tensor

#counter_rows =  shape_preds[0] #get the number of rows of the tensor(which is the number of the test size)
#mistakes = 0 #the number of mistakes

#final_preds = np.zeros(counter_rows)

#This is a function that will decide which category was predicted
#def category(tensor_preds, counter):  #this function will decided which category was predicted for each image of the test_set
#	maximum_pred = tensor_preds[counter,0] #assume that the highest prediction occurs in 0 position
#	maximum_pos = 0 ; #for 0 category 
#	for i in range(0,5):
#		if(tensor_preds[counter,i]>=maximum_pred):
#			maximum_pred = tensor_preds[counter,i]
#			maximum_pos = i;
#		
#	return maximum_pos ;


#Print Prediction and True Label
#print("Prediction        Label")           
#for i in range(0, counter_rows):
	#if(label[i].item()!= category(preds,i)):
		
		#mistakes=mistakes+1
		#print("@!@!@",end=" ")
	#Print for every sample the predicted and ground truth
	#print( category(preds,i),end ="    ")
	#print("------->", end=" ")
	#print(label[i].item())

 
#print the classification report
#print(final_preds.shape)
#print(labels.shape)


total_size_samples = len(labels)# counter_rows #the sample size of the test dataset is placed in the first position of the size tensor
accuracy = (1 - (mistakes/total_size_samples))*100.0 
print("The accuracy of the testing dataset is " + str(accuracy)+ "%")

print(confusion_matrix(labels,predictions))
print(classification_report(labels,predictions))












