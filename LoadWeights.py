from fastai import *
from fastai.vision import *
from fastai.callbacks import *
#import Path
import torch
import torchvision
import torchvision
from torchvision import models as models


class LoadWeights:

	def __init__(self, location=""):
		self.location = location

	def load_w(self,location):
		self.location = location;
		res = models.resnet50
		#hyperparameters
		batch_size = 10
		epochs = 50

		save_loc = 'resnet50model_trainedonHerlevsetandSipakmed' + str(epochs) + "batch" + str(batch_size) #location to save the model

		## Declaring path of dataset
		path_img = Path("/cluster/home/cantoniou/deep_project/sipakmedFormat")
		## Loading data 
		data = ImageDataBunch.from_folder(path=path_img, train='train',
            valid='val', ds_tfms=get_transforms(), size = 224, bs=batch_size)#, check_ext=False)
		## Normalizing data based on Image net parameters
		#data.normalize(imagenet_stats)
		#normalize now according to the batch data and not imagenet
		data.normalize() #defaults to batch 'stats'
		print(data.classes)
		len(data.classes),data.c


		#LOAD THE TRANSFER LEARNING MODEL
		## To create a ResNET 50 with pretrained weights based on the new dataset
		trans_model= cnn_learner(data, models.resnet50, metrics=[accuracy,FBeta(average="weighted")])
		#print(trans_model) #check the architecture of the loaded model to make sure it was loaded with a head of 5 ---to match the data classes
		trans_model = trans_model.load(location) #load the previous pretrained model weights form the harlev dataset

		print("Start training")

		#find best learning rate
		trans_model.lr_find()

		# Train the model
		trans_model.fit_one_cycle(epochs,callbacks=[SaveModelCallback(trans_model, every='improvement', mode = 'max', monitor='accuracy', name=save_loc)])


		# save the transfer learning model for our dataset
		#trans_model.save(save_loc) #save the weights of the model
		#trans_model.export(save_loc)

		# Analyze the results by checking which classes gave the highest loss
		#preds,y,losses = trans_model.get_preds(with_loss=True)
		#interpreter = ClassificationInterpretation(trans_model, preds, y, losses)	
		#interpreter.top_losses(10,largest=True)  #plot the top 10 largest losses
		#print("These are the losses")
		#print(interpreter.top_losses(10,largest=True))

		
