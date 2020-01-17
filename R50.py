from fastai import *
from fastai.vision import *
from fastai.callbacks import *
#import Path
#import torch
#import torchvision
#import torchvision
#from torchvision import models as models

res = models.resnet50
#hyperparameters
batch_size = 10
epochs = 50
#lr = 0.01 #you can set a specific learning rate or just let it perform cyclic training

#Storing path
save_loc = 'resnet50model_trainedonSIPAKMEDdataset5000' + str(epochs) + "batch" + str(batch_size)

## Declaring path of dataset
path_img = Path("/cluster/home/cantoniou/deep_project/sipakmedFormat")

#Declaring the .pth path for the model weights
weights_path = path_img/'models'/"resnet50modeltrainedon_DA_HERLEV_SIPAKMED95accuracy"/"resnet50model_trainedonExtendedDAdataset50batch100.001BEST"  #this needs to be of .pth extension

#Model path (.pkl) to the folder with the "export.pkl" seraialization file
model_path = path_img/'models'/"resnet50SimplesipakmedE10B5accuracy87"   #this needs to be of .pkl extension and it needs to have the name "export.pkl"
## Loading data 
data = ImageDataBunch.from_folder(path=path_img, train='train',
            valid='val', ds_tfms=get_transforms(), size = 224, bs=batch_size)#, check_ext=False)  #the size of the input pictures is quite important
## Normalizing data based on Image net parameters
data.normalize(imagenet_stats)
#data.show_batch(rows=3, figsize=(10,8))
print(data.classes)
len(data.classes),data.c


#make sure you are using a gpu
defaults.device = torch.device('cuda')


#LOAD THE TRANSFER LEARNING MODEL
## To create a ResNET 50 with pretrained weights based on the new dataset
#trans_model= create_cnn(data, models.resnet34, metrics=error_rate)
trans_model= cnn_learner(data, models.resnet50, metrics=[accuracy,FBeta(average="weighted")])


#Load the weights from a pevious training if you wish
#trans_model.load(weights_path)  #uncomment only when you want the weights from another saved model etc. from "yourfile.pth"


# Train the model
#trans_model.freeze()
#trans_model.fit_one_cycle(epochs,callbacks=[SaveModelCallback(trans_model, every='improvement', mode = 'max', monitor='accuracy', name=save_loc)])

#unfreeze the model and its convolutional base
trans_model.unfreeze()

#Try to find the best lr and print the plot of lr vs loss function (cyclic training observed)
trans_model.lr_find()
#trans_model.recorder.plot()
#Train again
trans_model.fit_one_cycle(epochs,callbacks=[SaveModelCallback(trans_model, every='improvement', mode = 'max', monitor='accuracy', name=save_loc)])


#trans_model.model[-1][-1]=nn.Linear(in_features=512 , out_features=7, bias=True) #uncomment only when you want to train on the Extended DA and then use the saved model weights in the "R50Herlev.py"
#The above line changes the output head to a size of 7 outputs.



# save the transfer learning model for our dataset and export the whole model
trans_model.save(save_loc)
trans_model.export()

#show results
print("The validation loss and accuracy is:"); print(trans_model.validate(trans_model.data.valid_dl)) #print the validation accuracy
print("The training loss and accuracy is:"); print(trans_model.validate(trans_model.data.train_dl)) #print the training accuracy 
trans_model.show_results(ds_type=DatasetType.Train) #show the training results
trans_model.show_results(ds_type=DatasetType.Valid) #show the validation results



