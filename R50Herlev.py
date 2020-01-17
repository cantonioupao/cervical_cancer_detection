from fastai import *
from fastai.vision import *
from fastai.callbacks import *
#import Path
import torch
import torchvision
import torchvision
from torchvision import models as models
from LoadWeights import LoadWeights as Load

#torch.cuda.set_device(0)
#hyperparameters
batch_size = 5
epochs = 10
current_classes_of_dataset = 7;
output_classes_of_nextdataset = 5 ;
previous_model_DA_data_path = "/cluster/home/cantoniou/deep_project/sipakmedFormat"
previous_model_DA = "/cluster/home/cantoniou/deep_project/sipakmedFormat/models/"
save_loc = "resnet50model_trainedonDAandHerlevdataset" + str(epochs) + "batch" + str(batch_size) #location to save the model


#################SECTION TO LOAD LEARNER WITH PREVIOUS DATA AND CHANGE OUTPUT HEADER(OF PREVIOUS MODEL)#####################################
#previous data
#prev_data = ImageDataBunch.from_folder(path=previous_model_DA_data_path, train='train',
#          valid='val', ds_tfms=get_transforms(), size = 224, bs=batch_size)#, check_ext=False)
#prev_data.normalize(imagenet_stats)
#name = 'ExtendedDAwith7outputs'
#prev_model =cnn_learner(prev_data, models.resnet50, metrics=[accuracy,FBeta(average="weighted")]).load(previous_model_DA)
#unfreeze the convolutional base 
#prev_model.unfreeze()
#prev_model.lr_find()
#prev_model.model[-1][-1]=nn.Linear(in_features=512 , out_features=current_classes_of_dataset, bias=True)
#previous_model_DA = previous_model_DA_data_path+"/"+name
#prev_model.save(name) #save the updated version of he previous model(with the desired output head)
############################################################################################################################################


## Declaring path of current dataset
path_img = Path("/cluster/home/cantoniou/deep_project/smear2005Format")
## Loading data 
data = ImageDataBunch.from_folder(path=path_img, train='train',
            valid='val', ds_tfms=get_transforms(), size = 224, bs=batch_size)#, check_ext=False)
## Normalizing data based on Image net parameters
data.normalize(imagenet_stats)
#data.show_batch(rows=3, figsize=(10,8))
print(data.classes)
len(data.classes),data.c


#LOAD THE TRANSFER LEARNING MODEL
## To create a ResNET 50 with pretrained weights based on the new dataset
#trans_model= create_cnn(data, models.resnet34, metrics=error_rate)
trans_model= cnn_learner(data, models.resnet50, metrics=[accuracy,FBeta(average="weighted")])
trans_model.load(previous_model_DA)

#unfreeze the convolutional base 
trans_model.unfreeze()
trans_model.lr_find()

# Train the model
#trans_model.fit_one_cycle(epochs)
#train the model using a callback
trans_model.fit_one_cycle(epochs,callbacks=[SaveModelCallback(trans_model, every='improvement', mode = 'max', monitor='accuracy', name=save_loc)])

print(trans_model)
#change the output head to 5 classes(to match the new dataset used for training) , before saving the model
trans_model.model[-1][-1]=nn.Linear(in_features=512 , out_features=output_classes_of_nextdataset, bias=True) 

# save the transfer learning model for our dataset
trans_model.save(save_loc) #save the weights of the model
trans_model.export()

# Analyze the results by checking which classes gave the highest loss
#preds,y,losses = trans_model.get_preds(with_loss=True)
#interpreter = ClassificationInterpretation(trans_model, preds, y, losses)	
#interpreter.top_losses(10,largest=True)  #plot the top 10 largest losses
#print("These are the losses")
#print(interpreter.top_losses(10,largest=True))



#Call the LoadWeights file and train on the Sipakmed dataset using the pretrained model with updated weights
new_w1 = Load()
saved = new_w1.load_w(path_img/"models"/save_loc);



