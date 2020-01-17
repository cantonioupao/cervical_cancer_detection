# Deep Cervical Cancer: Cervical Cancer Classification using Deep Learning
### Overview 
A Deep learning framework for cervical cancer classification on the Sipakmed dataset, to allow improved accuracy for PAP smear test evaluation and cancer prognosis 
<table style="border:0px">
   <tr>
       <td><img src="official collage.png" frame=void rules=none></td>
   </tr>
</table>

### Brief Summary
As part of an ETH project a deep learning framework is developed for cervical cancer detection and classification based on microscopics images of cells from PAP smear results. The purpose of this project is to provide another tool for doctors to rapidly detect if a patient has developed or is in danger of developing cervical cancer.It consitutes a rapid tool for detection and prognosis of cervical cancer for female patients. 

### Dataset
The model will be trained on the [Sipakmed](https://www.researchgate.net/figure/The-boundaries-of-the-cytoplasm-and-the-nucleus-of-each-cell-in-images-of-cell-clusters_fig1_327995161) which is a new Dataset for Feature and Image Based Classification of Normal and Pathological Cervical Cells in Pap Smear Images . The dataset can be downloaded [here](http://www.cs.uoi.gr/~marina/sipakmed.html)
The dataset consists of 996 cluster cell images of Pap smear slides (whole slide images) and has 5 cervical cell classification categories; 
a) Dyskeratotic 
b) Koilocytotic
c) Metaplastic
d) Parabasal
e) Superficial-Intermediate
As a result, this project focuses on a 5 classes categorical classification based on whole slide microscopic cell images (not just cropped cell images, but whole slided)


### Procedure
1. Download the Sipakmed dataset

2. The Sipakmed dataset structure needs to be similar to the structure below.Use the "implementation_DatasetDivision.py" to reform the sipakmed dataset structure to the desired dataset structure, with training_size = 0.6, validation_size=0.2 and testing_size=0.2:

- **Sipakmed Dataset**
  ```
  sipakmed
  ├── train
  │   ├── im_Dyskertotic
  │   ├── im_Koilocytotic 
  │   ├── im_Metaplastic 
  │   ├── im_Parabasal  
  │   └── im_Superficial-Intermediate
  ├── val
  │   ├── im_Dyskertotic
  │   ├── im_Koilocytotic 
  │   ├── im_Metaplastic 
  │   ├── im_Parabasal  
  │   └── im_Superficial-Intermediate
  ├── test
  │   ├── im_Dyskertotic
  │   ├── im_Koilocytotic 
  │   ├── im_Metaplastic 
  │   ├── im_Parabasal  
  │   └── im_Superficial-Intermediate
  └── models
  ```
3. Install the fastai library and all its dependencies, as well as the torch library.

4. To train the Resnet50 pretrained network on the Sipakmed dataset use the "R50.py". Initially specificy in the file, the path to the sipakmed (Formatted) dataset---->"path_folder". Then choose appropriate hyperparameters (Epoch = 50 , Batch = 10 , Learning rate = 0.001) and start training.

5. After training is completed the model weights are saved and the whole model is exported.You can specify beforehand in the "R50.py", the path to the storage location to which you want the model weights(.pth) and the whole model(.pkl) to be saved

6. In case you have trained your model up to a certain checkpoint and you save it, you can reload the model weigths by specifying path to the model weights file(.pth)  ---> "model_path" variable in "R50.py"

7. After completing a simple training of the Sipakmed (Formatted) dataset, the results can be further analyzed with the aid of an inference file "Inference_FastAI.py". The inference file will load the model pickle file (.pkl) and will perform classification on any given dataset. It will predict any sample cell microscopic image. The inference file, is set up to evaluate the performance of the model on the testing dataset from the Sipakmed. However it can be carried out on any set of images or single image.The user needs to specify the "test_dataset_path" and the "model_path" with paths to where the testset and the model file are respectively. The inference file will output the accuracy, recall,precision values and the confusion matrix for the various cell categories.

## Further training

The main procedure for training a Resnet50 model on the Sipakmed dataset is described above. Nevertheless for improving results usually further training is requierd and hence the methods of data augmentation, triple transfer learning and feature combinations are used and described below.

**Data Augmentation**

1. For applying data augmentation to the sipakmed dataset, the "implementation_DataAugmentation_Extension.py" file is executed. Specify in the file the target directory path of the sipakmed dataset (or any other dataset you want to apply data augmentation). set the path of the directory through the variable ---> "target_directory"

2. After executing the file, 14 new augmented images will be generated for each image of the Sipakmed dataset. The data augmentations carried out include affine transformations,filter response, color channels tweaking and etc.The generated images are saved automatically in the same location with the original image. As a result the dataset size increases by a factor of 14, assisting with obtaining better results.

**Triple Transfer Learning**
TTL(Trisple Transfer Learning) means that the mode was trained on the 3 datasets in total. Initially the model is trained on the Extended DA dataset, which is the formatted sipakmed dataset after data augmentation has been carried out (procedure described in the above section). After training of the model is completed, the new weigths are saved in a .pth file. Afterwards the model is trained on the [Herlev dataset](https://www.researchgate.net/publication/326477913_Clustering_Techniques_on_Pap-smear_Images_for_the_Detection_of_Cervical_Cancer) which is a cervical cell dataset with 7 classification classes; 
a) Superficial Squamus Epithelial
b) Intermediate Squamus Epithelial
c) Columnar Epithelial
d) Mild Dysplasia 
e) Moderate Dysplasia
f) Severe Dysplasia
g) Carcinoma in Situ
consisting of 916 images of single cells. Because there are more classes to be classified in the Herlev dataset, the output head(softmax layers) needs to be changed to 7 outputs. This is done automatically in the "R50Herlev.py". The weights after training the on the Herlev are saved and again retrieved again in preparation for training on the Sipakmed dataset. Hence the weights are loaded to the model and the 3rd training is carried out on the Sipakmed dataset and the final model is saved(.pth) and exported(.pkl). The procedure to carry out TTL is described below:

1. Download the Herlev dataset [here](http://mde-lab.aegean.gr/images/stories/docs/smear2005.zip)

2. Make sure that you have all 3 datasets in your main directory.The Extended DA, the Herlev and the Sipakmed(formatted) dataset

3. Only the 'R50Herlev.py file needs to be executed to succesfully complete the TTL

4. The path of the model weights from training on the Extended DA dataset needs to be specified. Training on Extended DA and saving the weights is no different that training on the Sipakmed(Formatted). With the aid of "R50.py" the model is trained on the Extended DA. Remember that if you want to train on the Extended DA, just change the "path_folder" variable and specify the path to the Extnded DA this time. Then choose the location at which the model weights will be saved (after training on the Extended DA). Now the path to the weights's location you chose when training before, will be the path that needs to be specified in the "R50Herlev.py"

5. Then just make sure that the number of output classes is set to 7 and 5 for the Herlev and Sipakmed respectively. The variables "current_classes_of_dataset" and "output_classes_of_next_dataset" need to be set to 7 and 5 respectively. It's very important to set appropriately the number of outputs corresponding to each of your datasets, to avoid any size mismatch in the output softmax layers when loading weights and architecture from one model to another.

6. Execute the "R50Herlev.py" after setting appropriate hyperparameters. The final output model will saved under the path specified by "save_loc" in the "TTLR50.py" file. All 3 files are needed to succesfully run the "R50Herleve.py" file

7.After training is completed, the model is saved(.pth) and exported(.pkl).Once again the model can be loaded to the learner of the inference file for better interpration and analysis of the results. The inference file prints the accuracy, precision, recall values along with the confusion matrix for the 5 categories


![#f03c15](https://placehold.it/15/f03c15/000000?text=+) 
**In general all python files are detailed with many descriptive and helpful comments that will guide you to any step described in the above procedures**  ![#f03c15](https://placehold.it/15/f03c15/000000?text=+) 


## Best framework
The model with the highest accuracy was obtained by implementing the TTL method. The best performing framework yielded outstanding results with very high accuracy,recall and precision vallues. The confusion matrix obtained with the aid of the inference file is shown below. The framework's outstanding performance on the Sipakmed, makes it one of the state-of-the-art frameworks of each field. The unprecedented accuacy achieved can be further improved, by combining TTL with the Feature Combination method. Soon more to come!


 

