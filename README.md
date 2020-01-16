# Cervical Cancer Detection from PAP-Test(smear) results using DL
### Overview 
A deep learning framework for cervical cancer classification to allow improved accuracy for PAP smear test 

### Brief Summary
As part of an ETH project a deep learning framework is developed for cervical cancer detection and classification based on microscopics images of cells from PAP smear results. The purpose of this project is to provide another tool for doctors to rapidly detect if a patient has developed or is in danger of developing cervical cancer.It consitutes a rapid tool for detection and prognosis of cervical cancer for female patients. 

### Dataset
The model will be trained on the Sipakmed which is a new Dataset for Feature and Image Based Classification of Normal and Pathological Cervical Cells in Pap Smear Images (https://www.researchgate.net/figure/The-boundaries-of-the-cytoplasm-and-the-nucleus-of-each-cell-in-images-of-cell-clusters_fig1_327995161). The dataset can be downloaded using the following link: http://www.cs.uoi.gr/~marina/sipakmed.html


### Procedure
1. Download the Sipakmed dataset
2.The Sipakmed dataset structure needs to be similar to this:

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
                  
3.Use the implementation_Dataset_Division to reform the dataset structure to the desired dataset structure with training_size = 0.6, validation_size=0.2 and testing_size=0.2

