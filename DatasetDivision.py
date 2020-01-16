import os
from sklearn.model_selection import train_test_split
import numpy as np
import glob
import shutil



class DatasetDivision:

	#root_dir = "/home/cantonioupao/Desktop/SIPakMed"
	#output_dir = "home/cantonioupao/Desktop/SIPakMed/Divided_Dataset"
	def __init__(self, root_dir="",output_dir=""):
		self.root_dir = root_dir
		self.output_dir = output_dir
		print("Instance of the class created")

	def printnow(self, new_dir):
		print("Just testing that the method calling is working"+new_dir)


	def divide_dataset(self, root_dir,output_dir):
		self.root_dir =root_dir
		self.output_dir = output_dir
		if os.path.exists(self.output_dir):
			if not os.path.exists(os.path.join(self.output_dir,'train')):
				os.mkdir(os.path.join(self.output_dir,'train'))  #create the first directory
				os.mkdir(os.path.join(self.output_dir,'val')) # 2nd directory
				os.mkdir(os.path.join(self.output_dir,'test')) #3 directory
		else:
			os.mkdir(self.output_dir)
			os.mkdir(os.path.join(self.output_dir,'train')) #create the first directory
			os.mkdir(os.path.join(self.output_dir, 'val')) # 2nd directory
			os.mkdir(os.path.join(self.output_dir, 'test')) #3 directory
		# Split train/val/test sets
		for file in os.listdir(root_dir):            #for any file inside the root directory 
			classes_path = os.path.join(root_dir, file)  #fSo for every folder class we create a class directory
			class_files = [name for name in glob.glob(os.path.join(classes_path,'*.bmp'))]  #alternatively we can use the globe as mentioned
			train_and_valid, test = train_test_split(class_files, test_size=0.2, random_state=42)  #this signifies that our test dataset will e the 20% of the dataset - sklearn function#
			train, val = train_test_split(train_and_valid, test_size=0.25, random_state=42)  #this signifies that the validation dataset will be 20% of it , leaving 60% for training #

			#Define the training, validation and testing directories that the frame folders will be moved to.
			train_dir = os.path.join(self.output_dir, 'train',file) #creates the path for Divided_Dataset->train->Dyskeratotic 
			val_dir = os.path.join(self.output_dir, 'val', file) #creates the path for Divided_Dataset->val->Dyskeratotic 
			test_dir = os.path.join(self.output_dir, 'test',file) #creates the path for Divided_Dataset->test->Dyskeratotic 
			if not os.path.exists(train_dir):
				os.mkdir(train_dir)
			if not os.path.exists(val_dir):
				os.mkdir(val_dir)
			if not os.path.exists(test_dir):
				os.mkdir(test_dir)

			for frame_folders in train:
				#get only the last directory of the path frame_folders
				frame_folder = os.path.join(root_dir,file,frame_folders)
				shutil.move(frame_folder,train_dir)
			for frame_folders in val:
				frame_folder = os.path.join(root_dir,file,frame_folders)
				shutil.move(frame_folder,val_dir)
			for frame_folders in test:
				frame_folder = os.path.join(root_dir,file,frame_folders)
				shutil.move(frame_folder,test_dir)
			print('Dataset Division finished.')        




