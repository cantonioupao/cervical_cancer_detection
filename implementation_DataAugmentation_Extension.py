from DataAugmentation_Extension import DataAugmentation_Extension

#directories
target_directory = "/cluster/scratch/cantoniou/Experimentation/sipakmed"
#create an instance of the class
datasetda = DataAugmentation_Extension()
datasetda.extend_dataset(target_directory)
