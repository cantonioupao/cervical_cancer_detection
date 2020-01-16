from DatasetDivision import DatasetDivision


#directories
path_dir= "/cluster/home/cantoniou/deep_project/smear2005"
output_dir = "/cluster/home/cantoniou/deep_project/smear2005Format"
#create an instance of the class
datasetdiv1 = DatasetDivision()
datasetdiv1.printnow("The new guy")
datasetdiv1.divide_dataset(path_dir, output_dir)
