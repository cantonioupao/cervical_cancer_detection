import  cv2
import imgaug
import imageio
import os
import numpy as np
from imgaug import augmenters as iaa


class DataAugmentation:


	def __init__(self, root_dir="",output_dir=""):
		self.root_dir = root_dir
		self.output_dir = output_dir
		print("Instance of the DataAugmentation class created")


	def augmentation_of_image(self, test_image, output_path):
		self.test_image = test_image;
		self.output_path = output_path;
		#define the Augmenters



		#properties: A range of values signifies that one of these numbers is randmoly chosen for every augmentation for every batch

		# Apply affine transformations to each image.
		rotate = iaa.Affine(rotate=(-90,90));  
		scale = iaa.Affine(scale={"x": (0.5, 0.9), "y": (0.5,0.9)}); 
		translation = iaa.Affine(translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)});
		shear = iaa.Affine(shear=(-2, 2)); #plagio parallhlogrammo wihthin a range (-8,8)
		zoom = iaa.PerspectiveTransform(scale=(0.01, 0.15), keep_size=True) # do not change the output size of the image
		h_flip = iaa.Fliplr(1.0); # flip horizontally all images (100%)
		v_flip = iaa.Flipud(1.0); #flip vertically all images
		padding=iaa.KeepSizeByResize(iaa.CropAndPad(percent=(0.05, 0.25)))#positive values correspond to padding 5%-25% of the image,but keeping the origial output size of the new image


		#More augmentations
		blur = iaa.GaussianBlur(sigma=(0, 1.22)) # blur images with a sigma 0-2,a number ofthis range is randomly chosen everytime.Low values suggested for this application
		contrast = iaa.contrast.LinearContrast((0.75, 1.5)); #change the contrast by a factor of 0.75 and 1.5 sampled randomly per image
		contrast_channels = iaa.LinearContrast((0.75, 1.5), per_channel=True) #and for 50% of all images also independently per channel:
		sharpen = iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)); #sharpen with an alpha from 0(no sharpening) - 1(full sharpening) and change the lightness form 0.75 to 1.5 
		gauss_noise = iaa.AdditiveGaussianNoise(scale=0.111*255, per_channel=True) #some random gaussian noise might occur in cell images,especially when image quality is poor
		laplace_noise = iaa.AdditiveLaplaceNoise(scale=(0, 0.111*255)) #we choose to be in a small range, as it is logical for training the cell images


		#Brightness 
		brightness = iaa.Multiply((0.35,1.65)) #change brightness between 35% or 165% of the original image
		brightness_channels = iaa.Multiply((0.5, 1.5), per_channel=0.75) # change birghtness for 25% of images.For the remaining 75%, change it, but also channel-wise.

		#CHANNELS (RGB)=(Red,Green,Blue)
		red =iaa.WithChannels(0, iaa.Add((10, 100))) #increase each Red-pixels value within the range 10-100
		red_rot = iaa.WithChannels(0,iaa.Affine(rotate=(0, 45))) #rotate each image's red channel by 0-45 degrees
		green= iaa.WithChannels(1, iaa.Add((10, 100)))#increase each Green-pixels value within the range 10-100
		green_rot=iaa.WithChannels(1,iaa.Affine(rotate=(0, 45))) #rotate each image's green channel by 0-45 degrees
		blue=iaa.WithChannels(2, iaa.Add((10, 100)))#increase each Blue-pixels value within the range 10-100
		blue_rot=iaa.WithChannels(2,iaa.Affine(rotate=(0, 45))) #rotate each image's blue channel by 0-45 degrees

		#colors
		channel_shuffle =iaa.ChannelShuffle(1.0); #shuffle all images of the batch
		grayscale = iaa.Grayscale(1.0)
		hue_n_saturation = iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True) #change hue and saturation with this range of values for different values 
		add_hue_saturation = iaa.AddToHueAndSaturation((-50, 50), per_channel=True) #add more hue and saturation to its pixels
		#Quantize colors using k-Means clustering
		kmeans_color = iaa.KMeansColorQuantization(n_colors=(4, 16)) #quantizes to k means 4 to 16 colors (randomly chosen). Quantizes colors up to 16 colors

		#Alpha Blending 
		blend =iaa.AlphaElementwise((0, 1.0), iaa.Grayscale((0,1.0))) ; #blend depending on which value is greater

		#Contrast augmentors
		clahe = iaa.CLAHE(tile_grid_size_px=((3, 21),[0,2,3,4,5,6,7])) #create a clahe contrast augmentor H=(3,21) and W=(0,7)
		histogram = iaa.HistogramEqualization() #performs histogram equalization

		#Augmentation list of metadata augmentors
		OneofRed = iaa.OneOf( [red]);
		OneofGreen = iaa.OneOf( [green] );
		OneofBlue = iaa.OneOf( [blue]);
		contrast_n_shit = iaa.OneOf([contrast, brightness, brightness_channels]);
		SomeAug = iaa.SomeOf(2,[rotate,scale, translation, shear, h_flip,v_flip],random_order=True);
		SomeClahe = iaa.SomeOf(2, [clahe, iaa.CLAHE(clip_limit=(1, 10)),iaa.CLAHE(tile_grid_size_px=(3, 21)),iaa.GammaContrast((0.5, 2.0)), iaa.AllChannelsCLAHE() , iaa.AllChannelsCLAHE(clip_limit=(1, 10), per_channel=True)],random_order=True) #Random selection from clahe augmentors
		edgedetection= iaa.OneOf([iaa.EdgeDetect(alpha=(0, 0.7)),iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0))]);# Search in some images either for all edges or for directed edges.These edges are then marked in a black and white image and overlayed with the original image using an alpha of 0 to 0.7.
		canny_filter = iaa.OneOf([iaa.Canny(), iaa.Canny(alpha=(0.5, 1.0), sobel_kernel_size=[3, 7])]); #choose one of the 2 canny filter options
		OneofNoise = iaa.OneOf([blur, gauss_noise, laplace_noise])
		Color_1 = iaa.OneOf([channel_shuffle,grayscale, hue_n_saturation , add_hue_saturation, kmeans_color]);
		Color_2 = iaa.OneOf([channel_shuffle,grayscale, hue_n_saturation , add_hue_saturation, kmeans_color]);
		Flip = iaa.OneOf([histogram , v_flip, h_flip]);

		#Define the augmentors used in the DA
		Augmentors= [SomeAug, SomeClahe, SomeClahe, edgedetection,sharpen, canny_filter, OneofRed, OneofGreen, OneofBlue, OneofNoise, Color_1, Color_2, Flip, contrast_n_shit]


		for i in range(0,14):
			img = cv2.imread(test_image) #read you image
			images = np.array([img for _ in range(14)], dtype=np.uint8)  # 12 is the size of the array that will hold 8 different images
			images_aug = Augmentors[i].augment_images(images)  #alternate between the different augmentors for a test image
			cv2.imwrite(os.path.join(output_path,test_image +"new"+str(i)+'.jpg'), images_aug[i])  #write all changed images


		#implementation - save new DA image
		# imglist = []
		# image = cv2.imread("test.jpg");
		# imglist.append(image);
		# image_aug = SomeClahe.augment_images(imglist);
		# cv2.imwrite("path.join(output_path, augmented_image.jpg")), image_aug[0]);
