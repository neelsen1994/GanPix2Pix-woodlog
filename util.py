# load, split and scale the maps dataset ready for training
from os import listdir
from numpy import asarray
import numpy as np
from numpy import vstack
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
#from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from matplotlib import pyplot
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os


# load all images in a directory into memory
def load_images(path, size=(512,1024)):
	src_list, tar_list = list(), list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# split into satellite and map
		sat_img, map_img = pixels[:, :512], pixels[:, 512:]
		src_list.append(sat_img)
		tar_list.append(map_img)
		#fnameV.append(filename)                                       #Change for training
	return [asarray(src_list), asarray(tar_list)]

def save_compressed_npy(path, filename):
	#path =  './maps/val/'
	[src_images, tar_images] = load_images(path)
	print('Loaded: ', src_images.shape, tar_images.shape)
	#filename = 'maps_256.npz' 
	savez_compressed(filename, src_images, tar_images)
	print('Saved dataset: ', filename)

# load and prepare training images
def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

def generate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset
	trainA, trainB = dataset
	# choose random instances
	ix = randint(0, trainA.shape[0], n_samples)
	# retrieve selected images
	X1, X2 = trainA[ix], trainB[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

def float_to_int(img):
    # Scale the input image to the range [0, 255]
    img = img * 255.0
    # Convert the floating point values to integers
    img = img.astype(np.uint8)
    # Return the resulting image
    return img