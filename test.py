# example of loading a pix2pix model and using it for image to image translation
from keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
from util import float_to_int
from visualize import plot_images
from util import save_compressed_npy, load_real_samples
import os
from PIL import Image
import numpy as np
import time
import tensorflow as tf

if __name__ == '__main__':
    path =  './maps/test/'
    save_compressed_npy(path, 'maps_256_test.npz')

    # load dataset
    [X1, X2] = load_real_samples('maps_256_test.npz')                                  # 'maps_256_test.npz'
    print('Loaded', X1.shape, X2.shape)
    # load model
    model = load_model('./model_000500.h5', compile=False)
    save_path='generated'
    fnameV=[]
    # select random example
    #ix = randint(0, len(X1), 1)
    for filename in os.listdir(path):
        fnameV.append(filename)

    for i in range(len(X1)):
        ix=[i]
        src_image, tar_image = X1[ix], X2[ix]
        # generate image from source
        t1=time.time()
        gen_image = model.predict(src_image)
        t2 = time.time()
        print("Pred Time:", t2-t1)
        #g2 = model.predict(tar_image)
        # plot all three images
        plot_images(src_image, gen_image, tar_image)
        tar_image = (tar_image + 1) / 2.0
        gen_image = (gen_image + 1) / 2.0
        src_image = (src_image + 1) / 2.0
        
        #g2 = (g2 + 1) / 2.0
        
        gen_image=float_to_int(gen_image)
        tar_image=float_to_int(tar_image)
        src_image=float_to_int(src_image)
        
        #g2=float_to_int(g2)
        
        #hvs=fnameV[i].split('.')[0] +'_1'+'.bmp'                                      #Change for training
        #plt=fnameV[i].split('.')[0] +'_2'+'.bmp'                                      #Change for training
        
        #cv2.imwrite('generated_img.bmp', np.squeeze(gnm))
        #plt.imsave('gen.bmp',gen_image, format='bmp')
        save_path_surface=os.path.join(save_path, fnameV[i].split('.')[0])             #Change for training
        if not os.path.exists(save_path_surface):
            os.makedirs(save_path_surface)
        
        
        #im = Image.fromarray(np.squeeze(gen_image))
        #im.save(os.path.join(save_path_surface,plt))
        
        #im = Image.fromarray(np.squeeze(tar_image))
        #im.save(os.path.join(save_path_surface,hvs))
        
        #im = Image.fromarray(np.squeeze(src_image))
        #im.save(os.path.join(save_path_surface,"src_Img.bmp"))
        
        #im = Image.fromarray(np.squeeze(g2))
        #im.save(os.path.join(save_path_surface,"g2.bmp"))
        
        im = Image.fromarray(np.squeeze(gen_image))
        im.save(os.path.join(save_path_surface,"gen_Img.bmp"))
    
        im = Image.fromarray(np.squeeze(tar_image))
        im.save(os.path.join(save_path_surface,"tar_Img.bmp"))
    
        im = Image.fromarray(np.squeeze(src_image))
        im.save(os.path.join(save_path_surface,"src_Img.bmp"))
    
    