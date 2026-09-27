#------------------------------------------------------------------------------
#%% load data
#------------------------------------------------------------------------------

import scipy.io as sio

import numpy as np

from matplotlib import pyplot as plt


# path of the toolbox:
data_path = '/autofs/cluster/kawin/berkin/Python_Code/Multishot_EPI/MS_EPI_Residual_CNN_Toolbox/'

#------------------------------------------------------------------------------
# training: use 3 of 4 subjects
dt = sio.loadmat(data_path + '/Img_Pat_142x142x48x5x3x4_single.mat')

# clean data:
img_pat_train = dt['Img_Pat']

# average over shots:
img_pat_train = np.mean(img_pat_train, 4)

img_pat_train = img_pat_train[:,:,:,:,1::]


# corrupted sliding-window data:
dt = sio.loadmat(data_path + 'Img_Sw_142x142x48x5x4_single.mat')

img_sw_train = dt['Img_Sw']

img_sw_train = img_sw_train[:,:,:,:,1::]
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# test: use single subject
dt = sio.loadmat(data_path + 'Img_Pat_142x142x48x5x3x4_single.mat')

# clean data:
img_pat_test = dt['Img_Pat']

img_pat_test = np.mean(img_pat_test, 4)

img_pat_test = img_pat_test[:,:,:,:,0]


# corrupted sliding-window data:
dt = sio.loadmat(data_path + 'Img_Sw_142x142x48x5x4_single.mat')

img_sw_test = dt['Img_Sw']

img_sw_test = img_sw_test[:,:,:,:,0]
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
#%% function definitions
#------------------------------------------------------------------------------


import numpy.fft as npf


def imagesc( img, fig_num, clim, title = '' ):

      
    fig = plt.figure(fig_num)
    fig.patch.set_facecolor('black')

    plt.imshow( img )
    plt.gray()
    plt.clim(clim)
    plt.suptitle(title, color='white', fontsize=48)




def fft2c( img ):

    k =  npf.fftshift( npf.fft2( npf.fftshift( img ) ) )

    return k
    

def ifft2c( k ):

    img =  npf.fftshift( npf.ifft2( npf.fftshift( k ) ) )

    return img



def mosaic(img, num_row, num_col, fig_num, clim, fig_title=''):
    
    fig = plt.figure(fig_num)
    fig.patch.set_facecolor('black')

    if img.ndim < 3:
        img_res = img
        plt.imshow(img_res)
        plt.gray()        
        plt.clim(clim)

    else:
        
        if img.shape[2] != (num_row * num_col):
            print('sizes do not match')    
        else:   
            img_res = np.zeros((img.shape[0]*num_row, img.shape[1]*num_col))
            
            idx = 0
            
            for r in range(0, num_row):
                for c in range(0, num_col):
                    img_res[r*img.shape[0] : (r+1)*img.shape[0], c*img.shape[1] : (c+1)*img.shape[1]] = img[:,:,idx]
                    idx = idx + 1
                
        plt.imshow(img_res)
        plt.gray()        
        plt.clim(clim)
        plt.title(fig_title, color='white')
        


mosaic( img_pat_train[:,:,24,:,1], 1, 5, 1, [0,0.5] )
mosaic( img_sw_train[:,:,24,:,1], 1, 5, 2, [0,0.5] )

mosaic( img_pat_test[:,:,24,:], 1, 5, 3, [0,0.5] )
mosaic( img_sw_test[:,:,24,:], 1, 5, 4, [0,0.5] )

 
 
#------------------------------------------------------------------------------
#%% scale data from 0 to 1
#------------------------------------------------------------------------------

subj_use = 3

img_train_clean = img_pat_train[:,:,:,:,:subj_use]
img_test_clean = img_pat_test


img_train_corrupt = img_sw_train[:,:,:,:,:subj_use]
img_test_corrupt = img_sw_test


row, col, slc, num_te, num_subj = img_train_clean.shape



scale_factor = np.max( np.abs( img_train_clean ) )

img_train_clean /= scale_factor
img_test_clean /= scale_factor

img_train_corrupt /= scale_factor
img_test_corrupt /= scale_factor


imagesc( np.concatenate( (img_train_clean[:,:,20,0,0], img_train_corrupt[:,:,20,0,0]), 1 ), 1, [-.0,.3] )
imagesc( np.concatenate( (img_train_clean[:,:,20,2,1], img_train_corrupt[:,:,20,2,1]), 1 ), 2, [-.0,.2] )
imagesc( np.concatenate( (img_train_clean[:,:,20,2,2], img_train_corrupt[:,:,20,2,2]), 1 ), 3, [-.0,.2] )



#------------------------------------------------------------------------------
#%% extract patches: training
#------------------------------------------------------------------------------

patch_size = 51

step_size = 3

patch_hsize = int( (patch_size - 1) / 2 )


ind = 0

for s in range(0, num_subj):
    for n in range(0, slc):
        for x in range(patch_hsize, row - patch_hsize, step_size):
            for y in range(patch_hsize, col - patch_hsize, step_size):   
                ind = ind + 1

print(ind)



patch_noisyImag = np.zeros((ind, patch_size, patch_size, num_te))
patch_noiseOnly = np.zeros((ind, patch_size, patch_size, num_te))


ind = 0
    

for s in range(0, num_subj):
    for n in range(0, slc):
        for x in range(patch_hsize, row - patch_hsize, step_size):
            for y in range(patch_hsize, col - patch_hsize, step_size):
        
                patch_noisyImag[ind, :, :, :] = img_train_corrupt[x - patch_hsize : x + patch_hsize + 1, y - patch_hsize : y + patch_hsize + 1, n, :, s]
    
                patch_noiseOnly[ind, :, :, :] = img_train_corrupt[x - patch_hsize : x + patch_hsize + 1, y - patch_hsize : y + patch_hsize + 1, n, :, s] \
                                                - img_train_clean[x - patch_hsize : x + patch_hsize + 1, y - patch_hsize : y + patch_hsize + 1, n, :, s]
                
                ind = ind + 1


# display random patch
rint = np.random.randint(0,ind)

imagesc( patch_noisyImag[rint,:,:,3], 1, [0,.2] )
imagesc( patch_noiseOnly[rint,:,:,3], 2, [0,.2] )
imagesc( patch_noisyImag[rint,:,:,3] - patch_noiseOnly[rint,:,:,3], 3, [0,.2] )

 
 
#------------------------------------------------------------------------------
#%% extract patches: test
#------------------------------------------------------------------------------

step_size = 6


ind = 0

for n in range(0, slc):
    for x in range(patch_hsize, row - patch_hsize, step_size):
        for y in range(patch_hsize, col - patch_hsize, step_size):   
            ind = ind + 1

print(ind)



patch_noisyImag_test = np.zeros((ind, patch_size, patch_size, num_te))
patch_noiseOnly_test = np.zeros((ind, patch_size, patch_size, num_te))


ind = 0
    
for n in range(0, slc):
    for x in range(patch_hsize, row - patch_hsize, step_size):
        for y in range(patch_hsize, col - patch_hsize, step_size):
    
            patch_noisyImag_test[ind, :, :, :] = img_test_corrupt[x - patch_hsize : x + patch_hsize + 1, y - patch_hsize : y + patch_hsize + 1, n, :]
     
            patch_noiseOnly_test[ind, :, :, :] = img_test_corrupt[x - patch_hsize : x + patch_hsize + 1, y - patch_hsize : y + patch_hsize + 1, n, :] \
                                            - img_test_clean[x - patch_hsize : x + patch_hsize + 1, y - patch_hsize : y + patch_hsize + 1, n, :]

     
            ind = ind + 1

 

# display random patch
rint = np.random.randint(0,ind)

imagesc( patch_noisyImag_test[rint,:,:,0], 1, [0,.2] )
imagesc( np.abs(patch_noiseOnly_test[rint,:,:,0]), 2, [0,.2] )
imagesc( patch_noisyImag_test[rint,:,:,0] - patch_noiseOnly_test[rint,:,:,0], 3, [0,.2] )



#------------------------------------------------------------------------------
#%% setup cnn
#------------------------------------------------------------------------------


from keras.layers import Conv2D, Activation, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam



kernel_size = [3, 3]
output_size = 64
num_layers = 25



model = Sequential()

# layer 1
model.add(Conv2D(output_size, kernel_size, input_shape = (patch_size, patch_size, num_te), padding = 'same'))
model.add(Activation('relu'))


# mid layers
for layers in range(1, num_layers-1):
    model.add(Conv2D(output_size, kernel_size, padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))


# last layer 
model.add(Conv2D(num_te, kernel_size, padding = 'same'))


adam_opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


model.compile(loss = 'mean_absolute_error', optimizer = adam_opt)



#------------------------------------------------------------------------------
#%% train cnn
#------------------------------------------------------------------------------

import time

num_epochs = 100
num_batch = 128


tic = time.clock()


hist = model.fit(patch_noisyImag, patch_noiseOnly, epochs = num_epochs, batch_size = num_batch, shuffle=True, validation_data=(patch_noisyImag_test, patch_noiseOnly_test))


toc = time.clock()


print(toc-tic)


model.save(data_path + 'ms_epi_residual_cnn_25layers.h5')


plt.figure(51)

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim((0,3e-2))



#------------------------------------------------------------------------------
#%% load pre-trained cnn
#------------------------------------------------------------------------------


from keras.models import load_model

# train on 3 subjects, test on 1 subject, average over TRs:
model = load_model(data_path + 'multishot_epi_dncnn_25layers_3shots_5echoes_4subjects_1tr.h5')



#------------------------------------------------------------------------------
#%% process test dataset
#------------------------------------------------------------------------------

patch_size = 51

step_size = 4

patch_hsize = int( (patch_size - 1) / 2 )




im_in = img_test_corrupt

# zero-pad for boundaries        
img_in = np.pad(im_in, ((patch_hsize, patch_hsize), (patch_hsize, patch_hsize), (0,0), (0,0)), 'constant')


img_true = img_test_clean


Row, Col, Slc, Te = img_in.shape

psf_img = np.zeros((Row,Col,Slc,Te))
res_img = np.zeros((Row,Col,Slc,Te))


# loop over slices and patches
for s in range(0, Slc):
    print(s)
    
    for x in range(patch_hsize, Row - patch_hsize, step_size):            
        for y in range(patch_hsize, Col - patch_hsize, step_size):
                
            temp = img_in[x - patch_hsize : x + patch_hsize + 1, y - patch_hsize : y + patch_hsize + 1, s, :].reshape((1,patch_size,patch_size,Te))  
            
            noise_predict = model.predict( temp )      
            
            patch_predict = temp - noise_predict
            
            psf_img[x - patch_hsize : x + patch_hsize + 1, y - patch_hsize : y + patch_hsize + 1, s, :] += 1
            res_img[x - patch_hsize : x + patch_hsize + 1, y - patch_hsize : y + patch_hsize + 1, s, :] += patch_predict.reshape((patch_size,patch_size,Te))
    
    
    
# normalize by the number of patches averaged over each voxel:    
res_norm = np.divide(res_img, psf_img + 1e-16)
res_norm = res_norm[patch_hsize:-patch_hsize, patch_hsize:-patch_hsize, :, :]
    
img_in = img_in[patch_hsize:-patch_hsize, patch_hsize:-patch_hsize, :, :]


rmse_in = 100 * np.sqrt(np.sum(np.square( img_in - img_true ))) / np.sqrt(np.sum(np.square( img_true )))
rmse_cnn = 100 * np.sqrt(np.sum(np.square( res_norm - img_true ))) / np.sqrt(np.sum(np.square( img_true )))

print(rmse_cnn, rmse_in)
 


mosaic( img_in[:,:,::2,0], 4, 6, 1, [0,.2], 'sliding window recon: ' + str(rmse_in) )

mosaic( res_norm[:,:,::2,0], 4, 6, 2, [0,.2], 'residual cnn: ' + str(rmse_cnn))

mosaic( img_true[:,:,::2,0], 4, 6, 3, [0,.2], 'clean data' )



#------------------------------------------------------------------------------
#%% grab middle slice and show all TE's:
#------------------------------------------------------------------------------

res_norm_mid = res_norm[:,:,23,:]
img_in_mid = img_in[:,:,23,:]
img_true_mid = img_true[:,:,23,:]
    

rmse_in_mid = 100 * np.sqrt(np.sum(np.square( img_in_mid - img_true_mid ))) / np.sqrt(np.sum(np.square( img_true_mid )))
rmse_cnn_mid = 100 * np.sqrt(np.sum(np.square( res_norm_mid - img_true_mid ))) / np.sqrt(np.sum(np.square( img_true_mid )))



mosaic( img_in_mid, 1, 5, 11, [0,.2], 'sliding window recon: ' + str(rmse_in_mid) )

mosaic( res_norm_mid, 1, 5, 12, [0,.2], 'residual cnn: ' + str(rmse_cnn_mid))

mosaic( img_true_mid, 1, 5, 13, [0,.2], 'clean data' )

    
