#========================================================================================================
#
# Deep Learning Based Joint Source and Channel Coding and Power Control
#
# main.py
#
# 0. import packages
# 1. Settings: DNN structure parameters & system parameters
# 2. Train, test data
# 3. DNN object
# 4. Train
# 5. Test
#
#========================================================================================================




#========================================================================================================
#
# 0. import packages
#
#========================================================================================================
import os
import glob
import csv
import numpy as np
from dnn import Dnn




#========================================================================================================
#
# 1. Settings: DNN structure & system parameters
#
#========================================================================================================
num_epoch = 500

lr = np.array([1e-3])
# lr = np.array([3e-4])

Batch_Size = 1

num_sample_train = 1600
num_sample_test = 400

no_link = 2

# no_pkt = 3
no_pkt = 12

# DNN structure
Layer_dim_list = [16, no_pkt*no_link+no_link]

# noise
noise = 1e-12*(10**(-0.7))

# R: spectral efficiency
R = np.array([1,2,3,4]).reshape([1,-1])     #1x4

# bits
image_size = 512 * 512 * 1  # 512x512 pixels, 1bpp
maximum_pkt_size = image_size / no_pkt  # the max num of bits per packet

# distortion step
D_STEP = 2 ** 14
STEP = np.arange(0, int(np.ceil(image_size / 64)) + 1, int(D_STEP / 64))    # e.g. D_STEP=16384 -> STEP=[0, 16384, 32768, ...]

# for psnr constraint in loss function
Gap_thr = 3
Alpha = 0.3
Delta = 1

# the number of path loss per images
num_pl_per_img_train = 20
num_pl_per_img_test = 20


#========================================================================================================
#
# 2. Train, test data
#
#   data: distortion-rate characteristic, path loss
#
#========================================================================================================
# ----------------------------------- distortion-rate characteristic -----------------------------------
i = 0
num_total_dr = num_sample_train+num_sample_test
total_input_dr = np.zeros([num_total_dr, STEP.shape[0]]) # shape: (num_total_dr, 17)

# file path
input_path = 'C:\python\sungmi\distortion_new'

# load train and test data from files
for input_file in glob.glob(os.path.join(input_path, 'distort_a*')):
    with open(input_file, 'r') as f:
        rdr = csv.reader(f, delimiter='\t')
        temp = np.array(list(rdr), dtype=np.float64)
        total_input_dr[i, :] = temp[STEP, 1]
        i = i + 1

    if i == num_total_dr:
        break

# split data into train and test data
input_dr_train = total_input_dr[:num_sample_train,:]
input_dr_test = total_input_dr[-num_sample_test:,:]



# ----------------------------------- path loss -----------------------------------
filename_h = 'reference\path_loss\\rx fix, tx random\path_loss_' + str(no_link) + 'links.dat'
with open(filename_h, 'r') as f:
    rdr = csv.reader(f, delimiter='\t')
    temp = np.array(list(rdr))
    input_h_temp = np.delete(temp, -1, axis=1).astype('float64')      # remove new line
input_h_train = input_h_temp[:16000,:]


filename_h = 'reference\path_loss\\rx fix, tx random\path_loss_' + str(no_link) + 'links_test.dat'
with open(filename_h, 'r') as f:
    rdr = csv.reader(f, delimiter='\t')
    temp = np.array(list(rdr))
    input_h_temp = np.delete(temp, -1, axis=1).astype('float64')      # remove new line
input_h_test = input_h_temp[:4000,:]




#========================================================================================================
#
# 3. DNN object
#
#========================================================================================================
dnnObj = Dnn(mode_shuffle = 'disable', mode_constraint = 'disable', mode_lrshift = 'disable', batch_size=Batch_Size,
             n_epoch=num_epoch, layer_dim_list=Layer_dim_list, max_pkt_size=maximum_pkt_size,
             num_pkt=no_pkt, D_step = D_STEP, r=R, num_link=no_link, N=noise, gap_thr=Gap_thr, alpha=Alpha, delta=Delta)




#========================================================================================================
#
# 5. Train
#
#========================================================================================================
for j in range(lr.shape[0]):
    dnnObj.train_dnn(input_dr_train, input_h_train, num_pl_per_img_train, lr[j])

# Warning message: the number of training samples are not a multiple of Batch_Size
if (num_sample_train*input_h_train.shape[0]) % Batch_Size != 0:
    print('===========================================================\n')
    print('Warning: num_sample_train is not a multiple of Batch_Size!!\n')
    print('===========================================================\n')




#========================================================================================================
#
# 7. Test
#
#========================================================================================================
for j in range(lr.shape[0]):
    dnnObj.test_dnn(input_dr_test, input_h_test, num_pl_per_img_test, lr[j])



