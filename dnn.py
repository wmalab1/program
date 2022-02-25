#========================================================================================================
#
# dnn.py
#
#========================================================================================================

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle



def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator



#========================================================================================================
#
# DNN class
#
#========================================================================================================
class Dnn:
    def __init__(self, mode_shuffle = 'disable', mode_constraint = 'disable', mode_lrshift = 'disable', batch_size=100, n_epoch=200, layer_dim_list = [16, 896],
                 max_pkt_size=2**11, num_pkt=128, D_step = 2**12, r=np.array([1,2,3,4]), num_link=2, N=0, gap_thr=10, alpha=0, delta=1):

        self.mode_shuffle = mode_shuffle
        self.mode_constraint = mode_constraint
        self.mode_lrshift = mode_lrshift

        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.layer_dim_list = layer_dim_list

        self.max_pkt_size = max_pkt_size
        self.num_pkt = num_pkt

        self.D_step = D_step

        self.num_link = num_link
        self.N = N

        self.R = r

        self.num_R = r.shape[1]

        self.R_max = np.max(r)

        self.gap_thr = gap_thr
        self.alpha = alpha
        self.delta = delta

        self.weights = []
        self.biases = []

        self.max_dr_log = 0
        self.min_dr_log = 0

        self.avg_h_log = 0
        self.std_h_log = 0


    # --------------------------------------------------------------------------------------------------------
    # train_dnn
    # --------------------------------------------------------------------------------------------------------
    def train_dnn(self, input_dr, input_h, num_pl_per_img, lr):
        # --------------------------------------------------------------------------------
        # 1. Settings
        # --------------------------------------------------------------------------------
        weights = []
        biases = []

        # the number of distortion-rate pair
        dr_pair = int(input_dr.shape[0]/2)

        num_batch = int(dr_pair / self.batch_size)

        seed_weight = 1000
        seed_shuffle = 2000
        np.random.seed(seed_shuffle)



        # --------------------------------------------------------------------------------
        # 2. Normalize train data
        # --------------------------------------------------------------------------------
        # ------------------------ normalize distortion ------------------------
        input_dr_log = np.log10(input_dr)

        self.max_dr_log = np.max(input_dr_log)
        self.min_dr_log = np.min(input_dr_log)

        temp = (input_dr_log - self.min_dr_log) / (self.max_dr_log - self.min_dr_log)
        input_dr = 2 * temp - 1

        input_dr_link1 = input_dr[:dr_pair]
        input_dr_link2 = input_dr[dr_pair:]

        # ------------------------ normalize path loss ------------------------
        input_h_log = np.log10(input_h)

        self.avg_h_log = np.mean(input_h_log)
        self.std_h_log = np.std(input_h_log)

        input_h = (input_h_log - self.avg_h_log) / self.std_h_log



        # --------------------------------------------------------------------------------
        # 3. Build Model(graph of tensors)
        # --------------------------------------------------------------------------------
        tf.reset_default_graph()

        # 1) placeholder
        x_ph = tf.placeholder(tf.float64, shape=[None,input_dr.shape[1]*self.num_link+self.num_link**2])

        # 2) neural network structure
        for i in range(len(self.layer_dim_list)):
            if i == 0:
                in_layer = x_ph
                in_dim = input_dr.shape[1]*self.num_link+self.num_link**2
                out_dim = self.layer_dim_list[i]
            else:
                in_layer = out_layer
                in_dim = self.layer_dim_list[i-1]
                out_dim = self.layer_dim_list[i]

            weight = tf.Variable(tf.random_normal([in_dim, out_dim], stddev=tf.sqrt(2.0 / tf.cast(in_dim, tf.float64)), seed=seed_weight * (i * i + 1), dtype=tf.float64), dtype=tf.float64)
            bias = tf.Variable(tf.zeros(out_dim, dtype=tf.float64), dtype=tf.float64)

            mult = tf.matmul(in_layer, weight) + bias

            # activation function
            if i < len(self.layer_dim_list)-1:  # hidden layer
                out_layer = tf.nn.relu(mult)

            else:   # output layer
                output = tf.nn.sigmoid(mult)

            weights.append(weight)
            biases.append(bias)

        # ---------------------------- output R ----------------------------
        # output R quantize
        # (e.g. R = 0~4 --> R_quant = [1.000015, 2.00001, 3.00001, 4.000005])
        output_R = self.R_max * output[:,:-self.num_link] # output: 0~1,  output_R: 0~4

        # quasi-quantizer
        slope = 1e-5
        R_quant_temp = tf.where(tf.less(output_R, 1.5), 1 + (output_R * slope),
                           tf.where(tf.less(output_R, 2.5), 2 + (output_R * slope) - 1.5 * slope,
                                    tf.where(tf.less(output_R, 3.5), 3 + (output_R * slope) - 2.5 * slope,
                                             4 + (output_R * slope) - 3.5 * slope)))

        R_quant_link1 = R_quant_temp[:,:self.num_pkt]
        R_quant_link2 = R_quant_temp[:,self.num_pkt:]

        R_quant = tf.stack([R_quant_link1,R_quant_link2])

        # ---------------------------- output power ----------------------------
        output_pw = output[:,-self.num_link:] + 1e-100

        # 3) loss function
        # --------------------- input_dr: log scale to linear scale ---------------------
        temp = (x_ph[:,:-(self.num_link**2)] + 1) / 2
        temp = temp * (self.max_dr_log - self.min_dr_log) + self.min_dr_log
        D = pow(10, temp) # log scale -> linear scale

        D = tf.reshape(D,[self.num_link, self.batch_size, -1])

        # --------------------- input_h: log scale to linear scale ---------------------
        H = tf.reshape(x_ph[:,-(self.num_link**2):], [-1, self.num_link, self.num_link])
        temp = H * self.std_h_log + self.avg_h_log
        H = pow(10, temp)  # log scale -> linear scale

        # -------------------------- SINR --------------------------
        SINR = []

        for i in range(self.num_link):
            den = self.N # noise power
            for j in range(self.num_link):
                if j != i:
                    den += output_pw[:,j]*H[:,i,j] # interference power
            SINR_per_link = output_pw[:,i] * H[:,i,i] / den

            SINR.append(SINR_per_link)

        SINR = tf.expand_dims(SINR, -1)

        # --------------------- outage probability ---------------------
        Pout = 1-tf.exp(-(2**R_quant-1)/SINR)

        link_idx = np.arange(0, self.num_link, 1).reshape(-1,1)
        link_idx = np.tile(link_idx[:,np.newaxis,:], (1,self.batch_size,1)) # np.tile(A, (n,m)): repeat A for (n,m)

        sample_idx = np.arange(0, self.batch_size, 1).reshape([self.batch_size, 1])
        sample_idx = np.tile(sample_idx, (self.num_link,1,1))

        # --------------------- expected distortion ---------------------
        E_D = tf.zeros([self.num_link, self.batch_size, 1], dtype=tf.float64)

        for success_pkt in range(self.num_pkt, 0, -1):
            # The number of source bits for the successfully received packets of each link
            total_bits = self.max_pkt_size * (tf.reduce_sum(R_quant[:, :, 0:success_pkt], -1, keep_dims=True) / self.R_max)

            # distortion interpolation
            D_idx1 = tf.cast(total_bits / self.D_step, tf.int32)
            D_idx2 = D_idx1 + 1

            small_v1 = tf.cast(D_idx1 * self.D_step, tf.float64)
            big_v2 = tf.cast(D_idx2 * self.D_step, tf.float64)

            w1 = (big_v2 - total_bits) / (big_v2 - small_v1)
            w2 = (total_bits - small_v1) / (big_v2 - small_v1)

            idx1 = tf.concat([link_idx, sample_idx, D_idx1], -1)
            idx2 = tf.concat([link_idx, sample_idx, D_idx2], -1)

            if success_pkt == self.num_pkt:
                Distortion = []
                for i in range(self.num_link):
                    for j in range(self.batch_size):
                        Distortion_temp = tf.cond(total_bits[i,j, 0] < self.max_pkt_size * self.num_pkt,
                                                  lambda: w1[i,j, 0] * tf.gather_nd(D[i,j, :], [D_idx1[i,j, 0]]) + w2[i,j, 0] * tf.gather_nd(D[i,j, :], [D_idx2[i,j, 0]]),
                                                  lambda: tf.gather_nd(D[i,j, :], [D_idx1[i,j, 0]]))
                        Distortion.append(Distortion_temp)
                value = tf.reshape(Distortion, [self.num_link, self.batch_size, 1])
            else:
                Distortion = w1 * tf.reshape(tf.gather_nd(D, idx1), [self.num_link, self.batch_size, 1])\
                             + w2 * tf.reshape(tf.gather_nd(D, idx2), [self.num_link, self.batch_size, 1])
                value = Distortion * tf.reshape(Pout[:,:, success_pkt], [self.num_link, self.batch_size, 1])

            E_D = (E_D + value) * (1 - tf.reshape(Pout[:,:, success_pkt - 1], [self.num_link, self.batch_size, 1]))

        E_D = E_D + tf.reshape(D[:, :, 0], [self.num_link, self.batch_size, 1]) * tf.reshape(Pout[:,:, 0], [self.num_link, self.batch_size, 1])


        # -------------------------- PSNR --------------------------
        psnr = 10 * log10(255**2/E_D)

        # -------------------------- loss --------------------------
        # 3-5) constraint O,X
        if self.mode_constraint == 'disable':
            # --------------------- no constraint ---------------------
            loss = -tf.reduce_mean(psnr)

        elif self.mode_constraint == 'enable':
            # --------------------- constraint ---------------------
            psnr_max_link1 = 10 * log10(255**2/D[0,:,-1])
            psnr_max_link2 = 10 * log10(255**2/D[1,:,-1])

            loss = -1 * (1-self.alpha) * tf.reduce_mean(psnr) \
                   + self.alpha * tf.nn.tanh(tf.maximum(abs((tf.reduce_mean(psnr[0,:]-psnr_max_link1))-(tf.reduce_mean(psnr[1,:]-psnr_max_link2)))-self.gap_thr,0) / self.delta)


        # 4) learning rate scheduling O,X
        if self.mode_lrshift == 'disable':
            # --------------------- no lr schedule ---------------------
            learning_rate = lr
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train = optimizer.minimize(loss)

        elif self.mode_lrshift == 'enable':
            # --------------------- lr schedule ---------------------
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = lr
            lr_shift_period = self.n_epoch * num_batch / 2
            lr_shift_rate = 3/10
            # lr_shift_rate = 1/2

            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, lr_shift_period, lr_shift_rate, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)  # lr: learning_rate
            train = optimizer.minimize(loss, global_step=global_step)


        # 5) initialization
        init = tf.global_variables_initializer()  # 위에서 선언한 variable들을 init에 연결함
        sess = tf.Session()
        sess.run(init)



        # --------------------------------------------------------------------------------
        # 4. train
        # --------------------------------------------------------------------------------
        ##########################       epoch       ##########################
        for e in range(self.n_epoch):

            # shuffle
            if (self.mode_shuffle == 'enable'):
                input_shuffle_dr_link1 = shuffle(input_dr_link1, random_state=seed_shuffle*e)
                input_shuffle_dr_link2 = shuffle(input_dr_link2, random_state=seed_shuffle*e)
                input_shuffle_h = shuffle(input_h, random_state=seed_shuffle*e)
            else:
                input_shuffle_dr_link1 = input_dr_link1
                input_shuffle_dr_link2 = input_dr_link2
                input_shuffle_h = input_h

            ##########################       batch       ##########################
            for j in range(num_batch):

                input_batch_link1 = input_shuffle_dr_link1[j * self.batch_size: (j + 1) * self.batch_size]
                input_batch_link2 = input_shuffle_dr_link2[j * self.batch_size: (j + 1) * self.batch_size]

                input_h_idx = np.random.choice(input_h.shape[0], num_pl_per_img)
                input_h_random = input_shuffle_h[input_h_idx,:]

                ##########################       h (path loss)       ##########################
                for h in range(input_h_random.shape[0]):
                    h_vec = np.tile(input_h_random[h,:], (self.batch_size, 1))
                    input_batch = np.concatenate((input_batch_link1, input_batch_link2, h_vec), axis=1)

                    # Back propagation
                    sess.run(train, feed_dict={x_ph: input_batch})

        # End of "epoch loop"



        # --------------------------------------------------------------------------------
        # 5. DNN weights, biases
        # --------------------------------------------------------------------------------
        self.weights, self.biases = sess.run([weights, biases])

        sess.close()




    # --------------------------------------------------------------------------------------------------------
    # test_dnn
    # --------------------------------------------------------------------------------------------------------
    def test_dnn(self, input_dr, input_h, num_pl_per_img, lr):
        # --------------------------------------------------------------------------------
        # 1. Settings
        # --------------------------------------------------------------------------------
        dr_pair = int(input_dr.shape[0]/2)


        # --------------------------------------------------------------------------------
        # 2. Normalize the test data
        # --------------------------------------------------------------------------------
        # ------------------------ normalize distortion ------------------------
        input_dr_log = np.log10(input_dr)

        temp = (input_dr_log - self.min_dr_log) / (self.max_dr_log - self.min_dr_log)
        input_dr = 2 * temp - 1

        input_dr_link1 = input_dr[:dr_pair]
        input_dr_link2 = input_dr[dr_pair:]

        # ------------------------ normalize path loss ------------------------
        input_h_log = np.log10(input_h)

        input_h = (input_h_log - self.avg_h_log) / self.std_h_log


        # --------------------------------------------------------------------------------
        # 3. Build Model (graph of tensors)
        # --------------------------------------------------------------------------------
        tf.reset_default_graph()

        # 1) placeholder
        x_ph = tf.placeholder(tf.float64, shape=[None, input_dr.shape[1]*self.num_link+self.num_link**2])

        # 2) neural network structure
        for i in range(len(self.layer_dim_list)):
            if i == 0:
                in_layer = x_ph
            else:
                in_layer = out_layer

            weight_layer = tf.convert_to_tensor(self.weights[i][:, :])
            bias_layer = tf.convert_to_tensor(self.biases[i][:])

            mult = tf.matmul(in_layer, weight_layer) + bias_layer

            if i < len(self.layer_dim_list) - 1:    # hidden layer
                out_layer = tf.nn.relu(mult)
            else:   # output layer
                output = tf.nn.sigmoid(mult)

        # ---------------------------- output R ----------------------------
        # output quantize (0~4 -> 1,2,3,4)
        output_R = self.R_max * output[:,:-self.num_link]
        R_int = tf.round(output_R)
        R_int = tf.where(tf.equal(R_int, 0), 1 + R_int * 0, R_int)

        R_int_link1 = R_int[:,:self.num_pkt]
        R_int_link2 = R_int[:,self.num_pkt:]

        # ---------------------------- output power ----------------------------
        output_pw = output[:,-self.num_link:] + 1e-100

        # 3) initialization
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)


        # --------------------------------------------------------------------------------
        # 4. DNN outputs session run
        # --------------------------------------------------------------------------------
        dnn_R_link1_per_sample = np.zeros([num_pl_per_img, dr_pair, self.num_pkt])
        dnn_R_link2_per_sample = np.zeros([num_pl_per_img, dr_pair, self.num_pkt])
        dnn_power_per_sample = np.zeros([num_pl_per_img, dr_pair, self.num_link])

        ##########################       h (path loss)       ##########################
        for h in range(num_pl_per_img):
            # DNN inputs: D-R curve, path loss
            h_vec = input_h[h::num_pl_per_img,:]
            input = np.concatenate((input_dr_link1, input_dr_link2, h_vec), axis=1)

            # DNN outputs: spectral efficiencies, transmit powers
            dnn_R_link1_per_sample[h] = sess.run(R_int_link1, feed_dict={x_ph: input})
            dnn_R_link2_per_sample[h] = sess.run(R_int_link2, feed_dict={x_ph: input})
            dnn_power_per_sample[h] = sess.run(output_pw, feed_dict={x_ph: input})

        sess.close()






