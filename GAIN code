import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler
import openpyxl
from itertools import product
import xlrd
from xlutils.copy import copy

###set parameters###
Dim = 28
Train_No = 5000
mb_size = 200
p_hint = 0.1
gamma = 5
alpha = 10
beta = 0.5
interations= 3001
n_hidden1 = 28
n_hidden2 = 28

for j in (0.4,0.5):
    for i in (1,2,3):
        for q in range(1,31):
            # import full dataset
            tf.reset_default_graph()

            data = pd.read_excel('C:/Users/Researcher/Desktop/GAN_newsimu/hpc_data/multi_accuracy/only full data_k='
                                 + str(i) + 'misrate=' + str(j) + '.xlsx')
            data = data.as_matrix()

            # import data with missing value
            xx = pd.read_excel('C:/Users/Researcher/Desktop/GAN_newsimu/hpc_data/multi_accuracy/only missing data_k='
                               + str(i) + 'misrate=' + str(j) + '.xlsx')
            xx = xx.as_matrix()

            # about continuous and categorical
            con = np.array(
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            con = [con, ] * Train_No
            con = np.asarray(con)

            # set up mask
            mask = np.isnan(xx)
            mask = mask + 0
            mask = 1. - mask

            # normalization (mydata is to use)
            xmax = np.nanmax(xx, axis=0)

            xmin = np.nanmin(xx, axis=0)

            max_min = xmax - xmin
            max_min = max_min + 1e-8
            mydata = ((data - xmin) / max_min) * con + (1 - con) * data
            mydata = np.asarray(mydata)

            # imputation value(Z)
            meanvalue = np.nanmean(a=mydata, axis=0)
            meanZ = [meanvalue, ] * Train_No
            meanZ = np.asarray(meanZ)


            ###DEFINE function

            # Z start from noise
            def sample_Z(m, n):
                return np.random.uniform(0., 1, size=[m, n])


            # for hint
            def sample_M(m, n, p):
                A = np.random.uniform(0., 1., size=[m, n])
                B = A > p
                C = 1. * B
                return C


            # for sampling mb
            def sample_idx(m, n):
                A = np.random.permutation(m)
                idx = A[:n]
                return idx


            # for start value
            def xavier_init(size):
                in_dim = size[0]
                xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
                return tf.random_normal(shape=size, stddev=xavier_stddev)


            # for draw a picture
            def plot(samples):
                fig = plt.figure(figsize=(5, 5))

                gs = gridspec.GridSpec(5, 5)

                gs.update(wspace=0.05, hspace=0.05)

                for i, sample in enumerate(samples):
                    ax = plt.subplot(gs[i])

                    plt.axis('off')

                    ax.set_xticklabels([])

                    ax.set_yticklabels([])

                    ax.set_aspect('equal')

                    plt.imshow(sample.reshape(5, 10), cmap='Greys_r')

                return fig


            ##define write to excel in for circle
            '''
            def write_excel_xls_append(path, value):
                index = len(value)  # get the No of lines of the data to be write
                workbook = xlrd.open_workbook(path)  # open the file
                sheets = workbook.sheet_names()  # get all the sheets
                worksheet = workbook.sheet_by_name(sheets[0])  # get the first sheet
                rows_old = worksheet.nrows  # get the lines No already in the sheet
                new_workbook = copy(workbook)  # transfer xlrd to xlwt
                new_worksheet = new_workbook.get_sheet(0)  # get the first sheet of transfered sheet
                for i in range(0, index):
                    for j in range(0, len(value[i])):
                        new_worksheet.write(i + rows_old, j, value[i][j])  # write from No i+rows_old line
                new_workbook.save(path)
                print("write to excel done！")
            '''


            ##define batch normolization
            def dense(x, size, scope):
                return tf.contrib.layers.fully_connected(x, size, activation_fn=None, scope=scope)


            def dense_batch_relu(x, size, phase, scope):
                with tf.variable_scope(scope):
                    h1 = tf.contrib.layers.fully_connected(x, size, activation_fn=None, scope='dense')
                    h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=phase, scope='bn')
                    return tf.nn.tanh(h2, 'relu')


            ###BUILD STRUCTURE
            # placeholder

            with tf.name_scope('input_layer'):
                X = tf.placeholder(tf.float32, shape=[None, Dim], name='X_input')  # data vector with missing value
                M = tf.placeholder(tf.float32, shape=[None, Dim], name='Y')  # mask vector
                H = tf.placeholder(tf.float32, shape=[None, Dim], name='H')  # hint vector
                Z = tf.placeholder(tf.float32, shape=[None, Dim], name='Z')  # mean vector
                CO = tf.placeholder(tf.float32, shape=[None, Dim], name='CO')  # continuous variables vector
                phase1 = tf.placeholder(tf.bool, name='phase1')
                phase2 = tf.placeholder(tf.bool, name='phase2')
            # parameters to be calculate by the GAIN(theta_D, theta_G)
            '''
            D_W1 = tf.Variable(xavier_init([Dim * 2, n_hidden1]))  # Data and Hint are inputs
            D_b1 = tf.Variable(tf.zeros(shape=[n_hidden1]))
            D_W2 = tf.Variable(xavier_init([n_hidden1, n_hidden2]))
            D_b2 = tf.Variable(tf.zeros(shape=[n_hidden2]))
            D_W3 = tf.Variable(xavier_init([n_hidden2, Dim]))
            D_b3 = tf.Variable(tf.zeros(shape=[Dim]))
            theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

            G_W1 = tf.Variable(xavier_init([Dim * 2, n_hidden1]))  # Data and hint as inputs (mean value are in Missing Components)
            G_b1 = tf.Variable(tf.zeros(shape=[n_hidden1]))
            G_W2 = tf.Variable(xavier_init([n_hidden1, n_hidden2]))
            G_b2 = tf.Variable(tf.zeros(shape=[n_hidden2]))
            G_W3 = tf.Variable(xavier_init([n_hidden2, Dim]))
            G_b3 = tf.Variable(tf.zeros(shape=[Dim]))
            theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
            '''


            # Define generator
            def generator(x, z, m):
                inp = x * m + z * (1 - m)
                input = tf.concat(axis=1, values=[inp, m])
                # G_hidden1_1= tf.layers.dense(input, n_hidden1, name="G_hidden1")
                # G_bn1 = tf.layers.batch_normalization(G_hidden1_1, training=training1, momentum=0.9)
                # G_hidden1 = tf.nn.elu(G_bn1)

                # G_hidden2_1 = tf.layers.dense(G_hidden1, n_hidden2, name="G_hidden2")
                # G_bn2 = tf.layers.batch_normalization(G_hidden2_1, training=training1, momentum=0.9)
                # G_hidden2 = tf.nn.elu(G_bn2)

                # G_logits_before_bn = tf.layers.dense(G_hidden2, Dim, name="G_outputs"
                # G_logits = tf.layers.batch_normalization(G_logits_before_bn, training=training1,momentum=0.9)
                G_h1 = dense_batch_relu(input, n_hidden1, phase1, 'Glayer1')
                G_h2 = dense_batch_relu(G_h1, n_hidden2, phase1, 'Glayer2')
                G_logits = dense(x=G_h2, size=Dim, scope='G_logits')

                G_prob = tf.nn.sigmoid(G_logits)
                return G_prob


            def discriminator(x, m, g, h):
                inp = x * m + g * (1 - m)
                input = tf.concat(axis=1, values=[inp, h])
                # D_hidden1_1 = tf.layers.dense(input, n_hidden1, name="D_hidden1")
                # D_bn1 = tf.layers.batch_normalization(D_hidden1_1, training=training2, momentum=0.9)
                # D_hidden1 = tf.nn.elu(D_bn1)

                # D_hidden2_1 = tf.layers.dense(D_hidden1, n_hidden2, name="D_hidden2")
                # D_bn2 = tf.layers.batch_normalization(D_hidden2_1, training=training2, momentum=0.9)
                # D_hidden2 = tf.nn.elu(D_bn2)

                # D_logits_before_bn = tf.layers.dense(D_hidden2, Dim, name="D_outputs")
                # D_logit = tf.layers.batch_normalization(D_logits_before_bn, training=training2,momentum=0.9)

                D_h1 = dense_batch_relu(input, n_hidden1, phase2, 'Dlayer1')
                D_h2 = dense_batch_relu(D_h1, n_hidden2, phase2, 'D_layer2')
                D_logits = dense(x=D_h2, size=Dim, scope='D_s_logits')
                D_prob = tf.nn.sigmoid(D_logits)
                return D_prob


            # model constructure
            G_sample = generator(X, Z, M)
            D_result = discriminator(X, M, G_sample, H)

            # define loss
            with tf.name_scope('D_loss'):
                D_loss = -tf.reduce_mean(M * tf.log(tf.clip_by_value(D_result, 1e-8, 1.)) + (1 - M) * tf.log(
                    tf.clip_by_value(1. - D_result, 1e-8, 1.))) * 2  ##D_loss is sigmoid cross-entropy to tell true/fake
                tf.summary.scalar('D_loss', D_loss)
            with tf.name_scope('G_loss1'):
                G_loss1 = -tf.reduce_mean((1 - M) * tf.log(tf.clip_by_value(D_result, 1e-8, 1.))) / tf.reduce_mean(
                    1 - M)  ##G_loss1 to min the possibility that D tell the true/fake
                tf.summary.scalar('G_loss1', G_loss1)
            with tf.name_scope('MSE_train'):
                MSE_train = tf.reduce_mean((M * X * CO - M * G_sample * CO) ** 2) / tf.reduce_mean(M * CO)
                tf.summary.scalar('MSE_train', MSE_train)
            with tf.name_scope('CROSS_train'):
                CROSS_train = -tf.reduce_mean(
                    (1 - CO) * X * M * tf.log(tf.clip_by_value(G_sample, 1e-8, 1)) + (1 - X) * (1 - CO) * M * tf.log(
                        tf.clip_by_value(1 - G_sample, 1e-8, 1.)))
                tf.summary.scalar('CROSS_train', CROSS_train)
            with tf.name_scope('G_loss'):
                G_loss = gamma * G_loss1 + alpha * MSE_train + beta * CROSS_train
                tf.summary.scalar('G_loss', G_loss)

            # test performance
            MSE_test = tf.reduce_mean(((1 - M) * X * CO - (1 - M) * G_sample * CO) ** 2) / tf.reduce_mean((1 - M) * CO)
            CROSS_test = -tf.reduce_mean(
                (1 - CO) * X * (1 - M) * tf.log(tf.clip_by_value(G_sample, 1e-8, 1.)) + (1 - X) * (1 - CO) * (
                        1 - M) * tf.log(
                    tf.clip_by_value(1 - G_sample, 1e-8, 1.)))

            # solver
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                D_solver = tf.train.GradientDescentOptimizer(0.2).minimize(D_loss)
            with tf.control_dependencies(extra_update_ops):
                G_solver = tf.train.GradientDescentOptimizer(0.5).minimize(G_loss)

            ### RUN
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter("logs/", sess.graph)

            ###for calculation moving_mean and moving_variance to calculate BN
            '''
            if not os.path.exists('Dovey_exp_out1/'):
                os.makedirs('Dovey_exp_out1/')

            i = 1
            '''
            for it in tqdm(range(interations)):
                mb_idx = sample_idx(Train_No, mb_size)
                X_mb = mydata[mb_idx, :]
                M_mb = mask[mb_idx, :]
                Z_mb = meanZ[mb_idx, :]
                # Z_mb = sample_Z(mb_size, Dim)
                H_mb1 = sample_M(mb_size, Dim, 1 - p_hint)
                H_mb = M_mb * H_mb1
                CO_mb = con[mb_idx, :]
                startX = M_mb * X_mb + (1 - M_mb) * Z_mb

                ##final
                finalX = mask * mydata + (1 - mask) * meanZ
                H_final1 = sample_M(Train_No, Dim, 1 - p_hint)
                H_final = H_final1 * mask
                ###
                _, D_loss_curr = sess.run([D_solver, D_loss],
                                          feed_dict={X: X_mb, M: M_mb, Z: startX, H: H_mb, CO: CO_mb, phase1: 1,
                                                     phase2: 1})
                _, G_loss_curr, MSE_train_curr, CROSS_train_curr = sess.run(
                    [G_solver, G_loss1, MSE_train, CROSS_train],
                    feed_dict={X: X_mb, M: M_mb, Z: startX, H: H_mb, CO: CO_mb, phase1: 1, phase2: 1})
                MSE_test_curr, CROSS_test_curr = sess.run([MSE_test, CROSS_test],
                                                          feed_dict={X: X_mb, M: M_mb, Z: startX, H: H_mb, CO: CO_mb,
                                                                     phase1: 0, phase2: 0})
                MSE_final, CROSS_final = sess.run([MSE_test, CROSS_test],
                                                  feed_dict={X: mydata, M: mask, Z: finalX, CO: con, phase1: 0,
                                                             phase2: 0})
                '''if it % 500 == 0:
                    mb_idx = sample_idx(Train_No, 5)
                    X_mb = mydata[mb_idx, :]
                    M_mb = mask[mb_idx, :]
                    Z_mb = meanZ[mb_idx, :]
                    startX = M_mb * X_mb + (1 - M_mb) * Z_mb
                    samples1 = X_mb
                    samples2 = M_mb * X_mb + (1 - M_mb) * Z_mb
                    samples31 = sess.run(G_sample, feed_dict={X: X_mb, M: M_mb, Z: startX})
                    samples3 = M_mb * X_mb + (1 - M_mb) * samples31
                    samples = np.vstack([samples1, samples2, samples3])
                    fig = plot(samples)
                    plt.savefig('Dovey_exp_out1/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                    i += 1
                    plt.close(fig)
                '''
                if it % 3000 == 0:
                    print('Iteraction:{}'.format(it))
                    print('G_loss:{:.4}'.format(G_loss_curr))
                    print('D_loss:{:.4}'.format(D_loss_curr))
                    print('Train_MSE_loss: {:.4}'.format(MSE_train_curr))
                    print('Train_CROSS_loss: {:.4}'.format(CROSS_train_curr))

                    print('Test_MSE_loss: {:.4}'.format(MSE_test_curr))
                    print('Test_CROSS_loss: {:.4}'.format(CROSS_test_curr))

                    print('Final_MSE: {:.4}'.format(MSE_final))
                    print('Final_CROSS: {:.4}'.format(CROSS_final))
                    print()

                    # it_No = it
                    # output = np.array([[it_No, G_loss_curr, D_loss_curr,
                    # MSE_train_curr, CROSS_train_curr, MSE_test_curr, CROSS_test_curr, MSE_final, CROSS_final], ])
                    # excel_name = 'Hyper_para_search_new_March13.xls'
                    # write_excel_xls_append(excel_name, output)

                if it % 3000 == 0:
                    X_final = mydata
                    M_final = mask
                    Z_final = meanZ
                    CO_final = con
                    finalX = M_final * X_final + (1 - M_final) * Z_final
                    sample_final1 = sess.run(G_sample,
                                             feed_dict={X: mydata, M: mask, Z: finalX, CO: con, phase1: 0, phase2: 0})
                    sample_final2 = X_final * M_final + (1 - M_final) * sample_final1
                    sample_final = sample_final2 * max_min + xmin
                    sample_final[:, 15:22] = np.round(sample_final[:, 15:22], 0)
                    ##export
                    writer = pd.ExcelWriter('C:/Users/Researcher/Desktop/GAN_newsimu/hpc_data/multi_accuracy/gain_k='
                                            + str(i) + 'misrate=' + str(j) + '_i='+str(q)+'.xlsx')
                    sample_final_write = pd.DataFrame(sample_final)
                    # sample_final_write = sample_final_write.round({'15':0,'16':0,'17':0,'18':0,'19':0,'20':0})
                    sample_final_write.to_excel(writer, 'Sheet1')
                    writer.save()
                '''
                if it % 15000 ==0:
                    it_No = it
                    output = np.array([[mb_size,p_hint,gamma, alpha,beta, it_No,G_loss_curr, D_loss_curr,MSE_train_curr,
                                        CROSS_train_curr, MSE_test_curr,CROSS_test_curr, MSE_final, CROSS_final],])
                    excel_name = 'Hyper_para_search_weights.xls'
                    write_excel_xls_append(excel_name, output)
                '''
            print('Final_MSE: {:.4}'.format(MSE_final))
            print('Final_CROSS: {:.4}'.format(CROSS_final))

