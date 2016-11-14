__author__ = 'schein'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os,sys


def build_volume_model(path_to_vol_models):
    train_data = None
    test_data = None
    train_labels = None
    test_labels = None

    label_map = {}
    model_index = {}
    volume_data = {}
    img_idx =0
    num_classes = 10
    base_dir = os.path.join(path_to_vol_models)
    for root_dir,dirs,files in os.walk(base_dir):
        for fname in files:
            #print(fname)
            f,ext = os.path.splitext(fname)
            f,mtype = os.path.split(root_dir)
            f,ctype = os.path.split(f)
            if ctype not in label_map:
                label_map[ctype] = len(label_map)

            if ext == '.npy':
                img_idx=img_idx+1
                if img_idx % 100 == 0:
                    print('read {} images'.format(img_idx))

                model = np.load(os.path.join(root_dir,fname))
                if mtype == 'train':
                    if train_data == None:
                        train_data = np.stack([model],axis=0)
                        d = np.zeros(shape=(num_classes))
                        d[label_map[ctype]] = 1
                        train_labels = [d]
                        #model_index[0] = os.path.join(root_dir,fname)
                    else:
                        train_data = np.concatenate((train_data,np.stack([model],axis=0)),axis=0)
                        d = np.zeros(shape=(num_classes))
                        d[label_map[ctype]] = 1
                        train_labels = np.concatenate((train_labels,[d]),axis=0)
                        #model_index[len(train_data)] = os.path.join(root_dir,fname)
                if mtype == 'test':
                    if test_data == None:
                        test_data = np.stack([model],axis=0)
                        d = np.zeros(shape=(num_classes))
                        d[label_map[ctype]] = 1
                        test_labels = [d]
                    else:
                        test_data = np.concatenate((test_data,np.stack([model],axis=0)),axis=0)
                        d = np.zeros(shape=(num_classes))
                        d[label_map[ctype]] = 1
                        test_labels = np.concatenate((test_labels,[d]),axis=0)
                #print(root_dir,fname,ctype,mtype)
                #if test_data != None and train_data != None: print(test_data.shape,train_data.shape)
            #break
    volume_data['test_data'] = test_data
    volume_data['train_data'] = train_data
    volume_data['train_labels'] = train_labels
    volume_data['test_labels'] = test_labels
    print("train data shape ",volume_data['train_data'].shape)
    print("train labels shape ",volume_data['train_labels'].shape)
    print("test data shape ",volume_data['test_data'].shape)
    print("test labels shape ",volume_data['test_labels'].shape)
    return volume_data

def train_logistic_model(volume_data):
    if len(volume_data['train_data']) < 1:
        print("No training data found")
        return

    vol_dim = volume_data['train_data'][0].shape
    #placeholders for input and output
    x = tf.placeholder(tf.float32, shape=[None, vol_dim[0]*vol_dim[1]*vol_dim[2]])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    #Weight and bias
    W = tf.Variable(tf.zeros([vol_dim[0]*vol_dim[1]*vol_dim[2],10]))
    b = tf.Variable(tf.zeros([10]))

    init = tf.initialize_all_variables()

    y = tf.nn.softmax(tf.matmul(x,W) + b)


    #cost function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


    sess = tf.Session()
    #sess = tf.InteractiveSession()
    sess.run(init)
    mini_batch_size = 200

    for i in range(10000):
        if i%1000 == 0:
            print("train iteration {}".format(i))
        idx = np.random.randint(0,len(volume_data['train_data']),size=mini_batch_size)

        batch_xs = volume_data['train_data'][idx].reshape(mini_batch_size,vol_dim[0]*vol_dim[1]*vol_dim[2])
        batch_ys = volume_data['train_labels'][idx]

        #print(type(batch_xs),batch_xs.shape)
        #print(type(batch_ys),batch_ys)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        #break

    #check prediction
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    test_x = volume_data['test_data'][:].reshape(len(volume_data['test_data']),vol_dim[0]*vol_dim[1]*vol_dim[2])
    test_y = volume_data['test_labels'][:]
    print("test accuracy {}".format(sess.run(accuracy, feed_dict={x:test_x , y_:test_y })))
    sess.close()


def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1,1, 1], padding='SAME')

def max_pool_2x2x2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2,2, 1],
                        strides=[1, 2, 2,2, 1], padding='SAME')
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def train_cnn_model(volume_data):

    if len(volume_data['train_data']) < 1:
        print("No training data found")
        return

    vol_dim = volume_data['train_data'][0].shape

    sess = tf.Session()

    x = tf.placeholder(tf.float32, shape=[None, vol_dim[0]*vol_dim[1]*vol_dim[2]])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    W_conv1 = weight_variable([5, 5, 5, 1, 128])
    b_conv1 = bias_variable([128])
    x_image = tf.reshape(x, [-1,16,16,16,1])## set up the input as a volume data cube with one color channel


    h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2x2(h_conv1)


    W_conv2 = weight_variable([5, 5, 5, 128, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2x2(h_conv2)


    W_fc1 = weight_variable([4 * 4 *4*64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 4*4*4*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)



    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.initialize_all_variables())
    mini_batch_size = 60
    error= []

    for i in range(1000):

        idx = np.random.randint(0,len(volume_data['train_data']),size=mini_batch_size)
        batch_xs = volume_data['train_data'][idx].reshape(mini_batch_size,vol_dim[0]*vol_dim[1]*vol_dim[2])
        batch_ys = volume_data['train_labels'][idx]
        if i%20 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_: batch_ys, keep_prob: 1.0},session=sess)
            print("step {}, training accuracy {}".format(i, train_accuracy))
            error.append(train_accuracy)


        train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5},session=sess)
    test_x = volume_data['test_data'][:].reshape(len(volume_data['test_data']),vol_dim[0]*vol_dim[1]*vol_dim[2])
    test_y = volume_data['test_labels'][:]
    print("test accuracy {}".format(accuracy.eval(feed_dict={x: test_x, y_: test_y, keep_prob: 1.0},session=sess)))



    sess.close()

    fig = plt.figure(0)
    plt.plot(error,'.r')
    plt.show()


def main():
    print("build models")
    volume_data = build_volume_model('/home/schein/3ddata/3dmodels/Voxel')
    #print('train logistic model')
    #train_logistic_model(volume_data)
    print("train cnn model")
    train_cnn_model(volume_data)

if __name__ == '__main__':
    main()
