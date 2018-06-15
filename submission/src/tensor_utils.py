
import tensorflow as tf
import numpy as np

def weight(shape):
    #xavier init
    #f = 2.0
    #in_plus_out = shape[-2] + shape[-1]
    #std = np.sqrt(f / in_plus_out)
    #w = tf.truncated_normal(shape, mean=0.0, stddev=std)
    f = 6.0
    in_plus_out = shape[-2] + shape[-1]
    std = np.sqrt(f / in_plus_out)
    w = tf.random_uniform(shape, minval=-std, maxval=std)
    return tf.Variable(w)

def bias(shape, v=0.0):
    #bias init to 0.0 in xavier paper
    return tf.Variable(tf.constant(v, shape=shape))

'''
def batch_normalize(x, activation='none'):
    #FIXME: add is_training var
    norm = tf.contrib.layer.batch_norm(x, is_training=self.)
    if activation == 'relu':
        norm = tf.nn.relu(norm)
    elif activation == 'elu':
        norm = tf.nn.elu(norm)
    return norm
'''

def residual_layer(x, y, activation='none'):
    x = tf.contrib.layers.flatten(x)
    y = tf.contrib.layers.flatten(y)
    res = tf.add(x, y)
    if activation == 'relu':
        res = tf.nn.relu(res)
    elif activation == 'elu':
        res = tf.nn.elu(res)
    return res

def lstm_layer(x, seq_len, n=256, name='lstm'):
    flat = tf.contrib.layers.flatten(x)
    seq = tf.expand_dims(flat, [0]) #fake sequence
    cell = tf.nn.rnn_cell.BasicLSTMCell(n)

    #seq_len = seq.get_shape()[1].value
    #print(seq_len)
    sizes = [cell.state_size.c, cell.state_size.h]
    init = [np.zeros((1, size), np.float32) for size in sizes]
    cell_in = [tf.placeholder(tf.float32, [None, size]) for size in sizes]
    cell_tuple = tf.nn.rnn_cell.LSTMStateTuple(cell_in[0], cell_in[1])
    out, cell_out = tf.nn.dynamic_rnn(cell, seq, 
            initial_state=cell_tuple,
            sequence_length=seq_len, 
            time_major=False, 
            scope=name,
            dtype=tf.float32)
    cell_out = [v[:1, :] for v in cell_out]
    return tf.reshape(out, [-1, n]), cell_in, cell_out, init

def conv_layer(x, filter_size=(3,3), out_channels=32, 
        activation='none', stride=(1,1)):
    #reshape for observations, greyscale images
    if len(x.get_shape()) is 3: #for non img input
        x = tf.reshape(x, (-1,) + (x.get_shape()[-2], x.get_shape()[-1]) 
                + (1,))
    #filter shape = [height, width, in_channels, out_channels]
    filter_shape = filter_size + (x.get_shape()[-1].value, 
            out_channels)
    w_conv = weight(filter_shape)
    b_conv = bias([out_channels])
    conv = tf.nn.conv2d(x, w_conv, 
            strides=(1,) + stride + (1,), padding='SAME') + b_conv
    if activation == 'relu':
        conv = tf.nn.relu(conv)
    elif activation == 'elu':
        conv = tf.nn.elu(conv)
    elif activation == 'lrelu':
        conv = tf.nn.leaky_relu(conv)
    return conv

def dense_layer(x, n=512, activation='none', keep_prob=None):
    #create 1d input
    flat = tf.contrib.layers.flatten(x)
    #init vars
    w = weight([flat.get_shape()[-1].value, n])
    b = bias([n])
    #fully connected layer
    dense = tf.matmul(flat, w) + b
    #activation
    if activation == 'relu':
        dense = tf.nn.relu(dense)
    elif activation == 'elu':
        dense = tf.nn.elu(dense)
    elif activation == 'lrelu':
        dense = tf.nn.leaky_relu(dense)
    #dropout
    if not keep_prob == None:
        dense = tf.nn.dropout(dense, keep_prob)

    return dense

def minimize(x, learn_rate, clip=None):
    step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learn_rate)
    if clip:
        g, v = zip(*optimizer.compute_gradients(x))
        g, _ = tf.clip_by_global_norm(g, clip)
        op = optimizer.apply_gradients(zip(g, v), global_step=step)
    else:
        op = optimizer.minimize(x, global_step=step)
    return op, step

'''
def universe_model(x):
    for _ in range(4):
        x = conv_layer(x, (3,3), 32, 'elu', (2,2))
    return lstm_layer(x)
'''

def conv_residual_model(x):
    x = conv_layer(x, (3,3), 32)
    a = conv_layer(x, (1,1), 32)
    a = conv_layer(a, (3,3), 32)
    x = residual_layer(x, a, 'relu')
    return dense_layer(x, 512, 'relu', drop=True)

def conv_lstm_model(x):
    x = conv_layer(x, (3,3), 32, 'relu')
    #x = conv_layer(x, (3,3), 32, 'relu')
    x = lstm_layer(x)
    return dense_layer(x, 512, 'relu', drop=True)

def small_conv(x):
    x = conv_layer(x, (3,3), 32)
    return dense_layer(x, 256, 'relu', drop=True)

def linear_model(x, n=256, activation='relu'):
    return dense_layer(x, n, activation, drop=True)










