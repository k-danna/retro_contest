#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True

import tensorflow as tf
import numpy as np
import gym_remote.exceptions as gre
import gym_remote.client as grc

from policy import Policy

#taken from https://github.com/scikit-image/scikit-image/blob/master/skimage/util/shape.py
def view_as_blocks(arr_in, block_shape):
    block_shape = np.array(block_shape)
    arr_shape = np.array(arr_in.shape)
    arr_in = np.ascontiguousarray(arr_in)

    new_shape = tuple(arr_shape // block_shape) + tuple(block_shape)
    new_strides = tuple(arr_in.strides * block_shape) + arr_in.strides

    arr_out = np.lib.stride_tricks.as_strided(arr_in, shape=new_shape, strides=new_strides)

    return arr_out

#taken from https://github.com/scikit-image/scikit-image/blob/master/skimage/measure/block.py
def block_reduce(image, block_size, func=np.sum, cval=0):
    pad_width = []
    for i in range(len(block_size)):
        if image.shape[i] % block_size[i] != 0:
            after_width = block_size[i] - (image.shape[i] % block_size[i])
        else:
            after_width = 0
        pad_width.append((0, after_width))

    image = np.pad(image, pad_width=pad_width, mode='constant',
            constant_values=cval)

    out = view_as_blocks(image, block_size)

    for i in range(len(out.shape) // 2):
        out = func(out, axis=-1)

    return out

def process_state(state, pad_value=0.0, normalize=True,
        downsample_type='slow', downsample_scale=3):
    #convert to standard size input (n x n matrix)
    #FIXME: should calc w once at start, not every call

    r, g, b = state[:, :, 0], state[:, :, 1], state[:, :, 2]
    state = 0.2989 * r + 0.5870 * g + 0.1140 * b
    state = state / 255.0 if normalize else state
    w = max(state.shape[0], state.shape[1])
    new_state = np.full((w,w), pad_value)
    new_state[:state.shape[0], :state.shape[1]] = state

    #only downsample if img input
    s = downsample_scale
    if downsample_type == 'slow':
        #removes skimage dependency
        new_state = block_reduce(new_state, (s,s), func=np.mean)
        #new_state = skimage.measure.block_reduce(new_state, (s,s),
        #        func=np.mean)

    elif downsample_type == 'fast':
        new_state = new_state[::s,::s]

    return new_state

def main():
    
    #remote env
    env = grc.RemoteEnv('tmp/sock')
    
    #FIXME: DEBUG
    #import retro
    #env = retro.make(game='SonictheHedgehog-Genesis', state='GreenHillZone.Act1')

    #load the policy
    name = 'learner_global'
    state = process_state(env.reset())
    test_policy = Policy(state.shape, env.action_space.n, name, 
            act_int=False, recover=True, sess=tf.Session(), pull_scope=name)

    #run the env
    lstm_state = test_policy.lstm_init_state
    while True:
        action, _, _, lstm_state = test_policy.act(state, lstm_state, 
                explore=False)
        state, reward, done, _ = env.step(action)
        state = process_state(state)
        if done:
            env.reset()

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as e:
        print('exception', e)

