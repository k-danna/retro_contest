
import os, time
import tensorflow as tf
import numpy as np

from tensor_utils import *

class Policy():
    def __init__(self, state_shape, n_actions, name, recover=False, 
            act_int=False, out_dir='logs', sess=None, 
            pull_scope='learner_global'):

        tf.set_random_seed(42)
        np.random.seed(42)
        self.act_int = act_int #return an int instead of hot vector action
        self.out_dir = out_dir
        self.name = name

        #ensure separate vars for each worker, learner
        with tf.name_scope(name):

            with tf.name_scope('input'):
                self.state_in = tf.placeholder(tf.float32, 
                        (None,) + state_shape, name='state_in')
                self.action_in = tf.placeholder(tf.float32, 
                        [None, n_actions], name='action_in')
                self.reward_in = tf.placeholder(tf.float32, 
                        [None], name='reward_in')
                self.value_in = tf.placeholder(tf.float32, 
                        [None], name='value_in')
                self.advantage_in = tf.placeholder(tf.float32, 
                        [None], name='advantage_in')
                self.nextstate_in = tf.placeholder(tf.float32, 
                        (None,) + state_shape, name='nextstate_in')
                self.logit_in = tf.placeholder(tf.float32, 
                        [None, n_actions], name='logit_in')
                self.keep_prob = tf.placeholder(tf.float32, 
                        name='keep_prob_in')

            with tf.name_scope('model'):
                x = self.state_in

                #net used in impala paper https://arxiv.org/pdf/1802.01561.pdf
                    #paper used relu
                x = conv_layer(x, (8,8), 16, 'elu', (4,4))
                x = conv_layer(x, (4,4), 32, 'elu', (2,2))
                x = dense_layer(x, 256, 'elu', self.keep_prob)
                x, self.lstm_in, self.lstm_out, self.lstm_init_state = lstm_layer(x, tf.shape(self.state_in)[:1], 256, name)

            with tf.name_scope('policy'):
                self.logits_class = dense_layer(x, n_actions)
                # subtract max --> new max=0 for stability
                shifted_logits = self.logits_class - tf.reduce_max(
                    self.logits_class, [1], keepdims=True) 

            with tf.name_scope('value'):
                logit_val = dense_layer(x, 1)
                self.value = tf.reduce_sum(logit_val, axis=1)

            with tf.name_scope('loss'):
                
                #log softmax
                policy = shifted_logits - tf.log(tf.reduce_sum(
                        tf.exp(shifted_logits), [1], keepdims=True))

                #estimate by avging over simultaneous actions
                probs_act = tf.reduce_sum(
                        policy * self.action_in, [1])
                probs_act = probs_act / (1e-8 + tf.reduce_sum(
                        self.action_in, [1]))

                #policy, value loss
                policy_loss = - tf.reduce_sum(probs_act * self.advantage_in)
                value_loss = 0.5 * tf.reduce_sum(tf.square(
                        self.value - self.reward_in))
                
                #entropy bonus to prevent early convergence
                    # -sum(policy * log(policy))
                probs_class = tf.nn.softmax(shifted_logits) #+ 1e-8
                self.policy = probs_class
                logprobs_class = tf.nn.log_softmax(shifted_logits) #+ 1e-8
                entropy = - tf.reduce_sum(probs_class * logprobs_class)
                self.loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
            with tf.name_scope('optimize'):
                self.optimize, self.step = minimize(self.loss, 
                        6e-4, 40.0) #hyper from impala paper

            with tf.name_scope('update_vars'):
                model_vars = self.get_vars()
                new_vars = tf.global_variables(pull_scope)
                #assign recovered variables over the initialized ones
                self.pull = [model_vars[i].assign(new_vars[i]) 
                        for i in range(len(model_vars))]

        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

        #load weights from disk if specified
        if recover:
            self.recover_vars()

    def act(self, state, lstm_state, explore=True):
        value, logit, lstm_c, lstm_h = self.sess.run([self.value, 
                self.policy] + self.lstm_out,
                feed_dict={
                    self.state_in: [state],
                    self.lstm_in[0]: lstm_state[0],
                    self.lstm_in[1]: lstm_state[1],
                    self.keep_prob: 1.0,
                })
        value = value[0]
        logit = logit[0]

        def softmax(arr):
            return np.exp(arr) / np.exp(arr).sum()

        #choose top k actions
        k = 3 #np.random.choice(k) + 1
        #k = k if not self.act_int else 1
        if explore:
            top_k = np.random.choice(len(logit), k, p=logit, 
                    replace=False)
        else:
            top_k = np.argpartition(logit, -k)[-k:]
        action = np.zeros(logit.shape, dtype=np.int32)
        action[top_k] = 1

        #zero out actions not taken
        mask = np.ones(logit.shape, dtype=np.bool)
        mask[top_k] = False #ignore false values
        logit[mask] = 0.0

        if self.act_int:
            action = np.argmax(action)

        return action, value, logit, (lstm_c, lstm_h)

    def get_vars(self):
        #FIXME: add step to trainable variables
        #return tf.trainable_variables(self.name)
        return tf.global_variables(self.name)

    def recover_vars(self):
        try:
            #load sess, trained variables
            saver = tf.train.Saver()
            saver.restore(self.sess, tf.train.latest_checkpoint(
                    os.path.join(self.out_dir, 'model')))
        except:
            print('[-] could not load model')
            return
        print('%s step: %s' % (self.name, self.sess.run([self.step])))

    def update_vars(self):
        self.sess.run([self.pull])
        #print('%s step: %s' % (self.name, self.sess.run([self.step])))
















