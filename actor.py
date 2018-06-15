
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skimage.measure

from batch import Batch

class Actor():
    def __init__(self, env, local_policy, learner_policy, 
            update_interval=10, steps=32):
        self.env = env
        self.local_policy = local_policy
        self.learner_policy = learner_policy
        self.update_interval = update_interval
        self.steps = steps

        #keep track of last observation
        self.last_obs = process_state(self.env.reset())

        #sync with global learner
        self.learner_policy.recover_vars()
        self.pull_vars()
        
        print('[*] %s retrieved at step %s' % (self.learner_policy.name, 
                self.learner_policy.get_step()))
        print('[*] %s initialized at step %s' % (self.local_policy.name, 
                self.local_policy.get_step()))

    def pull_vars(self):
        self.local_policy.update_vars()

    def run(self):
        #update local policy vars on an interval for stability
        if self.learner_policy.get_step() % self.update_interval == 0:
            self.pull_vars()
        #print('learner at %s, actor at %s' % (
        #        self.learner_policy.get_step(), 
        #        self.local_policy.get_step()))

        #FIXME: last obs might be from different game
            #since games are 4.5k steps, not a big deal
        n_actions = self.env.action_space.n

        batch = Batch()
        state = self.last_obs #first action in each new env is ~random
        lstm_state = self.local_policy.lstm_init_state
        done = step = 0
        while not done and step < self.steps:
            action, value, logit, lstm_state = self.local_policy.act(state, 
                    lstm_state)
            next_state, reward, done, _ = self.env.step(action)

            #skip the specified number of frame, aggregate rewards?
                #FIXME: dont just skip, stack the frames
                    #might mess things up if predicting above
                        #using non-diff and diff frames
                        #aggregate and non aggregate
                        #must be constant
                #note the env employs frame skipping already
                    #more skipping seems to lead to a better policy though
            #for _ in range(3):
            #    if done:
            #        break
            #    next_state, reward_s, done, _ = self.env.step(action)
                #reward += reward_s 

            #process observation data
            next_state = process_state(next_state)
            if type(action) == np.int64:
                action = to_onehot(action, n_actions)

            #add experience to batch
            batch.add((state, action, reward, value, done, next_state, 
                    logit, lstm_state))

            #update
            step += 1
            state = next_state
            self.last_obs = state

        return batch.get()

    def test(self, episodes=1, render=False):

        #update local model vars
        self.pull_vars()

        stats = {'step': [], 'reward': [],}
        for episode in range(episodes):
            state = process_state(self.env.reset())
            lstm_state = self.local_policy.lstm_init_state
            done = step = reward_sum = 0
            while not done:
                action, _, _, lstm_state = self.local_policy.act(state, 
                        lstm_state, explore=False)
                state, reward, done, _ = self.env.step(action)
                if render:
                    self.env.render()
                    time.sleep(0.08)

                #process observation data
                state = process_state(state)

                #update
                step += 1
                reward_sum += reward

            stats['step'].append(step)
            stats['reward'].append(reward_sum)
            #print('[*] episode %s: %s reward' % (episode, reward_sum))

        #return pd.DataFrame(data=stats)
        return stats

def render_state(state):
    f, a = plt.subplots(nrows=1, ncols=1, figsize=(3,3))
    f.tight_layout()
    a.imshow(state, cmap='gray')
    a.set_axis_off()
    plt.show()
    plt.close(f)

def process_state(state, pad_value=0.0, normalize=True,
        downsample_type='slow', downsample_scale=3):
    #convert to standard size input (n x n matrix)
    #FIXME: should calc w once at start, not every call

    dims = len(state.shape)
    if dims == 3: #rgb input --> greyscale
        r, g, b = state[:, :, 0], state[:, :, 1], state[:, :, 2]
        state = 0.2989 * r + 0.5870 * g + 0.1140 * b
        state = state / 255.0 if normalize else state
        w = max(state.shape[0], state.shape[1])
        new_state = np.full((w,w), pad_value)
        new_state[:state.shape[0], :state.shape[1]] = state
    elif dims == 2:
        w = max(state.shape[0], state.shape[1])
        new_state = np.full((w,w), pad_value)
        new_state[:state.shape[0], :state.shape[1]] = state
    elif dims == 1:
        w = 2
        while w**2 < state.shape[0]:
            w += 1
        state = np.reshape(state, (-1, w))
        new_state = np.full((w,w), pad_value)
        new_state[:state.shape[0], :state.shape[1]] = state
    else:
        print('unsupported state size: %s' % state.shape)
        sys.exit()

    #only downsample if img input
    s = downsample_scale
    if downsample_type == 'slow' and dims > 1:
        new_state = skimage.measure.block_reduce(new_state, (s,s),
                func=np.mean)
    elif downsample_type == 'fast' and dims > 1:
        new_state = new_state[::s,::s]

    #DEBUG
    #render_state(new_state)
    #sys.exit()

    return new_state

def to_onehot(action, n_actions):
    return np.eye(n_actions)[action]

