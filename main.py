#!/usr/bin/env python3

import sys, os, time
sys.dont_write_bytecode = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from datetime import datetime, timedelta
import tensorflow as tf
import numpy as np
import pandas as pd
import gym
import retro
import retro_contest.local

from policy import Policy
from actor import Actor, process_state

def main():
    
    #train/test vars
    recover=True
    train_steps = 10000 * 12 #batches per hour * hours
    train_time = 0 #seconds
    test_render = False
    test_episodes = 3
    print_interval = 10000 #FIXME saves on this interval also
    n_actors = 1

    gym_envs = False
    #output an int action if true, else one hot action
    act_int = True if gym_envs else False

    #init env
    #def get_env(seed=42):
    #    if gym_envs:
    #        env_name = ['CartPole-v0', 'Acrobot-v1'][0]
    #        env = gym.make(env_name)
    #        env.seed(seed)
    #        return env
    #    else:
    #        env_name = 'SonicTheHedgehog-Genesis'
    #        env_state = 'GreenHillZone.Act1'
    #        env = retro_contest.local.make(game=env_name, state=env_state)
    #        env.seed(seed)
    #        return env

    #get some info for the policy, init the emulator wrapper
    env = Env_Controller('./sonic-train.csv', 
            init=('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1'))
    state = env.reset()
    n_actions = env.action_space.n
    state_shape = process_state(state).shape

    #init the session
    with tf.Session() as sess:

        #global learner
        learner_name = 'learner_global'
        learner_policy = Policy(state_shape, n_actions, learner_name, 
                act_int=act_int, recover=True, sess=sess)

        actors = []
        for i in range(n_actors):
            #FIXME: for now just stay synchronous
                #only one emulator allowed per thread
            #env = get_env(42 + i)
            actor_policy = Policy(state_shape, n_actions, 
                    'actor_%s' % i, act_int=act_int, sess=sess,
                    pull_scope=learner_name)
            actor = Actor(env, actor_policy, learner_policy)
            actors.append(actor)

        #FIXME: new thread for each actor
        end_time = datetime.now() + timedelta(seconds=train_time)
        print('[*] training for %ss with %s actors (ending at %s)' % (
                train_time, len(actors), '{:%H:%M:%S}'.format(end_time)))
        start_time = time.time()
        start_step = learner_policy.get_step()
        #while(train_time > (time.time() - start_time)):
        while(True):
            for actor in actors:
                batch = actor.run()
                learner_policy.train(batch)
            step = learner_policy.get_step()
            if step % print_interval == 0:
                learner_policy.save()
                print('[*] learner step: %s' % step)
                info = actor.test(episodes=3, render=False)
                print(pd.DataFrame.from_dict(data=info
                        ).describe().loc[['min', 'max', 'mean', 'std']])

            #FIXME: debug limit train steps
            if (step - start_step) >= train_steps:
                break

        steps_per_s = float(learner_policy.get_step() - start_step) / float(
                time.time() - start_time)
        steps_per_min = round(steps_per_s * 60, 3)
        steps_per_sec = round(steps_per_s, 3)
        print('[+] done training: %s steps/min (%s /s)' % (steps_per_min, 
                steps_per_sec))
        learner_policy.save()

        #print('learner at: %s' % learner_policy.get_step())
        #print('actor at: %s' % actor_policy.get_step())

        #test with clean actor, clean environment
        print('[*] testing for %s episodes per state' % test_episodes)
        #env = get_env(429)
        env.keep_env = True
        actor_policy = Policy(state_shape, n_actions, 
                'actor_test', act_int=act_int, sess=sess)
        actor = Actor(env, actor_policy, learner_policy)

        all_stats = {g: {k:[] for k in env.game_states[g]} 
                for g in env.game_states.keys()}
        for game in env.game_states.keys():
            for state in env.game_states[game]:
                print('[*] testing %s (%s)' % (game, state))
                actor.env.switch_env(game, state)
                info = actor.test(episodes=3,
                        render=test_render)

                #extract some basic stats from raw results
                all_stats[game][state] = pd.DataFrame.from_dict(data=info
                        ).describe().loc[['min', 'max', 'mean', 'std']]
                #change column names to include state name
                all_stats[game][state].columns = ['%s_%s' % (state, x) 
                        for x in all_stats[game][state].columns.values]

                #print(all_stats[game][state], '\n')

        #output stats FIXME: write dfs_concat, calc_* to disk
        cols = {'rewards': [], 'steps': []}
        for game in all_stats:
            print(game, (79-len(game))*'*')
            dfs = [all_stats[game][state] for state in all_stats[game]]
            dfs_concat = pd.concat(dfs, axis=1)
            print(dfs_concat, '\n')
            #for col in dfs_concat.columns.values:
            #    if 'reward' in col:
            #        cols['rewards'].append(col)
            #    elif 'step' in col:
            #        cols['steps'].append(col)
            #r = dfs_concat[cols['rewards']]
            #s = dfs_concat[cols['steps']]

            #calc_s = pd.concat([s.min(axis=1).rename('calc_min'), 
            #        s.max(axis=1).rename('calc_max'), 
            #        s.mean(axis=1).rename('calc_mean'), 
            #        s.std(axis=1).rename('calc_std')], axis=1)
            #calc_r = pd.concat([r.min(axis=1).rename('calc_min'), 
            #        r.max(axis=1).rename('calc_max'), 
            #        r.mean(axis=1).rename('calc_mean'), 
            #        r.std(axis=1).rename('calc_std')], axis=1)
        
            cols = [x for x in dfs_concat.columns.values if 'reward' in x]
            calc_r = pd.concat([dfs_concat[cols].mean(
                    axis=1).rename('calc_mean'), dfs_concat[cols].std(
                    axis=1).rename('calc_std')], axis=1)
            print(calc_r, '\n\n')

            #print('\nsteps:\n', calc_s, '\nrewards:\n', calc_r)

#wrapper that handles learning on multiple retro env states
    #changes to different state / game each episode
    #uses delta rewards as recommended
class Env_Controller():
    def __init__(self, csv, init=None):
        #games, states read from csv
        #self.game_states = self._csv_dict(csv)

        games = ['SonicTheHedgehog-Genesis', 
                'SonicTheHedgehog2-Genesis',
                'SonicAndKnuckles3-Genesis']
        self.all_game_states = {k:retro.list_states(k) for k in games}
        #58 total states

        #states yet to be added
        self.game_states = {} #current states used for training
        self.available_states = []
        for game in games:
            for state in self.all_game_states[game]:
                self.available_states.append((game, state))
        
        #start with two env states
        add_n_games = 44 #FIXME: steps / 10k + 2
        for _ in range(add_n_games): 
            self._add_state()

        if init:
            self.env = retro_contest.local.make(game=init[0], state=init[1])
        self.act_int = False
        self.action_space = self.env.action_space
        self.done = False

        #FIXME: every x steps add an env state
            #progressively add
        self.keep_env = False
        self.switch_interval = 10000 #steps before adding a state
        self.switch_steps = 0
        self.add_interval = 300000 #350000 is ~ 1h at 3 batch/sec
        self.add_steps = 0
            # ~60 hrs to full dataset

        self.episode_steps = 0
        self.episode_len = 200 #limit length of episodes while training

        self._cur_r = 0.0
        self._max_r = 0.0

    def _add_state(self, game_state=None):
        if len(self.available_states) == 0:
            return
        if game_state == None:
            game_state = self.available_states.pop(0)

        self._last_added = game_state
        g, s = game_state
        if g in self.game_states.keys():
            self.game_states[g].append(s)
        else:
            self.game_states[g] = [s]

        print('[*] added: %s %s' % (g, s))
        
    def _csv_dict(self, csv):
        df = pd.read_csv(csv, header=0, index_col=0)
        return {k: list(v) for k,v in df.groupby("game")["state"]}

    def create_env(self, game=None, state=None):
        assert(game or not state) #dont provide a state without a game
        self._cur_r = 0.0
        self._max_r = 0.0
        self.switch_steps = 0
        self.episode_steps = 0
        #chooses random game, state
        if game == None:
            game = np.random.choice(list(self.game_states.keys()))
        if state == None:
            state = np.random.choice(self.game_states[game])

        #priority game switching during adding process
            #chance gradually decreases, has choice to be prev chosen 
        min_chance = 0.25 #real chance is min + (1 / num_envs)
        if len(self.available_states) > 0:
            arr = [self._last_added, (game, state)]
            idx = np.random.choice(len(arr), p=[min_chance, 1 - min_chance])
            game, state = arr[idx]

        self.env = retro_contest.local.make(game=game, state=state)
        #print('    curr env: %s(%s)' % (game, state))

    def switch_env(self, game=None, state=None):
        self.env.close()
        self.create_env(game, state)

    def seed(self, n):
        self.env.seed(n)

    def reset(self):
        self._cur_r = 0.0
        self._max_r = 0.0
        self.episode_steps = 0
        return self.env.reset()

    def render(self):
        self.env.render()

    def step(self, action):
        #add state on interval
        if self.add_steps > self.add_interval:
            self._add_state()
            self.add_steps = 0
        #swap env on interval
        if self.done:
            if not self.keep_env and self.switch_steps > self.switch_interval:
                self.switch_env() #when episode is finished, swap envs
            self.reset()
            obs = self.env.step(action)
        else:
            obs = self.env.step(action)
        
        state, reward, done, info = obs
        
        #scale reward
        reward = reward * 0.01
        
        #FIXME: enforce and increase limit on episode length on interval
        # if self.episode_steps >= self.episode_len:
            #done = False
        #if ???:
            #self.episode_len += 100

        self.done = done

        #delta rewards (currently not working, 4hr bad policy)
        #self._cur_r += reward
        #reward = max(0.0, self._cur_r - self._max_r)
        #self._max_r = max(self._max_r, self._cur_r)
        self.switch_steps += 1
        self.add_steps += 1
        self.episode_steps += 1
        return state, reward, done, info

if __name__ == '__main__':
    main()

