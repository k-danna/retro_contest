
import numpy as np

class Batch():
    #FIXME: this is very slow
    def __init__(self, max_size=50000):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.next_states = []
        self.logits = []
        self.lstm_states = []
        self.max_size = max_size
        self.size = 0

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.next_states = []
        self.logits = []
        self.lstm_states = []
        self.size = 0

    def _pop_front(self):
        self.states.pop(0)
        self.actions.pop(0)
        self.rewards.pop(0)
        self.dones.pop(0)
        self.next_states.pop(0)
        self.logits.pop(0)
        self.lstm_states.pop(0)
        self.size -= 1

    def add(self, experience):

        if self.size == self.max_size:
            self._pop_front()

        state, action, reward, value, done, next_state, logit, lstm_state = experience
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.next_states.append(next_state)
        self.logits.append(logit)
        self.lstm_states.append(lstm_state)
        self.size += 1

    def get(self):
        s = np.asarray(self.states, dtype=np.float32)
        act = np.asarray(self.actions, dtype=np.float32)
        r = np.asarray(self.rewards, dtype=np.float32)
        v = np.asarray(self.values, dtype=np.float32)
        d = np.asarray(self.dones, dtype=np.float32)
        ns = np.asarray(self.next_states, dtype=np.float32)
        l = np.asarray(self.logits, dtype=np.float32)

        #separate the tuples into two lists, recombine
        #ls = np.asarray(self.lstm_states, dtype=np.float32)
        #ls_c = ls[:, 0, :, :]
        #ls_h = ls[:, 1, :, :]
        #ls = np.asarray([ls_c, ls_h], dtype=np.float32)
        #print(ls.shape)
        ls = np.asarray(self.lstm_states[0], dtype=np.float32)
        
        def discount(arr, n, normalize=False):
            for i in range(len(arr) - 1, -1, -1): #reverse iterate
                n = arr[i] + 0.99 * n
                arr[i] = n
            if normalize:
                arr = (arr - arr.mean()) / (arr.std() + 1e-8)
            return arr
        
        #generalized advantage
            #adv = discount(rewards + gamma * value_next - value)
        def generalized_advantage():
            v_last = 0.0 if self.dones[-1] == 0.0 else self.values[-1]
            _v = np.asarray(self.values + [v_last], dtype=np.float32)
            adv = r + 0.99 * _v[1:] - _v[:-1]
            #adv = discount(adv, v_last)
            discount(r, v_last) #discount rewards after calc
            return discount(adv, v_last)

            #adv = rew + 0.99 * _v[1:] - _v[:-1]
            #return discount(adv, v_last, normalize=True)

        def basic_advantage():
            r_last = 0.0 if self.dones[-1] else self.values[-1]
            return discount(r, r_last) - v

        #adv = generalized_advantage()
        #adv = basic_advantage()

        #FIXME: v-trace
            # v_trace = v + adv + 0.99 * (v_trace - v)
                # adv = reward + discount * value_next - value

        #clip rewards
        #r = np.clip(r, -1.0, 1.0)

        #normalize rewards
        #r = (r - r.mean()) / (r.std() + 1e-8)


        #calc advantage
        adv = basic_advantage()

        #importance weights, all c,p = 1 for on policy case
        #p = np.ones(adv.shape, dtype=np.float32)
        #c = np.ones(adv.shape, dtype=np.float32)

        #p_bar_t = min(c, p_t) ..?
        #p_t = min(p_bar, ratio)

        #c_bar = ?
        #c_t = min(c_bar, ratio)

        #calc v_trace targets (arxiv:1802.01561v2)
            #using simple advantage vs the proposed generalized adv
        _last = 0.0 if self.dones[-1] else self.values[-1]
        t = np.asarray(self.values + [_last], dtype=np.float32)
        _v = np.asarray(self.values + [_last], dtype=np.float32)
        for i in range(len(adv) - 1, -1, -1): #reverse iterate
            #target = v + adv + gamma * (next_target - next_v)
            t[i] = _v[i] + adv[i] + 0.99 * (t[i+1] - _v[i+1])
       
        r = t[:-1] # remove bootsrapped value


        return s, act, r, v, adv, d, ns, l, ls









