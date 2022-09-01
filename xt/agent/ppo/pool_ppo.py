# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Build Atari agent for ppo algorithm."""

import numpy as np
from time import time, sleep
from zeus.common.ipc.message import message, set_msg_info
from absl import logging
from xt.agent.ppo.ppo import PPO
from xt.agent.ppo.default_config import GAMMA, LAM
from zeus.common.util.register import Registers
from collections import defaultdict
import logging


@Registers.agent
class PoolPpo(PPO):
    """Atari Agent with PPO algorithm."""

    def __init__(self, env, alg, agent_config, **kwargs):
        super().__init__(env, alg, agent_config, **kwargs)
        self.keep_seq_len = True
        self.next_state = None
        self.next_action = None
        self.next_value = None
        self.next_log_p = None
        self.env_num = self.env.size  # agent_config.get("wait_nums")
        self.trajectory = [defaultdict(list) for i in range(self.env.size)]

    def do_one_interaction(self, raw_state_list, use_explore=True):
        _start0 = time()
        action_list = self.infer_action(raw_state_list, use_explore)
        self._stats.inference_time += time() - _start0

        _start1 = time()
        next_raw_state, reward, done, info = self.env.step(action_list)
        self._stats.env_step_time += time() - _start1
        self._stats.iters += 1
        self.handle_env_feedback(next_raw_state, reward, done, info, use_explore)
        return next_raw_state

    def clear_trajectory(self):
        for _ in self.trajectory:
            _.clear()

    def run_one_episode(self, use_explore, need_collect):
        self.clear_trajectory()
        state = self.env.get_init_state()
        # GGLC
        # if isinstance(state, np.ndarray) and len(state.shape)==4:
        #     state = list(state)
        # print('[GGLC] pool_ppo#65 state: ', type(state), len(state), state[0].shape)

        self._stats.reset()
        self.transition_data = [defaultdict() for _ in range(self.env_num)]
        for _ in range(int(self.max_step)):
            state = self.do_one_interaction(state, use_explore)

            if need_collect:
                for i, transition_data in enumerate(self.transition_data):
                    self.add_to_trajectory(transition_data, i)

        # print('[GGLC] pool_ppo#76 state: ', type(state), len(state), state[0].shape)
        last_pred = self.alg.predict(state)
        return self.get_trajectory(last_pred)

    def handel_predict_value(self, state, predict_val):
        action = predict_val[0]
        logp = predict_val[1]
        value = predict_val[2]

        # update transition data
        for i in range(len(self.transition_data)):
            self.transition_data[i].update({
                'cur_state': state[i],
                'action': action[i],
                'logp': logp[i],
                'value': value[i],
            })

        return action

    def add_to_trajectory(self, transition_data, eid=None):
        # print('[GGLC] pool_ppo#95: ', eid, transition_data)
        if eid is not None:
            for k, val in transition_data.items():
                self.trajectory[eid][k].append(val)
        else:
            for k, val in transition_data.items():
                self.trajectory[k].append(val)

    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
        for i in range(len(self.transition_data)):
            self.transition_data[i].update({
                "reward": np.sign(reward[i]) if use_explore else reward[i],
                "done": done[i],
                "info": info[i]
            })

        return self.transition_data

    def data_proc2(self, index):
        """Process data."""
        traj = self.trajectory[index]
        state = np.asarray(traj['cur_state'])
        action = np.asarray(traj['action'])
        logp = np.asarray(traj['logp'])
        value = np.asarray(traj['value'])
        reward = np.asarray(traj['reward'])
        done = np.asarray(traj['done'])

        tr = np.sum(reward)
        if tr > 0:
            pass

        next_value = value[1:]
        value = value[:-1]

        done = np.expand_dims(done, axis=1)
        reward = np.expand_dims(reward, axis=1)
        discount = ~done * GAMMA
        delta_t = reward + discount * next_value - value
        adv = delta_t

        for j in range(len(adv) - 2, -1, -1):
            adv[j] += adv[j + 1] * discount[j] * LAM

        self.trajectory[index]['cur_state'] = state
        self.trajectory[index]['action'] = action
        self.trajectory[index]['logp'] = logp
        self.trajectory[index]['adv'] = adv
        self.trajectory[index]['old_value'] = value
        self.trajectory[index]['target_value'] = adv + value

        del self.trajectory[index]['value']

    def get_trajectory(self, last_pred=None):
        """Get trajectory"""
        # Need copy, when run with explore time > 1,
        # if not, will clear trajectory before sent.
        last_val = last_pred[2]

        trajectory = []
        for i, tj in enumerate(self.trajectory):
            self.trajectory[i]['value'].append(last_val[i])
            self.data_proc2(i)
            tmp = message(self.trajectory[i].copy())
            set_msg_info(tmp, agent_id=i)
            trajectory.append(tmp)
        # trajectory = message(deepcopy(self.trajectory))

        return trajectory
