# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
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
"""Bulid optimized impala algorithm by merging the data process and inferencing into tf.graph."""

import os
import threading
from time import time

import numpy as np

from xt.algorithm import Algorithm
from xt.algorithm.impala.default_config import BATCH_SIZE, GAMMA
from zeus.common.ipc.uni_comm import UniComm
from zeus.common.util.register import Registers
from xt.model.tf_compat import loss_to_val
from zeus.common.util.common import import_config
from xt.algorithm.alg_utils import DivideDistPolicy, FIFODistPolicy, EqualDistPolicy


@Registers.algorithm
class IMPALAOpt(Algorithm):
    """Build IMPALA algorithm."""

    def __init__(self, model_info, alg_config, **kwargs):
        import_config(globals(), alg_config)
        super().__init__(alg_name="impala",
                         model_info=model_info["actor"],
                         alg_config=alg_config)
        self.states = list()
        self.behavior_logits = list()
        self.actions = list()
        self.dones = list()
        self.rewards = list()
        self.async_flag = False

        # update to divide model policy
        self.dist_model_policy = FIFODistPolicy(
            alg_config["instance_num"],
            prepare_times=self._prepare_times_per_train)

        self.use_train_thread = False
        if self.use_train_thread:
            self.send_train = UniComm("LocalMsg")
            train_thread = threading.Thread(target=self._train_thread)
            train_thread.setDaemon(True)
            train_thread.start()
        self._init_train_list()

        self.episode_len = alg_config.get("episode_len", 128)

        self.train_times = 0
        self.train_time = 0

    def _init_train_list(self):
        self.state = list()
        self.action = list()
        self.dones = list()
        self.pred_a = list()
        self.rewards = list()

    def _train_thread(self):
        while True:
            data = self.send_train.recv()
            batch_state, batch_logit, batch_action, batch_done, batch_reward = data
            actor_loss = self.actor.train(
                batch_state,
                [batch_logit, batch_action, batch_done, batch_reward],
            )

    # def train(self, **kwargs):
    #     """Train impala agent by calling tf.sess."""
    #     states = np.concatenate(self.states)
    #     behavior_logits = np.concatenate(self.behavior_logits)
    #     actions = np.concatenate(self.actions)
    #     dones = np.concatenate(self.dones)
    #     rewards = np.concatenate(self.rewards)
    #
    #     nbatch = len(states)
    #     count = (nbatch + BATCH_SIZE - 1) // BATCH_SIZE
    #     loss_list = []
    #
    #     for start in range(count):
    #         start_index = start * BATCH_SIZE
    #         env_index = start_index + BATCH_SIZE
    #         batch_state = states[start_index:env_index]
    #         batch_logit = behavior_logits[start_index:env_index]
    #         batch_action = actions[start_index:env_index]
    #         batch_done = dones[start_index:env_index]
    #         batch_reward = rewards[start_index:env_index]
    #
    #         actor_loss = self.actor.train(
    #             batch_state,
    #             [batch_logit, batch_action, batch_done, batch_reward],
    #         )
    #         loss_list.append(loss_to_val(actor_loss))
    #
    #     # clear states for next iter
    #     self.states.clear()
    #     self.behavior_logits.clear()
    #     self.actions.clear()
    #     self.dones.clear()
    #     self.rewards.clear()
    #     return np.mean(loss_list)

    def prepare_data(self, train_data, **kwargs):
        """Prepare the data for impala algorithm."""
        states, actions, dones, pred_a, rewards = self._data_proc(train_data)
        # print(states.shape)
        self.state.append(states)
        self.action.append(actions)
        self.dones.append(dones)
        self.pred_a.append(pred_a)
        self.rewards.append(rewards)

    def _train_proc(self):
        states = np.concatenate(self.state)
        actions = np.concatenate(self.action)
        dones = np.concatenate(self.dones)
        pred_a = np.concatenate(self.pred_a)
        rewards = np.concatenate(self.rewards)

        # print(states.shape)

        outputs = self.actor.predict([states, np.zeros((states.shape[0], 1))])
        probs = outputs[0]
        values = outputs[1]

        # probs.reshape(645, 4)
        # values.reshape(645, 1)
        # print("pred_a.shape =============== {}".format(pred_a.shape))
        # print("actions.shape =============== {}".format(actions.shape))

        state_len = self.episode_len + 1
        shape = (probs.shape[0] // state_len, state_len, probs.shape[1])
        probs = probs.reshape(shape)
        shape = (values.shape[0] // state_len, state_len, values.shape[1])
        values = values.reshape(shape)
        shape = (dones.shape[0] // self.episode_len, self.episode_len, dones.shape[1])
        dones = dones.reshape(shape)
        rewards = rewards.reshape(shape)
        shape = (actions.shape[0] // self.episode_len, self.episode_len, ) + actions.shape[1:]
        actions = actions.reshape(shape)
        pred_a = pred_a.reshape(shape)

        value = values[:, :-1]
        value_next = values[:, 1:]
        target_action = probs[:, :-1]
        discounts = ~dones * GAMMA

        behaviour_logp = self._logp(pred_a, actions)
        target_logp = self._logp(target_action, actions)
        radio = np.exp(target_logp - behaviour_logp)
        radio = np.minimum(radio, 1.0)
        radio = radio.reshape(radio.shape + (1,))
        deltas = radio * (rewards + discounts * value_next - value)

        adv = deltas
        traj_len = adv.shape[1]
        for j in range(traj_len - 2, -1, -1):
            adv[:, j] += adv[:, j + 1] * discounts[:, j + 1] * radio[:, j + 1]

        target_value = value + adv
        target_value_next = target_value[:, 1:]
        last_value = value_next[:, -1]
        last_value = last_value.reshape((last_value.shape[0], 1, last_value.shape[1]))
        target_value_next = np.concatenate((target_value_next, last_value), axis=1)
        pg_adv = radio * (rewards + discounts * target_value_next - value)

        shape = (pg_adv.shape[0] * pg_adv.shape[1], pg_adv.shape[2])
        pg_adv = pg_adv.reshape(shape)
        target_value = target_value.reshape(shape)

        shape = (states.shape[0] // state_len, state_len, ) + states.shape[1:]
        states = states.reshape(shape)
        states = states[:, :-1]
        shape = (states.shape[0] * states.shape[1], ) + states.shape[2:]
        states = states.reshape(shape)

        shape = (actions.shape[0] * actions.shape[1], ) + actions.shape[2:]
        actions = actions.reshape(shape)

        return states, pg_adv, target_value, actions

    def train(self, **kwargs):
        """Train agent."""
        train_start = time()
        _train_proc_start = time()
        state, pg_adv, target_value, action_matrix = self._train_proc()

        nbatch = len(state)
        count = (nbatch + BATCH_SIZE - 1) // BATCH_SIZE
        loss_list = []
        for start in range(count):
            start_index = start * BATCH_SIZE
            env_index = start_index + BATCH_SIZE
            state_fit = state[start_index:env_index]
            pg_adv_fit = pg_adv[start_index:env_index]
            value_fit = target_value[start_index:env_index]
            action_matrix_fit = action_matrix[start_index:env_index]

            # print("state_fit.shape={}; pg_adv_fit.shape={}; action_matrix_fit.shape={}; value_fit.shape={}".
            #       format(state_fit.shape, pg_adv_fit.shape, action_matrix_fit.shape, value_fit.shape))
            # print("state_fit.type={}; pg_adv_fit.type={}; action_matrix_fit.type={}; value_fit.type={}".
            #       format(state_fit.dtype, pg_adv_fit.dtype, action_matrix_fit.dtype, value_fit.dtype))
            actor_loss = self.actor.train(
                [state_fit, pg_adv_fit], [action_matrix_fit, value_fit]
            )

            # lite
            # loss_list.extend(actor_loss)
            # print("actor_loss =============================== {}".format(len(actor_loss)))
            loss_list.append(loss_to_val(actor_loss))

        # print("len('loss_list') ==================== {}".format(len(loss_list)))


        # for loss in loss_list:
        #     print("loss.shape ===== {}".format(loss.shape))
        # print(len(loss_list))
        self._init_train_list()
        result = np.mean(loss_list)
        # print("result ====================== {}".format(result))
        # print(type(loss_list[0]))
        train_end = time()
        self.train_times += 1
        self.train_time += (train_end - train_start)
        # if self.train_times % 100 == 0:
        #     print("train_time ============= {}".format(self.train_time / self.train_times))
        return result

    def save(self, model_path, model_index):
        """Save model."""
        actor_name = "actor" + str(model_index).zfill(5)
        actor_name = self.actor.save_model(os.path.join(model_path, actor_name))
        actor_name = actor_name.split("/")[-1]

        return [actor_name]

    # def prepare_data(self, train_data, **kwargs):
    #     """Prepare the data for impala algorithm."""
    #     state, logit, action, done, reward = self._data_proc(train_data)
    #     self.states.append(state)
    #     self.behavior_logits.append(logit)
    #     self.actions.append(action)
    #     self.dones.append(done)
    #     self.rewards.append(reward)

    # def predict(self, state):
    #     """Predict with actor inference operation."""
    #     pred = self.actor.predict(state)
    #
    #     return pred

    def predict(self, state):
        """Predict with actor inference operation."""
        # state = state.reshape((1,) + state.shape)
        dummp_value = np.zeros((1, 1))
        pred = self.actor.predict([state, dummp_value])
        # print("pred ==================== {}".format(pred))
        return pred

    # @staticmethod
    # def _data_proc(episode_data):
    #     """
    #     Process data for impala.
    #
    #     Agent will record the follows:
    #         states, behavior_logits, actions, dones, rewards
    #     """
    #     states = episode_data["cur_state"]
    #
    #     behavior_logits = episode_data["logit"]
    #     actions = episode_data["action"]
    #     dones = np.asarray(episode_data["done"], dtype=np.bool)
    #
    #     rewards = np.asarray(episode_data["reward"])
    #
    #     return states, behavior_logits, actions, dones, rewards

    def save_keras_model(self):
        return self.actor.save_keras_model()

    @staticmethod
    def _data_proc(episode_data):
        """Process data for impala."""
        states = episode_data["cur_state"]
        actions = episode_data["real_action"]
        rewards = np.asarray(episode_data["reward"])
        rewards = rewards.reshape((rewards.shape[0], 1))
        dones = np.asarray(episode_data["done"])
        dones = dones.reshape((dones.shape[0], 1))
        pred_a = np.asarray(episode_data["action"])

        return (states, actions, dones, pred_a, rewards)


    @staticmethod
    def _logp(prob, action):
        """Calculate log probabiliy of an action."""
        action_prob = np.sum(prob * action, axis=-1)
        return np.log(action_prob + 1e-10)
