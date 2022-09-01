from time import time

import numpy as np

from xt.algorithm.impala.impala import IMPALA
from xt.algorithm.impala.default_config import BATCH_SIZE, GAMMA

from zeus.common.util.register import Registers


@Registers.algorithm
class IMPALALite(IMPALA):
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

            loss_list.append(actor_loss)
        self._init_train_list()
        result = np.mean(loss_list)
        # print(type(loss_list[0]))
        train_end = time()
        self.train_times += 1
        self.train_time += (train_end - train_start)
        # if self.train_times % 100 == 0:
        #     print("train_time ============= {}".format(self.train_time / self.train_times))
        return result

    def save_keras_model(self):
        return self.actor.save_keras_model()
