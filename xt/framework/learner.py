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
"""Create a module cover the training process within the RL problems."""
import multiprocessing
# multiprocessing.set_start_method('spawn')
import os
import pickle
import threading
from time import time, sleep
from copy import deepcopy
from multiprocessing import Queue, Process

import dill
import numpy as np
from absl import logging
from collections import deque, defaultdict
import setproctitle

from xt.environment import env_builder
from xt.framework.trainer import build_alg_with_trainer
from xt.framework.predictor import Predictor
from xt.algorithm.pbt import PbtAid
from zeus.visual.tensorboarder import SummaryBoard
from zeus.common.util.evaluate_xt import make_workspace_if_not_exist, parse_benchmark_args
from zeus.common.ipc.uni_comm import UniComm
from zeus.common.ipc.message import message, get_msg_data, set_msg_info, set_msg_data, get_msg_info
from zeus.common.util.common import bytes_to_str
from zeus.common.util.hw_cloud_helper import mox_makedir_if_not_existed
from zeus.common.util.logger import Logger, StatsRecorder
from zeus.common.util.profile_stats import PredictStats, TimerRecorder

import tensorflow as tf


class Learner(object):
    """Learner manage the train-processing of whole RL pipe-line."""

    def __init__(
            self,
            alg_para,
            env_para,
            agent_para,
            eval_adapter=None,
            data_url=None,
            benchmark_info=None,
            name="T0",
            **kwargs
    ):
        self._name = name
        self.alg_para = deepcopy(alg_para)
        self.process_num = self.alg_para.get("process_num", 1)

        self.eval_adapter = eval_adapter

        self.train_worker = None

        self.send_train = None
        self.send_predict = UniComm("ShareByPlasma")
        self.send_broker = None
        self.send_broker_predict = Queue()
        self.stats_deliver = None

        self.train_lock = threading.Lock()
        self.alg = None
        self.trainer = None
        self.shared_buff = None

        _model_dir = ["models", "benchmark"]
        self.bm_args = \
            parse_benchmark_args(env_para, alg_para, agent_para, benchmark_info)
        self.workspace, _archive, _job = \
            make_workspace_if_not_exist(self.bm_args, _model_dir, task_name=self._name)

        self.bm_board = SummaryBoard(_archive, _job)
        self.model_path = os.path.join(self.workspace, _model_dir[0])
        logging.info("{}\nworkspace:\n\t{}\n".format("*" * 10, self.workspace))

        self.max_step = agent_para.get("agent_config", {}).get("complete_step")
        self.max_episode = agent_para.get("agent_config", {}).get("complete_episode")
        self._log_interval = benchmark_info.get("log_interval_to_train", 10)


        self._explorer_ids = list()

        self._pbt_aid = None

        # For Cloud
        self.s3_path = None
        if data_url is not None:
            self.s3_path = os.path.join(data_url, _model_dir[0])
            mox_makedir_if_not_existed(self.s3_path)

        self.shared_queue = kwargs.get("shared_queue")
        self.quantization = kwargs.get("quantization", False)


    def add_to_pbt(self, pbt_config, metric, weights):
        """Add this lerner to population."""
        self._pbt_aid = PbtAid(self.name, self.alg_para, pbt_config, metric, weights)

    @property
    def explorer_ids(self):
        return self._explorer_ids

    @explorer_ids.setter
    def explorer_ids(self, val):
        self._explorer_ids = val

    @property
    def name(self):
        return self._name

    def async_predict(self):
        """Create predict thread."""
        predict = [
            PredictThread(
                i,
                self.alg,
                self.send_predict,
                self.send_broker,
                self.stats_deliver,
                self.train_lock,
            )
            for i in range(2)
        ]
        predict_thread = [threading.Thread(target=t.predict) for t in predict]

        for t in predict_thread:
            t.setDaemon(True)
            t.start()

    def create_predictor(self):
        """Create predictor."""
        config_info = {'alg_para': self.alg_para}
        predictor = Predictor(0, config_info, self.send_predict,
                              self.send_broker_predict, self._name)
        # self.send_predict = predictor.request_q
        # print("send predict", self.send_predict)
        p = Process(target=predictor.start)
        p.daemon = False

        p.start()

    def init_async_train(self):
        """Create train worker."""
        self.train_worker = TrainWorker(
            self.send_train,
            self.alg,
            self.train_lock,
            self.model_path,
            self.send_broker,
            self.s3_path,
            self.max_step,
            self.max_episode,
            self.stats_deliver,
            self.eval_adapter,
            log_interval=self._log_interval,
            name=self._name,
            shared_queue=self.shared_queue,
            quantization=self.quantization
        )
        self.train_worker.explorer_ids = self.explorer_ids
        self.train_worker.pbt_aid = self._pbt_aid

    def submit_algorithm(self, alg_instance, trainer_obj, shared_buff):
        """Submit an algorithm, to update algorithm instance description."""
        self.alg = alg_instance
        self.trainer = trainer_obj
        self.shared_buff = shared_buff

    def start(self):
        """Start all system."""
        self.create_predictor()
        alg, trainer_obj, shared_list = build_alg_with_trainer(
            deepcopy(self.alg_para), self.send_broker, self.model_path, self.process_num
        )
        self.submit_algorithm(alg, trainer_obj, shared_list)

        # self.async_predict()
        self.init_async_train()

    def main_loop(self):
        """Run with while True, cover the working loop."""
        self.train_worker.train()
        # user operation after train process


class TrainWorker(object):
    """TrainWorker Process manage the trajectory data set and optimizer."""

    def __init__(
            self,
            train_q,
            alg,
            lock,
            model_path,
            model_q,
            s3_path,
            max_step,
            max_episode,
            stats_deliver,
            eval_adapter=None,
            **kwargs,
    ):
        self.train_q = train_q
        self.alg = alg
        self.lock = lock
        self.model_path = model_path
        self.model_q = model_q
        self.actor_reward = defaultdict(float)
        self.actor_trajectory = defaultdict(int)
        self.rewards = []
        self.s3_path = s3_path
        self.max_step = max_step
        self.actual_step = 0
        self.name = kwargs.get('name', 'T0')

        self.quantization = kwargs.get('quantization', False)
        # print("quantization ==================== {}".format(self.quantization))

        self.max_episode = max_episode
        self.elapsed_episode = 0  # off policy, elapsed_episode > train count

        self.won_in_episodes = deque(maxlen=256)
        self.train_count = 0

        self.stats_deliver = stats_deliver
        self.e_adapter = eval_adapter

        self.logger = Logger(os.path.dirname(model_path))
        self._metric = TimerRecorder("leaner_model", maxlen=50,
                                     fields=("fix_weight", "send"))

        self._log_interval = kwargs["log_interval"]

        self._explorer_ids = None
        self._pbt_aid = None
        self._train_data_counter = defaultdict(int)
        shared_queue = kwargs.get("shared_queue")
        self.raw_weights_queue = shared_queue[0]  # type:Queue
        self.compress_weights_queue = shared_queue[1]  # type:Queue

        self.keras_model_file_path = "explorer" + str(time()) + ".h5"

        self.save_model_times = 0
        self.save_model_time = 0

        self.train_times = 0
        self.train_time = 0

    @property
    def explorer_ids(self):
        return self._explorer_ids

    @explorer_ids.setter
    def explorer_ids(self, val):
        self._explorer_ids = val

    @property
    def pbt_aid(self):
        return self._pbt_aid

    @pbt_aid.setter
    def pbt_aid(self, val):
        self._pbt_aid = val

    def _dist_policy(self, weight=None, save_index=-1, dist_cmd="explore"):
        """Distribute model tool."""
        explorer_set = self.explorer_ids

        ctr_info = self.alg.dist_model_policy.get_dist_info(save_index, explorer_set)

        if isinstance(ctr_info, dict):
            ctr_info = [ctr_info]

        for _ctr in ctr_info:
            to_send_data = message(weight, cmd=dist_cmd, **_ctr)
            self.model_q.send(to_send_data)

    def _handle_eval_process(self, loss):
        if not self.e_adapter:
            return

        if self.e_adapter.if_eval(self.train_count):
            weights = self.alg.get_weights()
            self.e_adapter.to_eval(weights,
                                   self.train_count,
                                   self.actual_step,
                                   self.logger.elapsed_time,
                                   self.logger.train_reward,
                                   loss)
        elif not self.e_adapter.eval_result_empty:
            eval_ret = self.e_adapter.fetch_eval_result()
            if eval_ret:
                logging.debug("eval stats: {}".format(eval_ret))
                self.stats_deliver.send({"data": eval_ret, "is_bm": True}, block=True)

    def _meet_stop(self):
        if self.max_step and self.actual_step > self.max_step:
            return True

        # Under pbt set, the max_episode need set into pbt_config
        # Owing to the reset of episode count after each pbt.exploit
        if self.max_episode and self.elapsed_episode > self.max_episode:
            return True

        return False

    def train(self):
        """Train model."""
        if not self.alg.async_flag:
            if self.quantization:
                keras_model_file_path = self.alg.save_keras_model()
                self.raw_weights_queue.put(keras_model_file_path)
                # here bolt
                serialized_model = self.compress_weights_queue.get()
                self._dist_policy(weight=serialized_model)

                # p = multiprocessing.Process(target=self.compass_weight)
                # p.start()
                pass
            else:
                policy_weight = self.alg.get_weights()
                self._dist_policy(weight=policy_weight)

        loss = 0


        while True:
            for _tf_val in range(self.alg.prepare_data_times):
                # logging.debug("wait data for preparing-{}...".format(_tf_val))
                with self.logger.wait_sample_timer:
                    data = self.train_q.recv()
                    # print("self.train_q.recv()======================")

                with self.logger.prepare_data_timer:
                    data = bytes_to_str(data)
                    self.record_reward(data)
                    self.alg.prepare_data(data["data"], ctr_info=data["ctr_info"])

                # dqn series algorithm will count the 'SARSA' as one episode.
                # and, episodic count will used for train ready flag.
                # each pbt exploit.step will reset the episodic count.
                self.elapsed_episode += 1
                # logging.debug("Prepared data-{}.".format(_tf_val))
                # support sync model before

            # run pbt if need.
            if self.pbt_aid:
                if self.pbt_aid.meet_stop(self.elapsed_episode):
                    break

                cur_info = dict(episodic_reward_mean=self.logger.train_reward_avg,
                                elapsed_step=self.actual_step,
                                elapsed_episode=self.elapsed_episode)
                new_alg = self.pbt_aid.step(cur_info, cur_alg=self.alg)

                if new_alg:  # re-assign algorithm if need
                    self.alg = new_alg
                    if not self.alg.async_flag:
                        policy_weight = self.alg.get_weights()
                        self._dist_policy(weight=policy_weight)
                        continue

            if self._meet_stop():
                self.stats_deliver.send(self.logger.get_new_info(), block=True)
                break

            if not self.alg.train_ready(self.elapsed_episode, dist_dummy_model=self._dist_policy):
                continue

            with self.lock, self.logger.train_timer:
                # logging.debug("start train process-{}.".format(self.train_count))
                train_start = time()
                loss = self.alg.train(episode_num=self.elapsed_episode)
                train_end = time()
                self.train_time += (train_end - train_start)
                self.train_times += 1
                # if self.train_times % 100 == 0:
                #     print(self.alg)
                #     print("train time ================ {}".format(self.train_time / self.train_times))

            # with open("/home/xys/primary_xingtian/xingtian/record.txt", "a+") as f:
            #     f.write("\n\n\n\ntrain_times === {}\n\n\n\n".format(self.train_times))

            if type(loss) in (float, np.float64, np.float32, np.float16, np.float):
                self.logger.record(train_loss=loss)

            with self.lock:
                if self.alg.if_save(self.train_count):
                    _name = self.alg.save(self.model_path, self.train_count)
                    # logging.debug("to save model: {}".format(_name))

            self._handle_eval_process(loss)

            # The requirement of distribute model is checkpoint ready.
            if not self.alg.async_flag and self.alg.checkpoint_ready(self.train_count):

                if self.quantization:
                    # logging.info("======================get weights====================")
                    # sleep(0.01)
                    start_compass = time()
                    # here bolt
                    # if not self.compress_weights_queue.empty():
                    if True:
                        # print("================get compass weight====================")
                        keras_model_file_path = self.alg.save_keras_model()
                        self.raw_weights_queue.put(keras_model_file_path)
                        serialized_model = self.compress_weights_queue.get()
                        # quantization_start = time()
                        # quantization_end = time()
                        # print("after compass ===================")
                        # print("quantization_time ============= {}".format(quantization_end - quantization_start))
                        if not self.raw_weights_queue.empty():
                            file = self.raw_weights_queue.get()
                            os.remove(file)
                        # self.raw_weights_queue.put(keras_model_file_path)
                        # print("self.raw_weights_queue .lenth ================{}".format(self.raw_weights_queue.qsize()))

                        # print("self.compress_weights_queue .lenth ================{}".format(self.compress_weights_queue.qsize()))
                        end_get = time()
                        # print("self.compress_weights_queue.get() =============== {}".format(end_get - start_get))
                        # while not self.compress_weights_queue.empty():
                        #     policy_weight = self.compress_weights_queue.get()
                        # print("======================serialized_model type ===================={}".
                        #              format(serialized_model))
                        _dist_st = time()
                        self._dist_policy(serialized_model, self.train_count)
                        # print("policy_weight ===================== {}".format(policy_weight))
                        self._metric.append(send=time() - _dist_st)
                        self._metric.report_if_need()
                    # else:
                    #     print("empty============ ===================")
                    self.save_model_times += 1
                    self.save_model_time += time() - start_compass

                    # if self.save_model_times % 100 == 0:
                    #     logging.info(
                    #         "save_model_time ============= {}".format(self.save_model_time / self.save_model_times))
                else:
                    # print("send weight=======================================")
                    _save_t1 = time()
                    policy_weight = self.alg.get_weights()
                    self._metric.append(fix_weight=time() - _save_t1)

                    _dist_st = time()
                    # if policy_weight
                    self._dist_policy(policy_weight, self.train_count)
                    # print("policy_weight ===================== {}".format(policy_weight))
                    self._metric.append(send=time() - _dist_st)
                    self._metric.report_if_need()

                quantization_end = time()
                # print("quantization_time ================================== {}".format(
                #     quantization_end - quantization_start))
            else:
                if self.alg.checkpoint_ready(self.train_count):
                    policy_weight = self.alg.get_weights()
                    weight_msg = message(policy_weight, cmd="predict{}".format(self.name), sub_cmd='sync_weights')
                    self.model_q.send(weight_msg)

            if self.train_count % self._log_interval == 0:
                self.stats_deliver.send(self.logger.get_new_info(), block=True)

            self.train_count += 1

    def record_reward(self, train_data):
        """Record reward in train."""
        broker_id = get_msg_info(train_data, 'broker_id')
        explorer_id = get_msg_info(train_data, 'explorer_id')
        agent_id = get_msg_info(train_data, 'agent_id')
        key = (broker_id, explorer_id, agent_id)
        # key = learner_stats_id(train_data["ctr_info"])
        # record the train_data received
        self._train_data_counter[key] += 1

        self.alg.dist_model_policy.add_processed_ctr_info(key)
        data_dict = get_msg_data(train_data)

        # update multi agent train reward without done flag
        if self.alg.alg_name in ("QMixAlg",) or self.alg.alg_name in ("SCCAlg",):  # fixme: unify the record op
            self.actual_step += np.sum(data_dict["filled"])
            self.won_in_episodes.append(data_dict.pop("battle_won"))
            self.logger.update(explore_won_rate=np.nanmean(self.won_in_episodes))
            self.logger.record(
                step=self.actual_step,
                train_reward=np.sum(data_dict["reward"]),
                train_count=self.train_count,
            )
            return
        elif self.alg.alg_config['api_type'] == "unified":
            self.actual_step += len(data_dict["done"])
            self.logger.record(
                step=self.actual_step,
                train_reward=np.sum(data_dict["reward"]),
                train_count=self.train_count,
            )
            return

        # print("==============================REW {}==================================".format(np.sum(data_dict['reward'])))
        data_length = len(data_dict["done"])  # fetch the train data length
        # print(data_dict)
        # for data_index in range(data_length):
        #   info = data_dict["info"][data_index]
        #  print(info)
        # print("{}=={}".format(info['eval_reward'],info['real_done']))

        for data_index in range(data_length):
            reward = data_dict["reward"][data_index]
            done = data_dict["done"][data_index]
            info = data_dict["info"][data_index]
            self.actual_step += 1
            if isinstance(info, dict):
                self.actor_reward[key] += info.get("eval_reward", reward)
                self.actor_trajectory[key] += 1
                done = info.get("real_done", done)

            if done:
                # print("=============================={},REW {}==================================".format(key,self.actor_reward[key]))
                self.logger.record(
                    step=self.actual_step,
                    train_count=self.train_count,
                    train_reward=self.actor_reward[key],
                    trajectory_length=self.actor_trajectory[key],
                )
                # logging.debug("{} epi reward-{}. with len-{}".format(
                #     key, self.actor_reward[key], self.actor_trajectory[key]))
                self.actor_reward[key] = 0.0
                self.actor_trajectory[key] = 0

    def compass_weight(self):
        while True:
            print("===========================compress weights....==============================")
            model_file_path = self.raw_weights_queue.get()
            print("===========================after get weights....==============================")

            if not self.raw_weights_queue.empty():
                continue
            logging.info(os.path.exists(model_file_path))
            print("model_file_path =============== {}".format(model_file_path))
            model1 = tf.keras.models.load_model("/home/xys/primary_xingtian/xingtian/" + model_file_path)
            converter = tf.lite.TFLiteConverter.from_keras_model(model1)
            print("okoooooooooooooooooooooo000000000000oo")

            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            serialize_tflite_model = dill.dumps(tflite_model)
            self.compress_weights_queue.put(serialize_tflite_model)


class PredictThread(object):
    """Predict Worker for async algorithm."""

    def __init__(self, thread_id, alg, request_q, reply_q, stats_deliver, lock):
        self.alg = alg
        self.thread_id = thread_id
        self.request_q = request_q
        self.reply_q = reply_q
        self.lock = lock

        self.stats_deliver = stats_deliver
        self._report_period = 200

        self._stats = PredictStats()

    def predict(self):
        """Predict action."""
        while True:

            start_t0 = time()
            data = self.request_q.recv()
            state = get_msg_data(data)
            self._stats.obs_wait_time += time() - start_t0

            start_t1 = time()
            with self.lock:
                action = self.alg.predict(state)
            self._stats.inference_time += time() - start_t1

            set_msg_info(data, cmd="predict_reply")
            set_msg_data(data, action)

            # logging.debug("msg to explore: ", data)
            self.reply_q.send(data)

            self._stats.iters += 1
            if self._stats.iters > self._report_period:
                _report = self._stats.get()
                self.stats_deliver.send(_report, block=True)


def patch_model_config_by_env_info(config, env_info):
    model_info = config["model_para"]
    if "model_config" not in model_info["actor"].keys():
        model_info["actor"].update({"model_config": dict()})

    model_config = model_info["actor"]["model_config"]
    model_config.update({"action_type": env_info.get("action_type")})

    return model_info


def patch_alg_within_config(config, node_type="node_config"):
    """combine the algorithm parameters"""
    alg_para = config["alg_para"].copy()
    agent_para = config["agent_para"]
    node_config = config[node_type]

    # for quickly run 2s_vs_1sc map
    env_attr = {
        "state_shape": 27,
        # obs_shape with been extend with action&agent id in algorithm!
        "obs_shape": 17,
        "n_actions": 7,
        "n_agents": 2,
        "episode_limit": 300,
        "api_type": "standalone",
        "agent_ids": [0],
    }

    # get env info
    env = env_builder(**config["env_para"])
    env_info = env.get_env_info()
    env.close()

    if "alg_config" not in alg_para:
        alg_para["alg_config"] = dict()

    alg_para["alg_config"].update(
        {
            "instance_num": config["env_num"] * len(node_config),
            "agent_num": agent_para.get("agent_num", 1),
            "env_attr": env_attr,
            "api_type": env_info.get("api_type")
        }
    )

    model_info = patch_model_config_by_env_info(config, env_info)
    # update env attr into model info
    model_info["actor"]["model_config"].update(env_attr)
    alg_para["model_info"] = model_info
    config.update({"alg_para": alg_para})

    return config


def setup_learner(config, eval_adapter, learner_index, data_url=None, shared_queue=None):
    """Start learner."""
    env_para = config["env_para"]
    agent_para = config["agent_para"]
    alg_para = deepcopy(config["alg_para"])
    model_info = alg_para["model_info"]

    # set actor.type as learner
    model_info["actor"].update({"type": "learner"})

    # add benchmark id
    bm_info = config.get("benchmark", dict())

    quantization = model_info["actor"].get("quantization", False)

    learner = Learner(
        alg_para,
        env_para,
        agent_para,
        eval_adapter=eval_adapter,
        data_url=data_url,
        benchmark_info=bm_info,
        name="T{}".format(learner_index),
        shared_queue=shared_queue,
        quantization=quantization
    )

    learner.config_info = config

    return learner


def learner_stats_id(ctr_info):
    """Assemble stats id."""
    broker_id = ctr_info.get('broker_id')
    explorer_id = ctr_info.get('explorer_id')
    agent_id = ctr_info.get('agent_id')
    return "_".join(map(str, (broker_id, explorer_id, agent_id)))
