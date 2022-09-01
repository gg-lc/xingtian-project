import pathlib
import pickle
from time import time

import dill
import os

import numpy
import numpy as np
from copy import deepcopy

import tensorflow as tf

from tensorflow import quint8, float32

from tensorflow import keras

from xt.model.tf_compat import get_sess_graph

from zeus.common.util.common import import_config

from xt.model import XTModel
from xt.model.impala.default_config import ENTROPY_LOSS, LR

from zeus.common.util.register import Registers

from tensorflow.keras import backend as K


# change files : algorithm/impala/impala add function get_convert_from_session

class CustomModel(keras.Model):
    # @tf.function
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        start_train = time()
        x, y = data  # 这个data就是传入model.fit()的数据
        # print("gogoggooooooooooooooooooooooooo1")
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # entropy, cross_entropy = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            policy = y_pred[0]
            log_policy = K.log(policy + 1e-10)
            entropy = -policy * K.log(policy + 1e-10)
            cross_entropy = -y["output_actions"] * log_policy
            l1 = K.mean(x["adv"] * cross_entropy - ENTROPY_LOSS * entropy, 1)
            l2 = tf.reduce_mean(tf.losses.MSE(y["output_value"], y_pred[1]))
            loss = l1 * 2.0 / 3.0 + l2 / 3.0
            # tf.print("loss ======================= {}".format(loss))
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        # return {m.name: m.result() for m in self.metrics}
        end_train = time()
        # print("train_time ======================== {}".format(end_train - start_train))
        return {"loss": loss.numpy()}


@Registers.model
class ImpalaCnnLite(XTModel):
    def __init__(self, model_info):
        tf.compat.v1.enable_eager_execution()
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)
        sess, self.graph = get_sess_graph()
        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']
        self.interpreter = None
        self.save_model_times = 0
        self.keras_model_file_path = "explorer" + str(self.save_model_times) + ".h5"
        self.predict_times = 0
        self.model = self.create_model(model_info)
        self.h5_file_prefix = model_info.get("h5_file_prefix", "explorer_tflite")
        self.quantization = model_info.get("quantization", False)

        # print("ImpalaCnnLite model create==========================")

        self.init_interpret()

    def create_model(self, model_info):
        import tensorflow as tf
        from tensorflow.keras import Input, Model
        from tensorflow.keras.layers import Dense, Conv2D, Lambda, Flatten
        from tensorflow.keras.optimizers import Adam
        state_input = Input(shape=self.state_dim, name='state_input', dtype='uint8')
        state_input_1 = Lambda(layer_function)(state_input)
        advantage = Input(shape=(1,), name="adv")

        # convlayer = Conv2D(32, (8, 8), strides=(4, 4), activation='relu', padding='valid')(state_input_1)
        # convlayer = Conv2D(256, (4, 4), strides=(2, 2), activation='relu', padding='valid')(convlayer)
        # convlayer = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='valid')(convlayer)
        # flattenlayer = Flatten()(convlayer)
        # denselayer = Dense(4096, activation='relu')(flattenlayer)
        # denselayer = Dense(2048, activation='relu')(denselayer)
        # denselayer = Dense(4096, activation='relu')(denselayer)

        convlayer = Conv2D(32, (8, 8), strides=(4, 4), activation='relu', padding='valid')(state_input_1)
        convlayer = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='valid')(convlayer)
        convlayer = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='valid')(convlayer)
        flattenlayer = Flatten()(convlayer)
        denselayer = Dense(256, activation='relu', name="last")(flattenlayer)

        out_actions = Dense(self.action_dim, activation='softmax', name='output_actions')(denselayer)
        out_value = Dense(1, name='output_value')(denselayer)
        model = CustomModel(inputs=[state_input, advantage], outputs=[out_actions, out_value])
        # losses = {"output_actions": custom_loss, "output_value": 'mse'}
        # lossweights = {"output_actions": 1.0, "output_value": .5}

        decay_value = 0.00000000512

        model.compile(loss="mse", optimizer=Adam(lr=LR, clipnorm=40., decay=decay_value))
        # converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # converter.convert()

        # model.save(self.keras_model_file_path,
        #                            save_format='h5', include_optimizer=False)
        # model = tf.keras.models.load_model(self.keras_model_file_path, custom_objects={"CustomModel": CustomModel})
        # model.summary()
        return model

    def post_model(self, tflite_model):
        self.interpreter = tf.lite.Interpreter(model_content=tflite_model)
        self.interpreter.allocate_tensors()

    def predict(self, state, use_interpreter=True):
        self.predict_times += 1
        # print()
        # print(self.predict_times)
        # print("len state =============== {}".format(len(state)))
        # print(" self.quantization ============== {}".format(self.quantization))
        # if state[0].shape[0] != 1 or not self.quantization:
        if not self.quantization:
            # with self.graph.as_default():
            #     K.set_session(self.sess)
            #     feed_dict = {self.infer_state: state[0], self.adv: state[1]}
            #     return self.sess.run([self.infer_p, self.infer_v], feed_dict)
            # self.infer_state = state[0]
            # self.adv = state[1]
            start_train_time = time()
            # result = self.model.predict(state)
            tmp_result = self.model(state, training=False)
            result = [1, 2]
            result[0] = tmp_result[0].numpy()
            result[1] = tmp_result[1].numpy()
            end_train_time = time()
            # print("time =============== {}".format(end_train_time - start_train_time))
            return result

        # else:
        #     if not self.quantization:
        #         start_predict = time()
        #         # result = self.model.predict(state, batch_size=1)
        #         # state = numpy.asarray(state)
        #         # result = self.model.predict(x=tf.data.Dataset.from_tensors(state))
        #         tmp_result = self.model(state, training=False)
        #         result = [1, 2]
        #         result[0] = tmp_result[0].numpy()
        #         result[1] = tmp_result[1].numpy()
        #
        #         # result = np.array(result)
        #         end_predict = time()
        #         # print("predict_time ================= {}".format(end_predict - start_predict))
        #         # print(result)
        #         return result

        # print("len state ============= {}".format(len(state)))
        return self.evaluate_model(state)

    def evaluate_model(self, state):
        # -------------------- 10-5s

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.interpreter.resize_tensor_input(input_details[0]['index'], (len(state[0]), 84, 84, 4))
        self.interpreter.allocate_tensors()

        input_index0 = self.interpreter.get_input_details()[0]["index"]
        # input_index1 = self.interpreter.get_input_details()[1]["index"]

        infer_p_ph = self.interpreter.get_output_details()[0]["index"]
        infer_v_ph = self.interpreter.get_output_details()[1]["index"]
        # -------------------- 10-5s

        # state[0] = tf.convert_to_tensor(state[0])
        # state[0] = tf.cast(state[0], dtype=quint8)
        # state[1] = tf.cast(state[1], dtype=float32)
        # state[1] = state[1].astype(np.float32)
        # state[0] = state[0].astype(np.uint8)
        self.interpreter.set_tensor(input_index0, state[0])
        # self.interpreter.set_tensor(input_index1, state[1])

        # Run inference.
        self.interpreter.invoke()
        infer_p = self.interpreter.tensor(infer_p_ph)()
        infer_v = self.interpreter.tensor(infer_v_ph)()
        p_infer = deepcopy(infer_p)
        v_infer = deepcopy(infer_v)
        # print("p_infer ====================== {}".format(infer_p))
        # print("v_infer ====================== {}".format(v_infer))

        del infer_p
        del infer_v
        return p_infer, v_infer

    def get_convert_from_session(self):
        with self.graph.as_default():
            K.set_session(self.sess)
            converter = tf.lite.TFLiteConverter.from_session(self.sess, input_tensors=self.model.inputs,
                                                             output_tensors=self.model.outputs)
        return converter

    def save_keras_model(self):
        # if os.path.exists(self.keras_model_file_path):
        #     delete_local_dir(self.keras_model_file_path)
        # with self.graph.as_default():
        #     K.set_session(self.sess)
        # -----------------------------------------------30ms
        start_get = time()
        self.save_model_times += 1
        self.keras_model_file_path = self.h5_file_prefix + str(time()) + ".h5"
        tf.keras.models.save_model(self.model, self.keras_model_file_path, save_format=".h5", include_optimizer=False)
        # self.model.save_weights(self.keras_model_file_path)
        end_get = time()
        # print("self.compress_weights_queue.get() =============== {}".
        #       format(end_get - start_get))
        # tf.saved_model.save(self.model, self.keras_model_file_path)
        return self.keras_model_file_path

    def set_weights(self, weights):
        """Set weight with memory tensor."""
        if self.quantization:
            # print("set weight==================================")
            serialized_model = weights
            tflite_model = dill.loads(serialized_model)
            self.post_model(tflite_model)
        else:
            self.model.set_weights(weights)

    def get_weights(self):
        """Set weight with memory tensor."""
        if self.quantization:
            return None
        return self.model.get_weights()

    def init_interpret(self):
        self.save_keras_model()
        # converter = tf.lite.TFLiteConverter.from_keras_model_file(self.keras_model_file_path, custom_objects={"loss": loss})
        model1 = tf.keras.models.load_model(self.keras_model_file_path, custom_objects={"CustomModel": CustomModel})
        converter = tf.lite.TFLiteConverter.from_keras_model(model1)
        # converter.experimental_new_converter = True
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
        # converter.allow_custom_ops = True
        tflite_model = converter.convert()
        print("no------------------------problem")
        self.post_model(tflite_model)

    def train(self, state, label):
        # print("state_fit.shape={}; pg_adv_fit.shape={}; action_matrix_fit.shape={}; value_fit.shape={}".
        #       format(state_fit.shape, pg_adv_fit.shape, action_matrix_fit.shape, value_fit.shape))
        # print("state_fit.type={}; pg_adv_fit.type={}; action_matrix_fit.type={}; value_fit.type={}".
        #       format(state_fit.dtype, pg_adv_fit.dtype, action_matrix_fit.dtype, value_fit.dtype))
        # print(state[0].shape, state[1].shape, label[0].shape, label[1].shape)

        # loss = self.model.fit(x={'state_input': state[0], 'adv': state[1]},
        #                       y={"output_actions": label[0],
        #                          "output_value": label[1]},
        #                       batch_size=128,
        #                       verbose=0)

        # loss = self.model.train_on_batch(x={'state_input': state[0], 'adv': state[1]},
        #                                  y={"output_actions": label[0],
        #                                     "output_value": label[1]})

        # train_start = time()
        # a = tf.data.Dataset.from_tensor_slices(state[0])
        # train_end = time()
        # print("train_time ===================== {}".format(train_end - train_start))
        loss = self.model.train_step(({'state_input': state[0], 'adv': state[1]},
                                      {"output_actions": label[0], "output_value": label[1]}))

        if isinstance(loss, dict):
            return loss["loss"]
        return loss


def layer_function(x):
    """Normalize data."""
    return K.cast(x, dtype='float32') / 255.


def custom_loss(y_true, y_pred):
    policy = y_pred
    log_policy = tf.math.log(policy + 1e-10)
    entropy = -policy * tf.math.log(policy + 1e-10)
    cross_entropy = -y_true * log_policy
    return entropy, cross_entropy


def impala_loss(advantage):
    """Compute loss for impala."""

    def loss(y_true, y_pred):
        policy = y_pred
        log_policy = tf.math.log(policy + 1e-10)
        entropy = -policy * tf.math.log(policy + 1e-10)
        cross_entropy = -y_true * log_policy
        return tf.math.reduce_mean(advantage * cross_entropy - ENTROPY_LOSS * entropy, 1)

    return loss


def delete_local_dir(delete_path):
    '''
     作用: 删除本地目录
     参数：需要删除的目录
     返回：无
    '''
    path = pathlib.Path(delete_path)
    for i in path.glob("**/*"):
        # 删除文件
        if (os.path.exists(i)):
            if (os.path.isfile(i)):
                os.remove(i)

    # 将目录内容存为数组，方便排序
    a = []
    for i in path.glob("**/*"):
        a.append(str(i))

    # 降序排序后从内层开始删除
    a.sort(reverse=True)
    for i in a:
        # 删除目录
        if (os.path.exists(i)):
            if (os.path.isdir(i)):
                os.removedirs(i)
