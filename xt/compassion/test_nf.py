import tensorflow as tf
import numpy as np

logits = [[1.0, 1.0, 1.0, 6.0]]


# labels = [[0.2, 0.3, 0.5]]  # 真实值
# logits = [[2, 0.5, 1]]  # 预测值


def test_tf(logits):
    result = tf.squeeze(tf.random.categorical(logits=logits, num_samples=1, dtype=tf.int32), axis=-1)
    print("tf_result = {}".format(result))
    x = tf.one_hot(result, 4)  # action_dim
    print("tf one_hot == {}".format(x))
    y = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(x, logits)
    y = -tf.expand_dims(y, axis=-1)
    print("tf cross_entropy == {}".format(y))


def test_np(logits):
    probs = sofmax(logits)
    a = [i for i in range(len(logits[0]))]
    action_index = np.random.choice(a, p=probs[0])
    print("np result = {}".format(action_index))

    # tf.one_hot
    x = np.eye(4)[action_index]
    print("np one_hot = {}".format(x))

    # softmax_cross_entropy_with_logits_v2
    b = -cross_entropy_error(x, np.array(probs[0]))
    print("np cross_entropy === {}".format(b))
    return b


def cross_entropy_error(t, y):
    delta = 1e-7  # 添加一个微小值可以防止负无限大(np.log(0))的发生。
    return -np.sum(t * np.log(y + delta))


def sofmax(logits):
    e_x = np.exp(logits)
    probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
    return probs


if __name__ == '__main__':
    # test_tf(logits)
    # test_np(logits)
    arr = np.array(1)
    print(arr)
    k = np.expand_dims(arr, axis=0)
    k = np.expand_dims(k, axis=0)
    print(k.shape)
