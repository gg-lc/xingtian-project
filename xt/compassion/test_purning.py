import multiprocessing
import pickle
import sys
import tempfile
import os
from time import time

import dill
import numpy as np

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.utils.vis_utils import plot_model
from xt.model.impala.default_config import ENTROPY_LOSS, LR

print("tensorflow.__version ================== {}".format(tf.__version__))

from tensorflow.keras import Input, Model

from tensorflow.keras.layers import Dense, Conv2D, Lambda, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam

from xt.structured_pruning.src import pruning

# os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

# %load_ext tensorboard

def structed_prun():
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(28, 28)),
        keras.layers.Reshape(target_shape=(28, 28, 1)),
        keras.layers.Conv2D(filters=1000, kernel_size=(3, 3), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=(2, 2), ),
        keras.layers.Flatten(),
        keras.layers.Dense(10)
    ])


def complex():
    import numpy as np
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the input image so that each pixel value is between 0 to 1.
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Define the model architecture.
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(28, 28)),
        keras.layers.Reshape(target_shape=(28, 28, 1)),
        keras.layers.Conv2D(filters=1000, kernel_size=(3, 3), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=(2, 2), ),
        keras.layers.Flatten(),
        keras.layers.Dense(10)
    ])

    # Train the digit classification model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(
        train_images,
        train_labels,
        epochs=3,
        validation_split=0.1,
    )

    # model.summary()

    baseline_model_start = time()
    # _, baseline_model_accuracy = model.evaluate(
    #     test_images, test_labels, verbose=0)
    baseline_model_end = time()

    test_size = 0
    # baseline_model_split_start = time()
    # for test_image, test_label in zip(test_images, test_labels):
    #     # test_image = image.img_to_array(test_image)
    #     test_size += 1
    #     test_image = np.expand_dims(test_image, axis=0)
    #     model.predict(test_image)
    #     if test_size == 1000:
    #         print("evaluate : {}".format(test_size))
    # baseline_model_split_end = time()

    # print("baseline_model 推理时间为 ============================== {}".
    #       format(baseline_model_end - baseline_model_start))

    # print("baseline_model_split_time 推理时间为 ============================== {}".
    #       format(baseline_model_split_end - baseline_model_split_start))

    # print('Baseline test accuracy:', baseline_model_accuracy)

    _, keras_file = tempfile.mkstemp('.h5')
    # keras_file = "base_model" + str(time()) + ".h5"
    tf.keras.models.save_model(model, keras_file, include_optimizer=False)
    print('Saved baseline model to:', keras_file)

    import tensorflow_model_optimization as tfmot

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    # Compute end step to finish pruning after 2 epochs.
    batch_size = 128
    epochs = 1
    validation_split = 0.1  # 10% of training set will be used for validation set.

    num_images = train_images.shape[0] * (1 - validation_split)
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

    # Define model for pruning.
    # pruning_params = {
    #     'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
    #                                                              final_sparsity=0.60,
    #                                                              begin_step=0,
    #                                                              end_step=end_step)
    # }

    pruning_params_sparsity_0_5 = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=0.5,
                                                                  begin_step=0,
                                                                  frequency=100)
    }

    model_for_pruning = prune_low_magnitude(model, **pruning_params_sparsity_0_5)

    # `prune_low_magnitude` requires a recompile.
    model_for_pruning.compile(optimizer='adam',
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                              metrics=['accuracy'])

    model_for_pruning.summary()

    logdir = tempfile.mkdtemp()

    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
    ]

    # model_for_pruning.fit(train_images, train_labels,
    #                       batch_size=batch_size, epochs=epochs, validation_split=validation_split,
    #                       callbacks=callbacks)

    model_for_pruning_start = time()
    _, model_for_pruning_accuracy = model_for_pruning.evaluate(
        test_images, test_labels, verbose=0)
    model_for_pruning_end = time()

    model_for_pruning_part_start = time()
    for i, test_image in enumerate(test_images):
        x = np.expand_dims(test_image, axis=0).astype(np.float32)
        model_for_pruning.predict(x)
        if i == 100:
            break
    model_for_pruning_part_end = time()

    # print("model_for_pruning 推理时间为=========================={}".
    #       format(model_for_pruning_end - model_for_pruning_start))

    # print('Baseline test accuracy:', baseline_model_accuracy)
    print('Pruned test accuracy:', model_for_pruning_accuracy)

    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    _, pruned_keras_file = tempfile.mkstemp('.h5')
    # pruned_keras_file = "pruned_keras_file" + str(time()) + ".h5"
    tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
    print('Saved pruned Keras model to:', pruned_keras_file)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    model_for_pruning_start = time()
    pruned_tflite_model = converter.convert()
    model_for_pruning_end = time()
    print("convert 时间为 ================== {}".format(model_for_pruning_end - model_for_pruning_start))

    _, pruned_tflite_file = tempfile.mkstemp('.tflite')

    with open(pruned_tflite_file, 'wb') as f:
        f.write(pruned_tflite_model)

    print('Saved pruned TFLite model to:', pruned_tflite_file)

    def get_gzipped_model_size(file):
        # Returns size of gzipped model, in bytes.
        import os
        import zipfile

        _, zipped_file = tempfile.mkstemp('.zip')
        with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
            f.write(file)

        return os.path.getsize(zipped_file)

    print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
    print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))
    print("Size of gzipped pruned TFlite model: %.2f bytes" % (get_gzipped_model_size(pruned_tflite_file)))

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]  # 定义模型的精度

    quantized_and_pruned_tflite_model = converter.convert()

    a = pickle.dumps(quantized_and_pruned_tflite_model)
    b = pickle.loads(a)
    quantized_and_pruned_tflite_model = a

    _, quantized_and_pruned_tflite_file = tempfile.mkstemp('.tflite')

    with open(quantized_and_pruned_tflite_file, 'wb') as f:
        f.write(quantized_and_pruned_tflite_model)

    print('Saved quantized and pruned TFLite model to:', quantized_and_pruned_tflite_file)

    print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
    print(
        "Size of gzipped pruned and quantized TFlite model: %.2f bytes" %
        (get_gzipped_model_size(quantized_and_pruned_tflite_file)))

    import numpy as np

    def evaluate_model(interpreter):
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]

        # Run predictions on ever y image in the "test" dataset.
        prediction_digits = []
        for i, test_image in enumerate(test_images):
            if i % 1000 == 0:
                print('Evaluated on {n} results so far.'.format(n=i))
            # Pre-processing: add batch dimension and convert to float32 to match with
            # the model's input data format.
            test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
            interpreter.set_tensor(input_index, test_image)
            # print(i)
            # Run inference.
            interpreter.invoke()
            sb_times = sys.getrefcount(interpreter)
            # print("sb_times ====================== {}".format(sb_times))
            # Post-processing: remove batch dimension and find the digit with highest
            # probability.
            output = interpreter.tensor(output_index)
            k = output()[0]
            fu(k)
            del k
            digit = np.argmax(output()[0])
            prediction_digits.append(digit)

        print('\n')
        # Compare prediction results with ground truth labels to calculate accuracy.
        prediction_digits = np.array(prediction_digits)
        accuracy = (prediction_digits == test_labels).mean()
        return accuracy

    def fu(a):
        pass

    def evaluate_predict(interpreter):
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]

        # Run predictions on ever y image in the "test" dataset.
        prediction_digits = []
        for i, test_image in enumerate(test_images):
            if i == 100:
                break
            # Pre-processing: add batch dimension and convert to float32 to match with
            # the model's input data format.
            test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
            interpreter.set_tensor(input_index, test_image)

            # Run inference.
            interpreter.invoke()

            # Post-processing: remove batch dimension and find the digit with highest
            # probability.
            output = interpreter.tensor(output_index)
            digit = np.argmax(output()[0])
            prediction_digits.append(digit)

    interpreter = tf.lite.Interpreter(model_content=quantized_and_pruned_tflite_model)
    interpreter.allocate_tensors()

    quantized_and_pruned_tflite_model_start = time()
    test_accuracy = evaluate_model(interpreter)
    quantized_and_pruned_tflite_model_end = time()
    # evaluate_predict(interpreter)

    print("quantized_and_pruned_tflite_model_time 推理时间为 =========================== {}".
          format(quantized_and_pruned_tflite_model_end - quantized_and_pruned_tflite_model_start))

    print('Pruned and quantized TFLite test_accuracy:', test_accuracy)
    print('Pruned TF test accuracy:', model_for_pruning_accuracy)


def simple():
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(28, 28)),
        keras.layers.Reshape(target_shape=(28, 28, 1)),
        keras.layers.Conv2D(filters=1000, kernel_size=(3, 3), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=(2, 2), ),
        keras.layers.Flatten(),
        keras.layers.Dense(10)
    ])

    # Train the digit classification model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    print("================sample=========================")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    #
    # # converter = tf.lite.TFLiteConverter.from_keras_model_file("/home/xys/primary_xingtian/xingtian/ppo_model1656766461.5305722.h5")
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.float32]  # 定义模型的精度
    #
    # quantized_and_pruned_tflite_model = converter.convert()
    return model


def sample():
    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file("fffffff_model.h5")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # # converter.target_spec.supported_types = [tf.float32]  # 定义模型的精度
    #
    quantized_and_pruned_tflite_model = converter.convert()
    #

    a = pickle.dumps(quantized_and_pruned_tflite_model)
    b = pickle.loads(a)

    quantized_and_pruned_tflite_model = b
    interpreter = tf.lite.Interpreter(model_content=quantized_and_pruned_tflite_model)
    interpreter.allocate_tensors()
    print("post_model OK======================= ")


class Fuck1:
    def __init__(self):
        pass

    def fuck1(self):
        print("fuck")


class Fuck:
    def __init__(self):
        self.nice = Fuck1()
        self.q = multiprocessing.Queue()
        pass

    def fuck1(self):
        print("fuc1k")

    def que(self):
        self.q.put(self.nice)


def fuck(a):
    k = a.get()
    k.fuck1()


def test_keras_model():
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import Dense, Conv2D, Lambda, Flatten
    from tensorflow.keras.optimizers import Adam

    state_input = Input(shape=(84, 84, 4), name='state_input', dtype='uint8')
    state_input_1 = Lambda(layer_function)(state_input)
    advantage = Input(shape=(1,), name="adv")

    convlayer = Conv2D(32, (8, 8), strides=(4, 4), activation='relu', padding='valid')(state_input_1)
    convlayer = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='valid')(convlayer)
    convlayer = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='valid')(convlayer)
    flattenlayer = Flatten()(convlayer)
    denselayer = Dense(256, activation='relu')(flattenlayer)

    out_actions = Dense(4, activation='softmax', name='output_actions')(denselayer)
    out_value = Dense(1, name='output_value')(denselayer)
    model = Model(inputs=[state_input, advantage], outputs=[out_actions, out_value])
    # losses = {"output_actions": impala_loss(advantage), "output_value": 'mse'}
    # advantage_data = []
    losses = {"output_actions": impala_loss(advantage), "output_value": 'mse'}

    lossweights = {"output_actions": 1.0, "output_value": .5}

    # self.losses = losses

    decay_value = 0.00000000512

    model.compile(optimizer=Adam(lr=LR, clipnorm=40., decay=decay_value),
                  loss=losses, loss_weights=lossweights)

    state = [1, 2]
    label = [1, 2]
    state[0] = np.zeros(shape=[128, 84, 84, 4])
    state[1] = np.zeros(shape=[128, 1])
    label[0] = np.zeros(shape=[128, 1])
    label[1] = np.zeros(shape=[128, 1])

    loss = model.fit(x={'state_input': state[0], 'adv': state[1]},
                     y={"output_actions": label[0],
                        "output_value": label[1]},
                     batch_size=128,
                     verbose=0)


def test_keras_model2():
    pass


def layer_function(x):
    """Normalize data."""
    return tf.cast(x, dtype='float32') / 255.


def impala_loss(advantage):
    """Compute loss for impala."""

    def loss(y_true, y_pred):
        policy = y_pred
        log_policy = tf.math.log(policy + 1e-10)
        entropy = -policy * tf.math.log(policy + 1e-10)
        cross_entropy = -y_true * log_policy
        return tf.math.reduce_mean(advantage * cross_entropy - ENTROPY_LOSS * entropy, 1)

    return loss


def test_load_model():
    import tensorflow as tf
    while True:
        model1 = tf.keras.models.load_model("/home/xys/primary_xingtian/xingtian/explorer1657725775.4454427")
        converter = tf.lite.TFLiteConverter.from_keras_model(model1)
        tflite_model = converter.convert()
        serialize_tflite_model = dill.dumps(tflite_model)
        print("okoooooooooooooooooooooooo")

if __name__ == '__main__':
    q = multiprocessing.Queue()
    q.put("adf")
    q.put("adf")
    q.get()
    print(q.qsize())
