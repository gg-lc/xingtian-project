import os
import sys
from time import time

from xt.model.dqn.dqn_mlp import layer_normalize, layer_add

from xt.model.muzero.default_config import HIDDEN_OUT

from xt.model.impala.default_config import LR

from xt.compassion.test_purning import layer_function, impala_loss

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from xt.structured_pruning.src import pruning
import tensorflow as tf

from tensorflow.keras import datasets, layers, models, Model, Input
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K


# print(tf.test.is_gpu_available())
# print("Load CIFAR10 Dataset as test dataset")
# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# train_images, test_images = train_images / 255.0, test_images / 255.0
#
# print("\nBuild model")
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dropout(0.3))
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dropout(0.25))
# model.add(layers.Dense(10, activation='softmax'))
# model.summary()

def model_muzero():
    obs = layers.Input(shape=(84, 84, 4), name='rep_input')
    obs_1 = layers.Lambda(lambda x: tf.cast(x, dtype='float32') / 255.)(obs)
    convlayer = layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', padding='valid')(obs_1)
    convlayer = layers.Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='valid')(convlayer)
    convlayer = layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='valid')(convlayer)
    flattenlayer = layers.Flatten()(convlayer)
    denselayer = layers.Dense(HIDDEN_OUT, activation='relu')(flattenlayer)
    # hidden = Lambda(hidden_normlize)(denselayer)
    hidden = denselayer
    return Model(inputs=obs, outputs=hidden)

def create_policy_network():
    hidden_input = layers.Input(shape=(HIDDEN_OUT, ), name='hidden_input')
    hidden = layers.Dense(128, activation='relu')(hidden_input)
    out_v = layers.Dense(4, activation='softmax')(hidden)
    out_p = layers.Dense(4, activation='softmax')(hidden)
    return Model(inputs=hidden_input, outputs=[out_p, out_v])

def impala_model():
    advantage = layers.Input(shape=(1,), name='adv')
    state_input = layers.Input(shape=(84, 84, 4), name='state_input', dtype='uint8')
    state_input_1 = layers.Lambda(layer_function)(state_input)

    convlayer = layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', padding='valid')(state_input_1)
    convlayer = layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='valid')(convlayer)
    convlayer = layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='valid')(convlayer)
    flattenlayer = layers.Flatten()(convlayer)
    denselayer = layers.Dense(256, activation='relu', name="last")(flattenlayer)

    out_actions = layers.Dense(4, activation='softmax', name='output_actions')(denselayer)
    out_value = layers.Dense(1, name='output_value')(denselayer)
    return Model(inputs=[state_input, advantage], outputs=[out_actions, out_value])

def create_dqncnn_model():
    """Create Deep-Q CNN network."""
    state = Input(shape=(84, 84, 4), dtype="uint8")
    state1 = layers.Lambda(lambda x: K.cast(x, dtype='float32') / 255.)(state)
    convlayer = layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', padding='valid')(state1)
    convlayer = layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='valid')(convlayer)
    convlayer = layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='valid')(convlayer)
    flattenlayer = layers.Flatten()(convlayer)
    denselayer = layers.Dense(256, activation='relu')(flattenlayer)
    value = layers.Dense(4, activation='linear')(denselayer)
    model = Model(inputs=state, outputs=value)
    adam = Adam(lr=0.004, clipnorm=10.)
    model.compile(loss='mse', optimizer=adam)
    return model

# source_model = impala_model()
# source_model.summary()
#
# model = Model(inputs=source_model.get_layer("state_input").input, outputs=source_model.get_layer("last").output)
# out_actions = layers.Dense(4, activation='softmax', name='output_actions')(model.get_layer("last").output)
# out_value = layers.Dense(1, name='output_value')(model.get_layer("last").output)

# losses = {"output_actions": impala_loss(advantage), "output_value": 'mse'}
# lossweights = {"output_actions": 1.0, "output_value": .5}

# self.losses = losses

decay_value = 0.00000000512

# model.compile(optimizer=Adam(lr=LR, clipnorm=40., decay=decay_value))
#
# print("\nCompile, train and evaluate model")
# comp = {
#     "optimizer": 'adam',
#     "loss": tf.keras.losses.SparseCategoricalCrossentropy(),
#     "metrics": ['accuracy']}

# model.compile(**comp)
# callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]

# model.fit(train_images, train_labels, validation_split=0.2, epochs=1, batch_size=128, callbacks=callbacks)

# model_test_loss, model_test_acc = model.evaluate(test_images, test_labels, verbose=2)
# print(f"Model accuracy after Training: {model_test_acc * 100:.2f}%")

# print("\nTest factor pruning")
# dense_prune_rate = 30
# conv_prune_rate = 40
# start = time()
# pruned_model = pruning.factor_pruning(model, dense_prune_rate, conv_prune_rate, 'L2', num_classes=10)
# end = time()
# print("factor_pruning time ============{}".format(end - start))
#
#
# print("\nCompile, retrain and evaluate pruned model")
# comp = {
#     "optimizer": 'adam',
#     "loss": tf.keras.losses.SparseCategoricalCrossentropy(),
#     "metrics": ['accuracy']}
#
# pruned_model.compile(**comp)
#
# out_actions = source_model.get_layer("output_actions")(pruned_model.get_layer("last").output)
# out_value = source_model.get_layer("output_value")(pruned_model.get_layer("last").output)
# result_model = Model(inputs=pruned_model.get_layer("state_input").input, outputs=[out_actions, out_value])
# print("result_model================")
# result_model.summary()


def pruning_model(source_model):
    model = Model(inputs=source_model.get_layer("state_input").input, outputs=source_model.get_layer("last").output)
    dense_prune_rate = 30
    conv_prune_rate = 40
    pruned_model = pruning.factor_pruning(model, dense_prune_rate, conv_prune_rate, 'L2', num_classes=10)
    out_actions = source_model.get_layer("output_actions")(pruned_model.get_layer("last").output)
    out_value = source_model.get_layer("output_value")(pruned_model.get_layer("last").output)
    result_model = Model(inputs=pruned_model.get_layer("state_input").input, outputs=[out_actions, out_value])
    # print("result_model================")
    # result_model.summary()
    return result_model

# pruning_model(source_model)

# pruned_model.fit(train_images, train_labels, epochs=1, validation_split=0.2)

# print("\nCompare model before and after pruning")
# model_test_loss, model_test_acc = model.evaluate(test_images, test_labels, verbose=2)
# pruned_model_test_loss, pruned_model_test_acc = pruned_model.evaluate(test_images, test_labels, verbose=2)
# print(f"Model accuracy before pruning: {model_test_acc * 100:.2f}%")
# print(f"Model accuracy after pruning: {pruned_model_test_acc * 100:.2f}%")

# print(f"\nTotal number of parameters before pruning: {model.count_params()}")
# print(f"Total number of parameters after pruning: {pruned_model.count_params()}")
# print(
#     f"Pruned model contains only {(pruned_model.count_params() / model.count_params()) * 100:.2f} % of the original number of parameters.")
#
# print("\nTest accuracy pruning")
# comp = {
#     "optimizer": 'adam',
#     "loss": tf.keras.losses.SparseCategoricalCrossentropy(),
#     "metrics": 'accuracy'
# }

# auto_model = pruning.accuracy_pruning(model, comp, train_images, train_labels, test_images,
#                                       test_labels, pruning_acc=None, max_acc_loss=3,
#                                       num_classes=10, label_one_hot=False)
#
# print("\nCompare model before and after pruning")
# model_test_loss, model_test_acc = model.evaluate(test_images, test_labels, verbose=2)
# auto_model_test_loss, auto_model_test_acc = auto_model.evaluate(test_images, test_labels, verbose=2)
# print(f"Model accuracy before pruning: {model_test_acc * 100:.2f}%")
# print(f"Model accuracy after pruning: {auto_model_test_acc * 100:.2f}%")
#
# print(f"\nTotal number of parameters before pruning: {model.count_params()}")
# print(f"Total number of parameters after pruning: {auto_model.count_params()}")
# print(
#     f"Pruned model contains only {(auto_model.count_params() / model.count_params()) * 100:.2f} % of the original number of parameters.")

if __name__ == '__main__':
    pass
