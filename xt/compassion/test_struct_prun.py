# import tensorflow as tf
# from tensorflow import keras
#
# import tensorflow_model_optimization as tfmot
# prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
#
# # Load MNIST dataset.
# mnist = keras.datasets.mnist
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#
# # Normalize the input image so that each pixel value is between 0 and 1.
# train_images = train_images / 255.0
# test_images = test_images / 255.0
#
# pruning_params_2_by_4 = {
#     'sparsity_m_by_n': (2, 4),
# }
#
# pruning_params_sparsity_0_5 = {
#     'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=0.5,
#                                                               begin_step=0,
#                                                               frequency=100)
# }
#
# model = keras.Sequential([
#     prune_low_magnitude(
#         keras.layers.Conv2D(
#             32, 5, padding='same', activation='relu',
#             input_shape=(28, 28, 1),
#             name="pruning_sparsity_0_5"),
#         **pruning_params_sparsity_0_5),
#     keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
#     prune_low_magnitude(
#         keras.layers.Conv2D(
#             64, 5, padding='same',
#             name="structural_pruning"),
#         **pruning_params_2_by_4),
#     keras.layers.BatchNormalization(),
#     keras.layers.ReLU(),
#     keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
#     keras.layers.Flatten(),
#     prune_low_magnitude(
#         keras.layers.Dense(
#             1024, activation='relu',
#             name="structural_pruning_dense"),
#         **pruning_params_2_by_4),
#     keras.layers.Dropout(0.4),
#     keras.layers.Dense(10)
# ])
#
# model.compile(optimizer='adam',
#               loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# model.summary()
#
# batch_size = 128
# epochs = 2
#
# model.fit(
#     train_images,
#     train_labels,
#     batch_size=batch_size,
#     epochs=epochs,
#     verbose=0,
#     callbacks=tfmot.sparsity.keras.UpdatePruningStep(),
#     validation_split=0.1)
#
# _, pruned_model_accuracy = model.evaluate(test_images, test_labels, verbose=0)
# print('Pruned test accuracy:', pruned_model_accuracy)
#
# model = tfmot.sparsity.keras.strip_pruning(model)
#
# model.summary()
#
# import tempfile
#
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
#
# _, tflite_file = tempfile.mkstemp('.tflite')
# print('Saved converted pruned model to:', tflite_file)
# with open(tflite_file, 'wb') as f:
#   f.write(tflite_model)