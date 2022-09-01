"""
 * Created with PyCharm
 * 作者: 阿光
 * 日期: 2022/1/1
 * 时间: 19:32
 * 描述:
"""
import time

import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import *

num_tags = 12
num_words = 10000
num_departments = 4

# 输入源
title_input = keras.Input(shape=(None,), name='title')
body_input = keras.Input(shape=(None,), name='body')
tags_input = keras.Input(shape=(num_tags,), name='tags')

title_features = Embedding(num_words, 64)(title_input)
body_features = Embedding(num_words, 64)(body_input)

title_features = LSTM(128)(title_features)
body_features = LSTM(128)(body_features)

x = Concatenate(axis=1)([title_features, body_features, tags_input])

# 输出源
priority_pred = Dense(1, name='priority')(x)
department_pred = Dense(num_departments, name='department')(x)

model = keras.Model(
    inputs=[title_input, body_input, tags_input],
    outputs=[priority_pred, department_pred]
)

# keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={
        'priority': keras.losses.BinaryCrossentropy(from_logits=True),
        'department': keras.losses.CategoricalCrossentropy(from_logits=True)
    },
    loss_weights=[1.0, 0.2]
)

title_data = np.random.randint(num_words, size=(1280, 10))
body_data = np.random.randint(num_words, size=(1280, 100))
tags_data = np.random.randint(2, size=(1280, num_tags)).astype('float32')

priority_targets = np.random.random(size=(1280, 1))
department_targets = np.random.randint(2, size=(1280, num_departments))

# model.fit(
#     {'title': title_data, 'body': body_data, 'tags': tags_data},
#     {'priority': priority_targets, 'department': department_targets},
#     epochs=2,
#     batch_size=32
# )

start_save = time.time()
model.save_weights("fuck123.h5")
end_save = time.time()
print(end_save - start_save)

if __name__ == '__main__':
    pass