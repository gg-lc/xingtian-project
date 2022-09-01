import os
from time import time

import paddle
import paddle.vision.models as models
from paddle.static import InputSpec as Input
from paddle.vision.datasets import Cifar10
import paddle.vision.transforms as T
from paddleslim.dygraph import L1NormFilterPruner

net = models.mobilenet_v1()
inputs = Input(shape=[None, 3, 32, 32], dtype='float32', name='image')
labels = Input(shape=[None, 1], dtype='int64', name='label')
optmizer = paddle.optimizer.Momentum(learning_rate=0.1, parameters=net.parameters())
model = paddle.Model(net, inputs, labels)
model.prepare(
    optimizer=optmizer,
    loss=paddle.nn.CrossEntropyLoss(),
    metrics=paddle.metric.Accuracy(topk=(1, 5))
)

transforms = T.Compose([
    T.Transpose(),
    T.Normalize([127.5], [127.5])
])

train_dataset = Cifar10(mode='train', transform=transforms)
test_dataset = Cifar10(mode='train', transform=transforms)

model.fit(train_dataset, epochs=1, batch_size=128, verbose=1)
flops = paddle.flops(net, input_size=[1, 3, 32, 32], print_detail=True)
# paddle.device.set_device('cpu')
model_start = time()
model.evaluate(test_dataset, batch_size=128, verbose=1)
model_end = time()

pruner = L1NormFilterPruner(net, [1, 3, 32, 32])
pruner.prune_vars({'conv2d_22.w_0': 0.5, 'conv2d_20.w_0': 0.6}, axis=0)
flops = paddle.flops(net, input_size=[1, 3, 32, 32], print_detail=True)

print("start after prun ...")
prun_model_start = time()
model.evaluate(test_dataset, batch_size=128, verbose=1)
prun_model_end = time()

print("model evaluate time ============== {}".format(model_end - model_start))
print("prun_model evaluate time ============== {}".format(prun_model_end - prun_model_start))

optimizer = paddle.optimizer.Momentum(
    learning_rate=0.1,
    parameters=net.parameters())

model.prepare(
    optimizer,
    paddle.nn.CrossEntropyLoss(),
    paddle.metric.Accuracy(topk=(1, 5)))

model.fit(train_dataset, epochs=1, batch_size=128, verbose=1)

model.evaluate(test_dataset, batch_size=128, verbose=1)

if __name__ == '__main__':
    pass
