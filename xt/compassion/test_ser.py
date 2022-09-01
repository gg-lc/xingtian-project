import multiprocessing
import pickle
import queue
import time

from multiprocessing import queues


class Food:
    def __init__(self, name, origin, calories, price):
        self.name = name
        self.origin = origin  # 产地
        self.calories = calories  # 卡路里
        self.price = price

    def fuck(self):
      print("fuck===")




import gc

'''
第59条: 用tracemalloc来掌握内存的使用及泄露情况
关键:
1 python内存管理
本质: 通过引用计数来 + 循环检测器
作用: 
1)当某个对象全部引用过期的时候，被引用对象可以得到清理
2) 来即回收机制能把自我引用对象清除
2 调试内存使用状况
方法1: 使用gc模块进行查询，列出来即收集器当前所知的每个对象
import gc
objs = gc.get_objects()
python2还可以使用heapy包等追踪内存使用量
方法2: python3.4推出新的内置模块，叫做tracemalloc
    可以把某个对象与该对象的内存分配点联系起来
用法:
import tracemalloc
tracemalloc.start(10)
time1 = tracemalloc.take_snapshow()
import waste_memory
x = waste_memory.run()
time2 = tracemalloc.take_snapshow()
stats = time2.compare_to(time1, 'lineno')
for stat in stats[:3]:
    print stat
参考:
Effectiv Python 编写高质量Python代码的59个有效方法
'''


def useGC():
    objs = gc.get_objects()
    print
    "%d objects before" % (len(objs))


def process():
    useGC()


if __name__ == "__main__":
    a = multiprocessing.Queue(maxsize=2)
    a.put(1)
    a.put(2)
    a.put(3)
    print(a.get())
