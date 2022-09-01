### 项目说明
1. 本项目为协作release版本，综合集中了XingTian项目的各项进展。推荐使用`Git`↔`PyCharm`↔`remote`方式协作。
2. 更新流程：
   1. 从master或所需分支**创建新的分支**并命名（如dev_pipeline）
   2. 对**本地分支**（dev_pipeline）进行**修改**
      1. 需要在README中**详细说明代码修改/增删位置**
      2. 需在增删/修改的代码处**添加注释**，并注明修改人
      3. 涉及到**功能性更新**需在相应分支的README中对更新功能进行详细的阐述，
      包括更新介绍、配置方式、注意事项等。参考2022.03.12更新
   3. **commit & push**
   4. **更新master分支的说明文档README.md**
   5. 如需将公共分支的更新**合并到个人开发分支**，使用merge
   6. 如需将个人开发分支的更新**合并到公共分支**，**讨论**后合并

---
### 2022.xx.xx 更新：添加`quantized`分支
1. 留空，暂未commit & push
2. 模型量化推理

### 2022.xx.xx 更新：添加`pipeline`分支
1. 留空，暂未commit & push
2. 分组流水线采样

### 2022.03.12 更新：添加`balanced_core_binding`分支
1. 修改[broker.py](xt/framework/broker.py)以实现均衡化绑核
   * [line #482](https://github.com/gg-lc/xingtian-project/blob/balanced_core_binding/xt/framework/broker.py#L482)
   * [line #499](https://github.com/gg-lc/xingtian-project/blob/balanced_core_binding/xt/framework/broker.py#L499)
2. 绑核方式：explorer绑核方式支持三种（不绑定内核、顺序绑定、均衡化绑定）
   * **不绑定内核**：由操作系统调度
   * **顺序绑定**：刑天默认绑核策略
   * **均衡化绑定**：优先将explorer分布到不同的socket，
     在每个socket中的explorer优先绑定到不同的物理核。
3. 绑核方式指定：通过yaml配置文件，指定`speedup`数值，`0`表示不绑定内核(默认)；
   `1`表示进行顺序绑定；`2`表示使用均衡化绑核策略。

   示例：[examples/balanced_core_binding/*.yaml](examples/balanced_core_binding)

4. 说明：本分支仅修改了对explorer的绑核策略，暂未修改evaluator绑核策略。
5. 问题：源码[broker.py#562](https://github.com/gg-lc/xingtian-project/blob/balanced_core_binding/xt/framework/broker.py#L562) `core_set += 1`对局部变量加一，语句无效，不能有效地对evaluator绑核。