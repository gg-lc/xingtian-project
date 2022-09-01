### 项目说明
1. 本项目为协作release版本，综合集中了XingTian项目的各项进展。推荐使用`Git`↔`PyCharm`↔`remote`方式协作。
2. 更新流程：
   1. 从master或所需分支**创建新的分支**并命名（如dev_pipeline）
   2. 对**本地分支**（dev_pipeline）进行**修改**
      1. 需要在README中**简要说明代码修改情况**
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

### 2022.xx.xx 更新：添加`pipeline_impala`分支
1. 更新了envpool及分组流水线采样(pipeline)实现
   * 以下给出了128和140两个服务器的较优配置文件，可直接运行
     * [examples_1](examples/breakout_impala_pipeline_128opt.yaml)
     * [examples_2](examples/breakout_impala_pipeline_140opt.yaml)
   * 使用pipeline参数配置（yaml文件）
     * `env_name`: EnvPool
     * `vector_env_size`: 单个envpool大小，即环境并行数量
     * `size`: 等同于vector_env_size，这两项设定应当相同
     * `wait_num`: envpool参数，实现异步并行采样
     * `env_num`: envpool的总数量，应当是group_num的整数倍
     * `group_num`: 将envpool划分成的组数，每一组的envpool将绑定在一组CPU核心进行采样
     * `speedup`: 3，绑核方式，0为不绑核，1为顺序绑核，2为均衡化绑核，3为pipeline绑核
   * 命令行参数（可选）
     * `--gpu n`: 指定使用编号为n的GPU进行推理，默认使用CPU推理
     * `-b n`: 等同yaml文件的speedup参数，优先级更高
     * `-s n`: 等同yaml文件的size参数，优先级更高
     * `-w n`: 等同yaml文件的wait_num参数，优先级更高
     * `-e n`: 等同yaml文件的env_num参数，优先级更高
     * `-g n`: 等同yaml文件的group_num参数，优先级更高
3. 代码修改（所有修改内容均添加了"ZZX"标记，可进行全局搜索）



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