# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE
"""
DESC: Main entrance for xingtian library.

Usage:
    python main.py -f examples/default_cases/cartpole_ppo.yaml -t train
"""

import argparse
import pprint
import yaml
from absl import logging

from xt.train import main as xt_train
from xt.train import makeup_multi_case
from xt.evaluate import main as xt_eval
from xt.benchmarking import main as xt_benchmarking
from zeus.common.util.get_xt_config import parse_xt_multi_case_paras
from zeus.common.util.get_xt_config import check_if_patch_local_node
from zeus.common.util.get_xt_config import OPEN_TASKS_SET
from zeus.common.util.logger import VERBOSITY_MAP
from xt.framework.remoter import distribute_xt_if_need
from zeus.common.util.logger import set_logging_format
set_logging_format()
# logging.set_verbosity(logging.INFO)


def main():
    """:return: config file for training or testing."""
    parser = argparse.ArgumentParser(description="XingTian Usage.")

    parser.add_argument("-f", "--config_file", required=True, help="""config file with yaml""",)
    # fixme: split local and hw_cloud,
    #  source path could read from yaml startswith s3
    parser.add_argument("-s3", "--save_to_s3", default=None, help="save model/records into s3 bucket.")
    parser.add_argument("-t", "--task", default="train", choices=list(OPEN_TASKS_SET), help="task choice to run xingtian.",)
    parser.add_argument("-v", "--verbosity", default="info", help="logging.set_verbosity")

    # revised by ZZX *begin
    parser.add_argument("-g", "--group_num", default=None)
    parser.add_argument("-e", "--env_num", default=None)
    parser.add_argument("-s", "--size", default=None)
    parser.add_argument("-w", "--wait_num", default=None)
    parser.add_argument("-b", "--speedup", default=None)
    parser.add_argument("-l", "--lock", default=1)
    parser.add_argument("-c", "--start_core", default=0)
    parser.add_argument("--gpu", default=-1)
    # revised by ZZX *end

    args, _ = parser.parse_known_args()
    if _:
        logging.warning("get unknown args: {}".format(_))

    # revised by ZZX *begin
    with open(args.config_file, "r") as conf_file:
        _info = yaml.safe_load(conf_file)
    if args.group_num is not None:  # group_num
        _info.update({"group_num": int(args.group_num)})
    if args.env_num is not None:  # env_num
        _info.update({"env_num": int(args.env_num)})

    env_para = _info.get("env_para")
    env_info = env_para.get("env_info")
    if args.size is not None:  # size
        env_info.update({"size": int(args.size)})
        env_info.update({"vector_env_size": int(args.size)})
    if args.wait_num is not None:  # wait_num
        env_info.update({"wait_num": int(args.wait_num)})
    env_para.update({"env_info": env_info})
    _info.update({"env_para": env_para})

    if args.speedup is not None:  # speedup
        _info.update({"speedup": int(args.speedup)})
    if args.lock is not None:  # lock
        _info.update({"lock": int(args.lock)})
    if args.start_core is not None:  # start_core
        _info.update({"start_core": int(args.start_core)})

    _info.update({"gpu":args.gpu})  # gpu

    root = os.path.expanduser(r'~/.xt/')
    if not os.path.isdir(root):
        os.mkdir(root)
    args.config_file = os.path.expanduser(r'~/.xt/config.yaml')
    with open(args.config_file, "w") as conf_file:
        yaml.dump(_info, conf_file)

    print('\t---------------------------------')
    print('\t| [GGLC] CONFIG/group_num  | {:2d} |'.format(_info.get("group_num", -1)))
    print('\t| [GGLC] CONFIG/env_num    | {:2d} |'.format(_info.get("env_num", -1)))
    print('\t| [GGLC] CONFIG/size       | {:2d} |'.format(_info.get("env_para").get("env_info").get("size", -1)))
    print('\t| [GGLC] CONFIG/wait_num   | {:2d} |'.format(_info.get("env_para").get("env_info").get("wait_num", -1)))
    print('\t| [GGLC] CONFIG/speedup    | {:2d} |'.format(_info.get("speedup", -1)))
    print('\t| [GGLC] CONFIG/lock       | {:2d} |'.format(_info.get("lock", -1)))
    print('\t| [GGLC] CONFIG/start_core | {:2d} |'.format(_info.get("start_core", -1)))
    print('\t| [GGLC] CONFIG/GPU        | {} |'.format(_info.get("gpu", -1)))
    print('\t| [GGLC] ADD/env_per_group | {:2d} |'.format(_info.get("env_num") // _info.get("group_num", 1)))
    print('\t---------------------------------')

    assert _info.get("env_num") % _info.get("group_num", 1) == 0, "env_num % group_num == 0"
    if _info.get("start_core", 0) % 10 != 0:
        print('\t[WARN] start_core % 10 == 0 recommended.')
    print('\t---------------------------------')
    # revised by ZZX *end

    if args.verbosity in VERBOSITY_MAP.keys():
        logging.set_verbosity(VERBOSITY_MAP[args.verbosity])
        pass
    else:
        logging.warning("un-known logging level-{}".format(args.verbosity))

    _exp_params = pprint.pformat(args, indent=0, width=1,)
    logging.info(
        "\n{}\n XT start work...\n{}\n{}".format("*" * 50, _exp_params, "*" * 50)
    )

    with open(args.config_file, "r") as conf_file:
        _info = yaml.safe_load(conf_file)

    _info = check_if_patch_local_node(_info, args.task)
    distribute_xt_if_need(config=_info, remote_env=_info.get("remote_env"))

    if args.task in ("train", "train_with_evaluate"):
        ret_para = parse_xt_multi_case_paras(args.config_file)
        if len(ret_para) > 1:
            makeup_multi_case(args.config_file, args.save_to_s3)
        else:
            xt_train(args.config_file, args.task, args.save_to_s3, args.verbosity)

    elif args.task == "evaluate":
        xt_eval(args.config_file, args.save_to_s3)

    elif args.task == "benchmark":
        # fixme: with benchmark usage in code.
        # xt_benchmark(args.config_file)
        xt_benchmarking()
    else:
        logging.fatal("Get invalid task: {}".format(args.task))


if __name__ == "__main__":
    main()
