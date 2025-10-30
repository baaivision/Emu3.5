#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: logging_utils.py
# Created Date: Wednesday December 18th 2024
# Author: Zhengxiong Luo
# Contact: <zxluo@baai.ac.cn>
#
# Last Modified: Wednesday December 18th 2024 3:03:32 pm
#
# Copyright (c) 2024 Beijing Academy of Artificial Intelligence (BAAI)
# All rights reserved.
# -----
# HISTORY:
# Date      	 By	Comments
# ----------	---	----------------------------------------------------------
###
import os
from datetime import datetime
import os.path as osp
import builtins
old_print = builtins.print


def setup_print_file(file):
    def print(*args, **kwargs):
        msg = " ".join(map(str, args))
        with open(file, "a") as f:
            f.write(msg + "\n")
        old_print(msg)

    builtins.print = print


def setup_logger(log_dir="./", log_name="log"):
    logfile = osp.join(
        log_dir,
        f'{log_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    )
    os.makedirs(osp.dirname(logfile), exist_ok=True)
    setup_print_file(logfile)
