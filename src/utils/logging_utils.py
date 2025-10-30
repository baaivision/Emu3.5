# -*- coding:utf-8 -*-
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

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
