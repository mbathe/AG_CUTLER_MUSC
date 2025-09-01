# Copyright (c) Meta Platforms, Inc. and affiliates.

from . import config
from . import engine
from . import modeling
from . import structures
from . import tools
from . import demo

# dataset loading
from . import data  # register all new datasets
from .data import datasets  # register all new datasets
from .solver import *

# from .data import register_all_imagenet