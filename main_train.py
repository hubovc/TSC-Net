import torch
from easydict import EasyDict
import yaml
from models.processor import *
import os
import sys

if len(sys.argv) > 1:
    epoch = int(sys.argv[1])
else:
    epoch = 0

with open('configs/tsc.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config = EasyDict(config)



train_tsc_net(config, epoch )
