#  #Copyright 2019 Korea University under XAI Project supported by Ministry of Science and ICT, Korea
#
#  #Licensed under the Apache License, Version 2.0 (the "License");
#  #you may not use this file except in compliance with the License.
#  #You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
#  #Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import argparse
import os

import GPUtil
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.set_defaults(train=True)
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--test', dest='train', action='store_false')
parser.add_argument('--GPU', type=int, default=-1)
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--C', type=int, default=2)
parser.add_argument('--K', type=int, default=3)
ARGS = parser.parse_args()

is_train = ARGS.train
GPU = ARGS.GPU
seed = ARGS.seed
C_way = ARGS.C
K_shot = ARGS.K

if seed != -1:
    np.random.seed(seed)
    tf.compat.v1.random.set_random_seed(seed)

if GPU == -1:
    devices = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
else:
    devices = "%d" % GPU
os.environ["CUDA_VISIBLE_DEVICES"] = devices

project_path = "/home/jsyoon/project/XAIOS/"
result_path = project_path + "summ/"
data_path = os.path.join(project_path, "data/")

in_feat = 93

rank = 1
weight_decay = 0.00005
weight_const = 2.5

data_type = "float32"

epoch = 250
batch_size = 16
iter_cnt = 1000
