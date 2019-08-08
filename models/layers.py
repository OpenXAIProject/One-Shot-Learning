
#  #Copyright 2019 Korea University under XAI Project supported by Ministry of Science and ICT, Korea
#
#  #Licensed under the Apache License, Version 2.0 (the "License");
#  #you may not use this file except in compliance with the License.
#  #You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
#  #Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import tensorflow as tf
import config
l = tf.keras.layers

def dense(f, act=None, b=True, k_init="he_normal", k_reg=config.weight_decay, k_const=config.weight_const):
    if k_reg:
        k_reg = tf.keras.regularizers.l2(k_reg)
    if k_const:
        k_const = tf.keras.constraints.MaxNorm(max_value=k_const, axis=[0])

    return l.Dense(units=f, activation=act, use_bias=b, kernel_initializer=k_init,
                   bias_initializer='zeros', kernel_regularizer=k_reg, kernel_constraint=k_const)

def conv(f, k=3, s=1, p='same', act=None, b=True, k_init="he_normal", k_reg=config.weight_decay,
         k_const=config.weight_const, rank=config.rank):
    if rank==3:
        conv_layer = l.Conv3D
    elif rank==2:
        conv_layer = l.Conv2D
    elif rank==1:
        conv_layer = l.Conv1D
    else:
        raise Exception("Conv Rank Error!")

    if k_reg:
        k_reg = tf.keras.regularizers.l2(k_reg)
    if k_const:
        k_const = tf.keras.constraints.MaxNorm(max_value=k_const, axis=[0, 1, 2, 3])

    return conv_layer(filters=f, kernel_size=k, strides=s, padding=p, activation=act, use_bias=b,
                         kernel_initializer=k_init, bias_initializer='zeros', kernel_regularizer=k_reg, kernel_constraint=k_const)

def conv_transpose(f, k=4, s=2, p='same', out_p="auto", act=None, b=True, k_init="he_normal", k_reg=config.weight_decay,
                   k_const=config.weight_const, rank=config.rank):
    if rank==3:
        conv_transpose_layer = l.Conv3DTranspose
    elif rank==2:
        conv_transpose_layer = l.Conv2DTranspose
    else:
        raise Exception("Conv Transpose Rank Error!")

    if k_reg:
        k_reg = tf.keras.regularizers.l2(k_reg)
    if k_const:
        k_const = tf.keras.constraints.MaxNorm(max_value=k_const, axis=[0, 1, 2, 3])

    if out_p=="auto":
        out_p = (config.W%s,config.H%s,config.D%s)

    return conv_transpose_layer(filters=f, kernel_size=k, strides=s, padding=p, output_padding=out_p,
                                  activation=act, use_bias=b, kernel_initializer=k_init, bias_initializer='zeros',
                                  kernel_regularizer=k_reg, kernel_constraint=k_const)


def maxpool(k=2, s=2, p="same", rank=config.rank):
    if rank==3:
        maxpool_layer = l.MaxPool3D
    elif rank==2:
        maxpool_layer = l.MaxPool2D
    elif rank==1:
        maxpool_layer = l.MaxPool1D
    else:
        raise Exception("MaxPool Rank Error!")

    return maxpool_layer(pool_size=k, strides=s, padding=p)

def global_avgpool(rank=config.rank):
    if rank==3:
        global_avgpool_layer = l.GlobalAveragePooling3D
    elif rank==2:
        global_avgpool_layer = l.GlobalAveragePooling2D
    elif rank==1:
        global_avgpool_layer = l.GlobalAveragePooling1D
    else:
        raise Exception("global_avgpool Rank Error!")

    return global_avgpool_layer()

def crop(size, mode="edge"):
    if len(size)==3:
        crop_layer = l.Cropping3D
    elif len(size)==2:
        crop_layer = l.Cropping2D
    elif len(size)==1:
        crop_layer = l.Cropping1D
    else:
        raise Exception("Crop Rank Error!")

    if mode=="edge":
        cropping= [[s//2, s-s//2] for s in size]
    elif mode=="front":
        cropping = [[s, 0] for s in size]
    elif mode=="end":
        cropping = [[0, s] for s in size]
    else:
        raise Exception("Crop Mode Error!")
    return crop_layer(cropping=cropping)

def flatten():
    return l.Flatten()

def Dropout(rate=0.2):
    return l.Dropout(rate=rate)

def concat(axis=-1):
    return l.Concatenate(axis=axis)


def batch_norm(m=0.99, e=1e-3):
    return l.BatchNormalization(momentum=m, epsilon=e)

def relu():
    return l.ReLU()

def in_layer(name=None, in_shape=(config.in_feat, 1)):
    return l.Input(shape=in_shape, name=name)
    # return l.InputLayer(input_shape=in_shape, name=name, batch_size=None, dtype=config.data_type)
