
#  #Copyright 2019 Korea University under XAI Project supported by Ministry of Science and ICT, Korea
#
#  #Licensed under the Apache License, Version 2.0 (the "License");
#  #you may not use this file except in compliance with the License.
#  #You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
#  #Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import os

import tensorflow as tf
from tqdm import trange

import config
import utils
from models.MN import MatchingNetworks

keras = tf.keras
K = keras.backend


class Trainer:
    def __init__(self, summ_path):
        self.summ_path = summ_path
        self.train_writer = tf.summary.create_file_writer(summ_path+"train")
        self.valid_writer = tf.summary.create_file_writer(summ_path+"valid")

        MatchNets = MatchingNetworks(fce=True)
        self.all_model = MatchNets.model

        (self.trn_dat, self.trn_lbl), (self.val_dat, self.val_lbl), (self.tst_sup_dat, self.tst_qry_dat, self.tst_lbl) = utils.load_adni()

    def train_one_step(self, x_sup, x_qry, lbl, model, optim, vars, step=0, log=False):
        with tf.GradientTape() as tape:
            outputs = model({"sup": x_sup, "qry": x_qry[:, None, None]}, training=True)

            f_emb = outputs["f_emb"]
            g_emb = outputs["g_emb"]

            sum_support = K.sum(tf.square(g_emb), -1)
            support_magnitude = K.sqrt(K.clip(sum_support, 1e-10, float("inf")))
            dot_product = tf.matmul(f_emb, g_emb, transpose_b=True)
            dot_product = tf.squeeze(dot_product, [1, ])
            logit = dot_product * support_magnitude

            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)(lbl, logit)
        grads = tape.gradient(loss, vars)
        optim.apply_gradients(zip(grads, vars))

        if log:
            acc = keras.metrics.SparseCategoricalAccuracy()(lbl, logit)
            with self.train_writer.as_default():
                tf.summary.scalar(name="loss", data=loss, step=step)
                tf.summary.scalar(name="acc", data=acc, step=step)
                tf.summary.histogram(name="simmilarity", data=logit, step=step)

    def logger(self, x_sup, x_qry, lbl, model, step=0):
        outputs = model({"sup": x_sup, "qry": x_qry[:, None, None]}, training=False)
        f_emb = outputs["f_emb"]
        g_emb = outputs["g_emb"]

        sum_support = K.sum(tf.square(g_emb), -1)
        support_magnitude = K.sqrt(K.clip(sum_support, 1e-10, float("inf")))
        dot_product = tf.matmul(f_emb, g_emb, transpose_b=True)
        dot_product = tf.squeeze(dot_product, [1, ])
        logit = dot_product * support_magnitude

        loss = keras.metrics.SparseCategoricalCrossentropy(from_logits=True)(lbl, logit)
        acc = keras.metrics.SparseCategoricalAccuracy()(lbl, logit)
        with self.valid_writer.as_default():
            tf.summary.scalar(name="loss", data=loss, step=step)
            tf.summary.scalar(name="acc", data=acc, step=step)
            tf.summary.histogram(name="simmilarity", data=logit, step=step)

    def train(self):
        os.makedirs(self.summ_path+"model/")
        optimizer = keras.optimizers.Adam(0.00001)
        global_step = 0
        print("Training Started. Results and summary files are stored at", self.summ_path)
        for cur_epoch in trange(config.epoch):
            trn_sup_idx, trn_qry_idx, trn_lbl = utils.get_idx(lbl=self.trn_lbl)
            for cur_step in trange(0, config.iter_cnt, config.batch_size):
                cur_sup_dat = self.trn_dat[trn_sup_idx[cur_step:cur_step+config.batch_size]]
                cur_qry_dat = self.trn_dat[trn_qry_idx[cur_step:cur_step+config.batch_size]]
                cur_lbl = trn_lbl[cur_step:cur_step+config.batch_size]
                self.train_one_step(x_sup=cur_sup_dat, x_qry=cur_qry_dat, lbl=cur_lbl,
                                    model=self.all_model, optim=optimizer, vars=self.all_model.trainable_variables,
                                    step=global_step, log=global_step%10==0)

                global_step+=1

            val_sup_idx, val_qry_idx, val_lbl = utils.get_idx(lbl=self.val_lbl, iter_cnt=100)
            self.logger(x_sup=self.val_dat[val_sup_idx], x_qry=self.val_dat[val_qry_idx], lbl=val_lbl,
                        model=self.all_model, step=global_step)

            self.all_model.save_weights(self.summ_path+"model/%04d.h5"%cur_epoch)