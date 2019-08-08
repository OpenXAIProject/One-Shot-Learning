#  #Copyright 2019 Korea University under XAI Project supported by Ministry of Science and ICT, Korea
#
#  #Licensed under the Apache License, Version 2.0 (the "License");
#  #you may not use this file except in compliance with the License.
#  #You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
#  #Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import utils
import tensorflow as tf
from models.MN import MatchingNetworks
keras = tf.keras
K = keras.backend
class Tester:
    def __init__(self):
        self.all_model= MatchingNetworks().model
        self.all_model.load_weights("/Datafast/jsyoon/XAIOS/summ/0808_150340189_1/model/0115.h5")

        (self.trn_dat, self.trn_lbl), (self.val_dat, self.val_lbl), (self.tst_sup_dat, self.tst_qry_dat, self.tst_lbl) = utils.load_adni()


    def test(self):
        outputs = self.all_model({"sup": self.tst_sup_dat, "qry": self.tst_qry_dat[:, None,None]}, training=False)
        f_emb = outputs["f_emb"]
        g_emb = outputs["g_emb"]

        sum_support = K.sum(tf.square(g_emb), -1)
        support_magnitude = K.sqrt(K.clip(sum_support, 1e-10, float("inf")))
        dot_product = tf.matmul(f_emb, g_emb, transpose_b=True)
        dot_product = tf.squeeze(dot_product, [1, ])
        logit = dot_product * support_magnitude

        loss = keras.metrics.SparseCategoricalCrossentropy(from_logits=True)(self.tst_lbl, logit)
        acc = keras.metrics.SparseCategoricalAccuracy()(self.tst_lbl, logit)

        print(loss.numpy(), acc.numpy())