
#  #Copyright 2019 Korea University under XAI Project supported by Ministry of Science and ICT, Korea
#
#  #Licensed under the Apache License, Version 2.0 (the "License");
#  #you may not use this file except in compliance with the License.
#  #You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
#  #Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import config
import tensorflow as tf
import models.layers as L
keras = tf.keras
K = keras.backend

class MatchingNetworks:
    def __init__(self, dim=128, fce=True):
        self.fce = fce
        self.dim = dim
        self.build_model()


    def _classifier(self):
        model = keras.models.Sequential(name="aesoirgj")
        model.add(L.dense(f=self.dim))
        model.add(L.batch_norm())
        model.add(L.relu())

        model.add(L.dense(f=self.dim))
        model.add(L.batch_norm())
        model.add(L.relu())
        return model

    def _g_biLSTM(self, x):
        return keras.layers.LSTM(self.dim, return_sequences=True)(x)

    def _f_LSTM(self, sup_emb, qry_emb, c_way=config.C_way):
        lstm_layer = keras.layers.LSTM(self.dim, return_state=True, return_sequences=True)
        att_layer = L.dense(f=self.dim, act="softmax")
        bn_layer = L.batch_norm()

        att_softmax = K.ones_like(qry_emb)
        att_softmax /= c_way


        for i in range(c_way):
            initial_state = None
            if i>0:
                temp_H = keras.layers.Multiply()([sup_emb, att_softmax])
                temp_H = keras.layers.AveragePooling1D(pool_size=(temp_H.shape[1]))(temp_H)
                temp_H = keras.layers.Reshape((temp_H.shape[-1],))(temp_H)

                H = keras.layers.Add()([H, temp_H])
                initial_state = (C, H)

            x, C, H = lstm_layer(qry_emb, initial_state=initial_state)
            x = bn_layer(x)
            att_softmax = att_layer(x)

        return x

    def build_model(self):
        query_input = L.in_layer(in_shape=(1, 1, config.in_feat))
        support_input = L.in_layer(in_shape=(config.C_way, config.K_shot, config.in_feat))

        self.cls_model = self._classifier()

        g_emb = self.cls_model(support_input)
        g_emb = keras.layers.AveragePooling2D(pool_size=(1, g_emb.shape[2]))(g_emb)
        g_emb = keras.layers.Reshape((g_emb.shape[1], g_emb.shape[-1]))(g_emb)

        f_emb = self.cls_model(query_input)
        f_emb = keras.layers.Reshape((1,f_emb.shape[-1]))(f_emb)

        if self.fce:
            g_emb = self._g_biLSTM(g_emb)
            f_emb = self._f_LSTM(sup_emb=g_emb, qry_emb=f_emb)

        self.model = keras.models.Model(inputs={"qry": query_input, "sup":support_input},
                                        outputs={"g_emb":g_emb, "f_emb":f_emb})