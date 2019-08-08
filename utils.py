
#  #Copyright 2019 Korea University under XAI Project supported by Ministry of Science and ICT, Korea
#
#  #Licensed under the Apache License, Version 2.0 (the "License");
#  #you may not use this file except in compliance with the License.
#  #You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
#  #Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import scipy.io as sio
import numpy as np
import config

def split_data(x, rand_idx, ratio):
    trn_cnt = int(len(rand_idx)*ratio[0])
    trn_x = x[rand_idx[:trn_cnt]]
    val_x = x[rand_idx[trn_cnt:]]
    return trn_x, val_x

def make_k_shot_test(trn_dat, trn_lbl, tst_dat):
    tst_lbl_cnt = len(tst_dat)
    sup_cnt = tst_lbl_cnt-config.K_shot
    unique_lbl = np.unique(trn_lbl)
    trn_sup_idx = np.zeros(shape=(sup_cnt, config.C_way-1, config.K_shot), dtype=np.uint16)
    for cur_tst_cnt, cur_sup_c_lbl in enumerate(np.random.choice(unique_lbl, (sup_cnt, config.C_way-1), replace=True)):
        for cur_c_cnt, cur_sup_k_lbl in enumerate(cur_sup_c_lbl):
            cur_idx = np.argwhere(trn_lbl==cur_sup_k_lbl).squeeze()
            trn_sup_idx[cur_tst_cnt, cur_c_cnt] = np.random.choice(cur_idx, size=config.K_shot, replace=False)

    tst_sup_idx = np.random.choice(tst_lbl_cnt, size=config.K_shot, replace=False)
    tst_qry_idx = np.setdiff1d(np.arange(tst_lbl_cnt), tst_sup_idx)

    trn_sup_dat = trn_dat[trn_sup_idx]
    tst_sup_dat = np.tile(tst_dat[tst_sup_idx][None, None], (sup_cnt, 1, 1, 1))

    sup_dat = np.concatenate((trn_sup_dat, tst_sup_dat), axis=1)
    tst_lbl = np.array([np.random.permutation(sup_dat.shape[1]) for _ in range(sup_dat.shape[0])])
    sup_dat = sup_dat[np.arange(len(sup_dat))[:, None], tst_lbl]
    qry_dat = tst_dat[tst_qry_idx]
    tst_lbl = np.argmax(tst_lbl==(sup_dat.shape[1]-1), axis=-1)

    return sup_dat, qry_dat, tst_lbl


def load_adni():
    dat = sio.loadmat(config.data_path+"ADNI/Mean_ROI_Features.mat")

    dat_nc = np.transpose(dat['NC_Feature'])
    dat_ad = np.transpose(dat['AD_Feature'])
    dat_mci = np.transpose(dat['MCI_Feature']).astype(config.data_type)

    dat_all = np.concatenate((dat_nc, dat_ad), axis=0).astype(config.data_type)
    lbl_all = np.concatenate((np.full(shape=len(dat_nc), fill_value=0),
                              np.full(shape=len(dat_mci), fill_value=1)))

    rand_idx = np.random.permutation(len(dat_all))

    trn_dat, val_dat = split_data(dat_all, rand_idx, [0.9,0.1])
    trn_lbl, val_lbl = split_data(lbl_all, rand_idx, [0.9,0.1])
    sup_dat, qry_dat, tst_lbl= make_k_shot_test(trn_dat, trn_lbl, dat_mci)
    return (trn_dat, trn_lbl), (val_dat, val_lbl), (sup_dat, qry_dat, tst_lbl)


def get_idx(lbl, iter_cnt=config.iter_cnt, c_way=config.C_way, k_shot=config.K_shot):
    unique_lbl = np.unique(lbl)
    cur_sup_idx = np.zeros(shape=(iter_cnt, c_way, k_shot+1), dtype=np.uint16)

    for cur_tst_cnt in range(iter_cnt):
        for cur_c_cnt, cur_sup_k_lbl in enumerate(np.random.choice(unique_lbl, c_way, replace=False)):
            cur_idx = np.argwhere(lbl==cur_sup_k_lbl).squeeze()
            cur_sup_idx[cur_tst_cnt, cur_c_cnt] = np.random.choice(cur_idx, size=config.K_shot+1, replace=False)
    cur_qry_idx = cur_sup_idx[...,-1]
    cur_sup_idx = cur_sup_idx[...,:-1]
    cur_lbl = np.random.choice(cur_qry_idx.shape[-1], size=len(cur_qry_idx), replace=True)

    cur_qry_idx = cur_qry_idx[np.arange(iter_cnt), cur_lbl]

    return cur_sup_idx, cur_qry_idx, cur_lbl