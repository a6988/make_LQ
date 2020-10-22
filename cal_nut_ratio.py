#!/bin/python
# -*- coding:utf-8 -*-

import xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


def flow_get(this_target_river : str, 
        flow_pd_limited : pd.DataFrame)-> (pd.DataFrame, float, float):
    '''対象河川の期間中の流量、最大流量、最小流量を取得'''
    
    target_flow = flow_pd_limited.loc[:,this_target_river]
    max_flow = target_flow.max()
    min_flow = target_flow.min()

    return target_flow, max_flow, min_flow

# L-Qの条件ファイル
condition_excel_file = 'L-Q計算設定ファイル.xlsx'

# 設定を読み取る
## 設定が記載されたシート
setting_sheet_name = '設定事項'
## 設定事項のシートをオープン
setting_sheet = xlrd.open_workbook(condition_excel_file).\
        sheet_by_name(setting_sheet_name)
## 設定の読み取り
### cellは(0,0)が左上のセルになる
params = {}
params['load_term'] = setting_sheet.cell(1,1).value # 負荷合計期間
params['file_WQ']   = setting_sheet.cell(2,1).value # 水質ファイル
params['norm_LQ_term'] = setting_sheet.cell(3,1).value # 平常時L-Q作成対象期間

# 処理対象河川の設定
target_river_sheet_name = '処理対象河川'
target_river_list = list(pd.read_excel(io=condition_excel_file,
    sheet_name=target_river_sheet_name,skiprows=0)['対象河川'])

# 平常時水質L-Q作成期間
norm_LQ_term = params['norm_LQ_term'].split('-')

# 結果用の変数
res = dict()

for this_target_river in target_river_list:

    # 結果格納辞書に河川のキーを追加
    res[this_target_river] = {}
    print("河川:{0}".format(this_target_river))

    # 水質観測値の読み込み
    obs_skiprows = 17
    ## 観測値の読み込み。sheet_nameはtarget_riverと一致していることを想定
    obs_pd = pd.read_excel(io=params['file_WQ'],sheet_name = this_target_river,
            skiprows = obs_skiprows)
    ## 栄養塩で抜き出す列名とnutコードの対応を付ける
    nut_col_on_obs_pd = {'TN':['全窒素','亜硝酸性窒素','硝酸性窒素',\
            'アンモニア性窒素'],'TP':['全リン','リン酸性リン']}

    ## 平常時L-Q期間に限定
    norm_LQ_limited = obs_pd[(obs_pd['Date'] >= norm_LQ_term[0]) & 
            (obs_pd['Date'] <= norm_LQ_term[1])]
    ## 流量と濃度を抜き出す
    ## TN
    TN_nut = norm_LQ_limited.loc[:,nut_col_on_obs_pd['TN']]
    ## TP
    TP_nut = norm_LQ_limited.loc[:,nut_col_on_obs_pd['TP']]

    ## na値がある行を削除
    TN_nut.dropna(how='any',inplace=True)
    TP_nut.dropna(how='any',inplace=True)
    
    ## 平均値を算出
    TN_nut['NH4_ratio'] = TN_nut['アンモニア性窒素'] / TN_nut['全窒素']
    TN_nut['NOx_ratio'] = ( TN_nut['亜硝酸性窒素'] + TN_nut['硝酸性窒素'] ) / \
            TN_nut['全窒素']
    TP_nut['PO4_ratio'] = TP_nut['リン酸性リン'] / TP_nut['全リン']

    NH4_ratio = TN_nut['NH4_ratio'].mean()
    NOx_ratio = TN_nut['NOx_ratio'].mean()
    PO4_ratio = TP_nut['PO4_ratio'].mean()

    # 結果の格納
    res[this_target_river]['NH4_ratio'] = NH4_ratio 
    res[this_target_river]['NOx_ratio'] = NOx_ratio 
    res[this_target_river]['PO4_ratio'] = PO4_ratio 

# 結果の出力
res_file = 'res_nut_ratio.csv'
f = open(res_file, "w",encoding="shift-jis")

f.write('河川,NH4/TN,NOx/TN,PO4/TP\n')

for this_target_river in target_river_list:

    f.write("{0},{1:4.2f},{2:4.2f},{3:4.2f}\n".format( \
            this_target_river, \
            res[this_target_river]['NH4_ratio'], \
            res[this_target_river]['NOx_ratio'], \
            res[this_target_river]['PO4_ratio']
            ))

f.close()







