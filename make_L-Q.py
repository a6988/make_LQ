#!/bin/python
# -*- coding:utf-8 -*-

import xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def output_res(target_river_list: list, nut_list: list, res: dict) -> None:
    '''結果をcsvに出力'''

    # 結果の出力
    res_file = 'res.csv'
    with open(res_file, "w",encoding="shift-jis") as f:

        f.write('河川,項目,平常時a,平常時b,出水時a,出水時b,切替流量,ブロック負荷\n')

        for this_target_river in target_river_list:

            for this_nut in nut_list:
                f.write("{0},{1},{2},{3},{4},{5}, \
                        {6},{7}\n".format(
                        this_target_river, \
                        this_nut, \
                        res[this_target_river][this_nut][0]['a'], \
                        res[this_target_river][this_nut][0]['b'], \
                        res[this_target_river][this_nut][1]['a'], \
                        res[this_target_river][this_nut][1]['b'], \
                        res[this_target_river]['change_flow'], \
                        res[this_target_river][this_nut][2]
                        ))


def newton_sol(rain_flow_div : pd.Series, residue : float) -> None:
    '''ニュートン法によりrain_bを求める
    residueはニュートン法の左辺を0にするために引く
    '''

    # ニュートン法パラメータ
    df_x = 0.000000001 # 差分からの微分を求める時のdiff
    EPS = 0.000000000000001 # 計算収束条件
    N_max = 1000 # 計算最大回数
    x0 = 2       # xの初期値

    # 繰り返し計算
    for i in range(0,N_max):

        # 微分を求める
        ## x0の時のf(x)
        temp1 = rain_flow_div ** x0 
        temp1 = temp1.sum() - residue
        ## x0 + df_xの時のf(x+Δx)
        temp2 = rain_flow_div ** (x0 + df_x)
        temp2 = temp2.sum() - residue
        ## f'(x)の差分
        df = ( temp2 - temp1 ) / df_x

        # f(x0) / f'(x0)を求める
        mod = temp1 / df

        if abs(mod) > EPS:
            # 新しいx0を算出
            x0 -= mod
            print('回数:{0},b:{1}'.format(i,x0))
        else: 
            # bの答え
            return x0
        
    print('終了回数までに終わりませんでした。')
    sys.exit()
    


def flow_get(this_target_river : str, 
        flow_pd_limited : pd.DataFrame)-> (pd.DataFrame, float, float):
    '''対象河川の期間中の流量、最大流量、最小流量を取得'''
    
    target_flow = flow_pd_limited.loc[:,this_target_river]
    max_flow = target_flow.max()
    min_flow = target_flow.min()

    return target_flow, max_flow, min_flow
    
def cal_rainLQ(this_a, change_flow, this_b, target_flow, Lsum_all):
    '''
    出水時L-Qの作成
    '''
    ## 切り替え流量時の負荷(g/s)
    change_L = this_a * change_flow ** this_b

    # 平常時負荷量(g/year)の合計の取得
    Lsum_norm = 0.0
    ## 年間の内切り替え流量以下の流量を取得
    target_flow_below_change_flow = target_flow[target_flow <= change_flow]
    ## 平常時L-Q式を使用する。但しこの時g/sをg/hに変換するため3600を乗じる
    for now_flow in target_flow_below_change_flow:
        Lsum_norm += ( this_a * now_flow ** this_b ) * 3600

    # 出水時L-Q式の取得
    ## 出水時負荷量の取得(g/年)
    Lsum_rain = Lsum_all - Lsum_norm
    ## 出水時流量/change_flow
    rain_flow_div = \
            target_flow[target_flow > change_flow] / change_flow

    ## 左辺の残差項
    residue = Lsum_rain / ( change_L * 3600 )

    ## bの算出
    rain_b = newton_sol(rain_flow_div, residue)

    ## aの算出
    rain_a = change_L / ( change_flow ** rain_b )
    ## 格納
    rain_LQ_coef = {'a' : rain_a, 'b' : rain_b }

    return rain_LQ_coef



def check_LQ(target_flow : pd.Series, change_flow : float,
        normal_LQ_coef : list, rain_LQ_coef : list, Lsum_all : float) -> None:
    '''計算したL-Q式から負荷を算出する'''

    # 平常時の負荷量の算出(g/year)
    normal_flow = target_flow[target_flow <= change_flow]
    normal_load_sr = ( normal_LQ_coef['a'] * 
            normal_flow ** normal_LQ_coef['b'] ) * 3600
    normal_sum = normal_load_sr.sum() # g/year 

    # 出水時の負荷量の算出(g/year)
    rain_flow = target_flow[target_flow > change_flow]
    rain_load_sr = ( rain_LQ_coef['a'] * 
            rain_flow ** rain_LQ_coef['b'] ) * 3600
    rain_sum = rain_load_sr.sum() # g/year 

    # L-Q式から算出した合計流出負荷量(kg/year)
    all_sum = normal_sum + rain_sum

    print('normal_sum : {0} g/年'.format(normal_sum))
    print('rain_sum : {0} g/年'.format(rain_sum))
    print('L-Qから算出した流出負荷量は{0}kg/年です.'.format(all_sum/1000))
    print('設定したブロック負荷量は{0}kg/年です.'.format(Lsum_all/1000))

def draw_normalLQ(change_flow : float, normal_LQ_coef: dict,
        this_target_river : str, this_nut: str, flow_and_nut: pd.DataFrame,
        min_flow : float, max_flow : float,fig, ax ) -> None:
    '''平常時のL-Qを表示'''

    # L-Q式の設定
    min_flow = min(min_flow, flow_and_nut[flow_col].min())
    max_flow = max(max_flow, flow_and_nut[flow_col].max())
    norm_flow = np.linspace(min_flow, change_flow, 10)
    norm_load = [ normal_LQ_coef['a'] * f ** normal_LQ_coef['b'] for f in norm_flow]

    # L-Q式の記述
    ## 平常時
    ax.plot(norm_flow,norm_load,color='mediumblue',label='平常時L-Q')
    ## 出水時
    # 平常時の観測値
    ax.plot(flow_and_nut[flow_col],flow_and_nut['load(g/s)'],
             'o', label = '平常時観測値', markeredgecolor = 'black',
            markerfacecolor = 'white')
    ax.set_xlabel("流量(" + r"$m^3/s$" + ")")
    ax.set_ylabel("負荷量(g/s)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("{0}({1})".format(this_target_river,this_nut))
    ax.grid(True)
    ax.legend(loc='upper left',fontsize=20)

    return fig, ax
            
def draw_rainLQ(change_flow : float, rain_LQ_coef: dict,
        this_target_river : str, this_nut: str, flow_and_nut: pd.DataFrame,
        min_flow : float, max_flow : float,fig, ax ) -> None:
    '''出水時のL-Qを表示'''

    # L-Q式の設定
    min_flow = min(min_flow, flow_and_nut[flow_col].min())
    max_flow = max(max_flow, flow_and_nut[flow_col].max())
    rain_flow = np.linspace(change_flow,max_flow,10)
    rain_load = [ rain_LQ_coef['a'] * f ** rain_LQ_coef['b'] for f in rain_flow]

    # L-Q式の記述
    ## 出水時
    ax.plot(rain_flow,rain_load,color='crimson',label='出水時L-Q')
    ax.set_xlabel("流量(" + r"$m^3/s$" + ")")
    ax.set_ylabel("負荷量(g/s)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("{0}({1})".format(this_target_river,this_nut))
    ax.grid(True)
    ax.legend(loc='upper left',fontsize=20)
            
    return fig, ax

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

# 負荷のための流量合計期間の設定
load_term = params['load_term'].split('-')
# 平常時水質L-Q作成期間
norm_LQ_term = params['norm_LQ_term'].split('-')

# 栄養塩ループ用リスト
nut_list = ['COD','TN','TP']

## for debug
#this_target_river = target_river_list[0]
#this_nut = nut_list[0]

# 流量の取得
## 河川のデータの読み込み
river_sheet_name = '流量'
river_skiprows = 2
flow_pd = pd.read_excel(io=condition_excel_file,
        sheet_name = river_sheet_name, skiprows = river_skiprows)

# 結果用の変数
res = dict()

for this_target_river in target_river_list:

    ## 負荷のための流量合計期間に限定
    flow_pd_limited = flow_pd[(flow_pd['Date'] >= load_term[0]) & (flow_pd['Date'] <= load_term[1])]
    ## 対象河川の流量、最大流量、最小流量を取得
    target_flow, max_flow, min_flow = flow_get(this_target_river,flow_pd_limited)

    # 対応するブロックの取得
    corres_block_sheet_name = 'ブロック対応'
    corres_skiprows = 2
    corres_pd = pd.read_excel(io=condition_excel_file,
            sheet_name = corres_block_sheet_name, skiprows=corres_skiprows)
    ## 対応するブロックのリストの取得
    this_corres_block_list = corres_pd[this_target_river].iloc[0].split(',')

    # 結果格納辞書に河川のキーを追加
    res[this_target_river] = {}
    print("河川:{0}".format(this_target_river))

    for this_nut in nut_list:

        print("項目:{0}".format(this_nut))
        
        # 負荷量の取得
        # TODO ここは負荷量シート次第で変わりうるので注意
        nut_block_sheet_name = this_nut + '負荷量'
        block_sum_col = '計' # 負荷量合計の列名
        nut_skiprows = 2

        ## 負荷量のpdを作成。indexにブロックを指定
        nut_block_pd = pd.read_excel(io=condition_excel_file,
                sheet_name = nut_block_sheet_name, skiprows = nut_skiprows, index_col=0)
        this_nut_sum = 0.0
        ## 対応ブロックの「計」を読み込んで加算
        for this_corres_block in this_corres_block_list:
            this_nut_sum += nut_block_pd.loc[this_corres_block,block_sum_col]
        ## L-Q式の合計負荷量を入れる
        ## 単位をg/year
        ## 但しブロックの単位はkg/日なので、
        ## 年に換算するため365を、kg->gのため1000を乗じる
        #import pdb; pdb.set_trace()

        # TODO 単位確認すること
        Lsum_all = this_nut_sum * 365 * 1000

        # 水質観測値の読み込み
        # TODO ここも新しいファイルで変化しないか確認
        obs_skiprows = 17
        ## 観測値の読み込み。sheet_nameはtarget_riverと一致していることを想定
        obs_pd = pd.read_excel(io=params['file_WQ'],sheet_name = this_target_river,
                skiprows = obs_skiprows)
        ## シート上の列名とnutコードの対応を付ける
        # TODO 新しいファイルでの列名の付け方との整合を取る
        nut_col_on_obs_pd = {'COD':'COD','TN':'全窒素','TP':'全リン'}
        this_nut_col_name = nut_col_on_obs_pd[this_nut]

        ## 流量の列名
        # TODO 新しいファイルの流量の列名が以下と一致しているか確認
        flow_col = '流量'

        ## 平常時L-Q期間に限定
        # TODO Dateの列を作成しているか確認
        norm_LQ_limited = obs_pd[(obs_pd['Date'] >= norm_LQ_term[0]) & 
                (obs_pd['Date'] <= norm_LQ_term[1])]

        ## 流量と濃度を抜き出す
        flow_and_nut = norm_LQ_limited.loc[:,[flow_col,this_nut_col_name]]
        ## 切り替え流量の取得(平常時観測値の最大値)
        ## dropna()を行う前にすることで、対象期間中
        ## 公共用水域観測が行われたことのある最大流量を設定している
        change_flow = flow_and_nut[flow_col].max()

        ## 負荷量(g/s)を算出
        flow_and_nut['load(g/s)'] = flow_and_nut[flow_col] * \
                flow_and_nut[this_nut_col_name]
        flow_and_nut.dropna(how='any',inplace=True) # 欠測値を含む行を削除
        this_load = flow_and_nut['load(g/s)']
        this_flow = flow_and_nut[flow_col]
        ## 回帰式を作成。logをとって1次式として扱う
        this_x = np.log10(this_flow)
        this_y = np.log10(this_load)
        coefs = np.polyfit(this_x,this_y,1)
        this_b = coefs[0]
        this_a = 10**(coefs[1])
        ## 結果の格納
        normal_LQ_coef = { 'a' : this_a, 'b' : this_b }

        rain_LQ_coef = cal_rainLQ(this_a, change_flow, this_b, target_flow, Lsum_all)

        check_LQ(target_flow, change_flow, normal_LQ_coef, rain_LQ_coef, Lsum_all)
        #print('Lsum_norm : {0} g/year'.format(Lsum_norm ))
        #print('Lsum_rain : {0} g/year'.format(Lsum_rain ))

        with plt.style.context(('equal_hw')):

            fig = plt.figure()
            ax = fig.add_subplot()
            fig, ax = draw_normalLQ(change_flow, normal_LQ_coef, this_target_river, 
                    this_nut, flow_and_nut, min_flow, max_flow,fig,ax)
            fig, ax = draw_rainLQ(change_flow, rain_LQ_coef, this_target_river, 
                    this_nut, flow_and_nut, min_flow, max_flow,fig,ax)

#        print('normal LQ a : {0}, b : {1}'.format(normal_LQ_coef['a'], normal_LQ_coef['b']))
#        print('rain LQ a : {0}, b : {1}'.format(rain_LQ_coef['a'], rain_LQ_coef['b']))

        save_file = './L-Qfig/L-Qfig_{0}_{1}.png'.format(this_target_river, this_nut)
        plt.savefig(save_file)
        plt.clf()
        plt.close()
        # 結果の格納
        res[this_target_river][this_nut] = [normal_LQ_coef, rain_LQ_coef, 
                Lsum_all / 1000 ] 
        res[this_target_river]['change_flow'] = change_flow
        

# 結果の出力
output_res(target_river_list, nut_list, res)






