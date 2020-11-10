#!/bin/python
# -*- coding:utf-8 -*-

import xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import calNormalFlow
import readParams
import pprint

# グローバル変数
flowCol = '流量'
# 栄養塩の種類
nutList = ['COD','TN','TP']
# 保存先
## LQ図
resDir = './res'
LQdir = resDir + '/LQ'

def getTargetObsData(params, thisTargetRiver):
    '''
    対象期間の観測値のデータを取得する
    '''
    # 水質観測値の読み込み
    obs_skiprows = 17
    ## 観測値の読み込み。sheet_nameはtarget_riverと一致していることを想定
    obs_pd = pd.read_excel(io=params['WQFilename'],sheet_name = thisTargetRiver,
            skiprows = obs_skiprows)

    # 平常時水質L-Q作成期間
    normLQterm = params['normalLQTargetDate'].split('-')
    ## データを平常時L-Q期間に限定
    normLQLimited = obs_pd[(obs_pd['Date'] >= normLQterm[0]) & 
            (obs_pd['Date'] <= normLQterm[1])]

    return normLQLimited

def getObsMaxFlow(params,thisTargetRiver):
    '''
    観測値における最大流量を取得する
    '''

    normLQLimited = getTargetObsData(params, thisTargetRiver)
    obsMaxFlow = normLQLimited[flowCol].max()

    return obsMaxFlow


def calNormalLQ(params, thisTargetRiver, thisNut, changeFlow):
    '''
    平常時L-Q式の作成
    '''
    ## 観測値で対象期間のデータを取得
    normLQLimited = getTargetObsData(params, thisTargetRiver)
    ## シート上の列名とnutコードの対応を付ける
    nutColOnObsPd = {'COD':'COD','TN':'全窒素','TP':'全リン'}
    thisNutColName = nutColOnObsPd[thisNut]
    ## 流量と濃度を抜き出す
    flowAndNutObs = normLQLimited.loc[:,[flowCol,thisNutColName]]

    # 負荷量(g/s)を算出
    flowAndNutObs['load(g/s)'] = flowAndNutObs[flowCol] * \
            flowAndNutObs[thisNutColName]
    flowAndNutObs.dropna(how='any',inplace=True) # 欠測値を含む行を削除
    thisLoad = flowAndNutObs['load(g/s)']
    thisFlow = flowAndNutObs[flowCol]

    # 回帰式の作成。logをとって1次式として扱う
    this_x = np.log10(thisFlow)
    this_y = np.log10(thisLoad)
    coefs = np.polyfit(this_x,this_y,1)
    this_b = coefs[0]
    this_a = 10**(coefs[1])
    ## 結果の格納
    normalLQCoef = { 'a' : this_a, 'b' : this_b }

    return normalLQCoef, flowAndNutObs 

def calRainLQ(thisNut, thisTargetRiver, thisTargetFlow, params, nutLoadPd, 
        changeFlow, normalLQCoef):
    '''
    出水時L-Qを作成する
    '''
    ## 対象河川の流量、最大流量、最小流量を取得
    #target_flow, max_flow, min_flow = flow_get(this_target_river,flow_pd_limited)

    # 負荷量の取得
    ## kg/日 -> g/年 に変換
    allLoadSum = nutLoadPd.loc[thisTargetRiver,thisNut] * 365 * 1000

    # 切り替え流量時の負荷の算出
    changeLoad = normalLQCoef['a'] * changeFlow ** normalLQCoef['b']

    # 平常時の負荷量合計の算出
    ## 平常時の流量
    targetFlowOfNormalFlow = thisTargetFlow[thisTargetFlow <= changeFlow]
    ## 平常時の負荷
    ## 平常時L-Q式を使用する。但しこの時g/sをg/hに変換するため3600を乗じる
    targetFlowOfNormalLoad = targetFlowOfNormalFlow.apply(
            lambda x : normalLQCoef['a'] * x ** normalLQCoef['b']) * 3600
    normalLoadSum = targetFlowOfNormalLoad.sum()

    # 出水時LQの取得
    # 出水時の負荷量の合計
    rainLoadSum = allLoadSum - normalLoadSum
    ## 出水時の流量
    targetFlowOfRainFlow = thisTargetFlow[thisTargetFlow > changeFlow]
    ## 係数の計算
    rainLQCoef = calRainCoef(rainLoadSum,targetFlowOfRainFlow,
            changeFlow,changeLoad)


    return rainLQCoef, allLoadSum
    
def calRainCoef(rainLoadSum,targetFlowOfRainFlow,changeFlow,changeLoad):
    '''
    出水時のLQ係数aとbを算出する
    '''
    rainFlowDiv = \
            targetFlowOfRainFlow[targetFlowOfRainFlow > changeFlow] / changeFlow

    ## 左辺の残差項
    residue = rainLoadSum / ( changeLoad * 3600 )

    ## bの算出
    rain_b = newton_sol(rainFlowDiv, residue)

    # Noneは解が求められなかった時
    if rain_b == None:
        rain_a = None
    else:
        ## aの算出
        rain_a = changeLoad / ( changeFlow ** rain_b )

    ## 格納
    rainLQCoef = {'a' : rain_a, 'b' : rain_b }

    return rainLQCoef



def output_res(targetRiverNames: list, nutList: list, res: dict) -> None:
    '''結果をcsvに出力'''

    # 結果の出力
    res_file = resDir + './res.csv'
    with open(res_file, "w",encoding="shift-jis") as f:

        f.write('河川,項目,平常時a,平常時b,出水時a,出水時b,切替流量\n')

        for thisRow in res:

            f.write("{},{},{},{},{},{},{}\n".format(
                thisRow['河川名'],thisRow['項目'],thisRow['平常時a'],
                thisRow['平常時b'],thisRow['出水時a'],thisRow['出水時b'],
                thisRow['切替流量']))


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
    return None
    
    
def check_LQ(target_flow : pd.Series, change_flow : float,
        normal_LQ_coef : list, rain_LQ_coef : list, Lsum_all : float) -> None:
    '''計算したL-Q式から負荷を算出する'''
    
    # 平常時の負荷量の算出(g/year)
    normal_flow = target_flow[target_flow <= change_flow]
    normal_load_sr = ( normal_LQ_coef['a'] * 
            normal_flow ** normal_LQ_coef['b'] ) * 3600
    normal_sum = normal_load_sr.sum() # g/year 
    print('normal_sum : {0} kg/日'.format(normal_sum/1000/365))

    # 出水時の解が求まった場合
    if not rain_LQ_coef['b'] == None:
        # 出水時の負荷量の算出(g/year)
        rain_flow = target_flow[target_flow > change_flow]
        rain_load_sr = ( rain_LQ_coef['a'] * 
                rain_flow ** rain_LQ_coef['b'] ) * 3600
        rain_sum = rain_load_sr.sum() # g/year 

        # L-Q式から算出した合計流出負荷量(kg/year)
        all_sum = normal_sum + rain_sum

        print('rain_sum : {0} kg/日'.format(rain_sum/1000/365))
        print('L-Qから算出した流出負荷量は{0}kg/日です.'.format(
            round(all_sum/1000/365),3))
    # 求まらなかった場合
    else:
        print('出水時計算失敗')

    print('設定したブロック負荷量は{0}kg/日です.'.format(
        round(Lsum_all/1000/365,3)))

def drawLQNormal(ax, thresFlows, normalLQCoef, flowAndNutObs,
        thisTargetRiver, thisNut):
    '''
    平常時のLQを図示する
    '''

    # 平常時L-Qに従った流量と負荷の線を作成する
    normFlow = np.linspace(thresFlows['minFlow'], thresFlows['changeFlow'], 10)
    normLoad = [ normalLQCoef['a'] * f ** normalLQCoef['b'] for f in normFlow]

    # 平常時L-Qに基づく線
    ## LQ式のラベル設定(TeXの{}をエスケープするためにf(python3.6以降)を使う
    ## f文の中では、{var}で変数varの値を用いることができる
    ## 特殊文字{}や\は二つ重ねることで一つのその文字となる
    ## {}はTeXでシステム的に用いられる文字で例えば上付き^{var}のように使う。
    ## {}の中に変数を用いたい時には、まず{{}}として{をエスケープする
    ## その上で、変数を使うために{var}とする。
    ## 結果として^{{{var}}}のようにかっこが三つ連続することになる。
    a = round(normalLQCoef['a'],3)
    b = round(normalLQCoef['b'],3)
    LQLabel = '平常時L-Q : ' + f'$y = {a} \\times x ^{{{b}}}$'
    ax.plot(normFlow,normLoad,color='mediumblue',label=LQLabel)
    # 平常時観測値の描画
    ax.plot(flowAndNutObs[flowCol],flowAndNutObs['load(g/s)'],
             'o', label = '平常時観測値', markeredgecolor = 'black',
            markerfacecolor = 'white')
    ax.set_xlabel("流量(" + r"$m^3/s$" + ")")
    ax.set_ylabel("負荷量(g/s)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("{0}({1})".format(thisTargetRiver,thisNut))

    return ax

def drawLQRain(ax, thresFlows: dict, rainLQCoef: dict,
        thisTargetRiver : str, thisNut: str):
    '''
    出水時のL-Qを描画
    '''

    # L-Q式の設定
    ## 出水時係数bが計算できなかった時はそのまま終わる
    if rainLQCoef['b'] == None:
        return ax
    rainFlow = np.linspace(thresFlows['changeFlow'],thresFlows['maxFlow'],10)
    rainLoad = [ rainLQCoef['a'] * f ** rainLQCoef['b'] for f in rainFlow]

    # L-Q式の記述
    ## 出水時
    a = round(rainLQCoef['a'],3)
    b = round(rainLQCoef['b'],3)
    LQLabel = '出水時L-Q : ' + f'$y = {a} \\times x ^{{{b}}}$'
    ax.plot(rainFlow,rainLoad,color='crimson',label=LQLabel)
    ax.set_xlabel("流量(" + r"$m^3/s$" + ")")
    ax.set_ylabel("負荷量(g/s)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("{0}({1})".format(thisTargetRiver,thisNut))
            
    return ax

def drawLQ(thresFlows, normalLQCoef, rainLQCoef, flowAndNutObs,
        thisTargetRiver, thisNut):
    '''
    平常時と出水時のLQを描画する
    '''
    plt.style.use('equal_hw')
    mpl.rcParams['figure.figsize']=(6,6)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ## 平常時LQの描画
    ax = drawLQNormal(ax, thresFlows, normalLQCoef, flowAndNutObs,
            thisTargetRiver, thisNut)

    if not rainLQCoef['b'] == None:
        ## 出水時LQの描画
        ax = drawLQRain(ax, thresFlows, rainLQCoef,thisTargetRiver, thisNut)

        # 全体の調整
        ax.grid(True)
        ax.grid(which='minor')

        ## 凡例の順番を変更(平常時線、観測値、出水時線となるので
        ## 平常時線、出水時線、観測値にする
        hans, labs = ax.get_legend_handles_labels()
        ### リストの交換
        hans[1],hans[2] = hans[2],hans[1]
        labs[1],labs[2] = labs[2],labs[1]
        ax.legend(handles=hans,labels=labs,loc='upper left',fontsize='small')
        # ax.set_xlim(thresFlows['minFlow'],thresFlows['maxFlow'])

    plt.savefig(LQdir + '/LQ_' + thisTargetRiver + '_' + thisNut + '.png')
    plt.close()
    
    return


def main(conditionExcelFilename):
    '''
    L-Qを作成するための一連の操作の実行
    conditionExcelFilenameに設定ファイル名を指定する
    '''

    
    # 設定を読み取る
    ## 設定が記載されたシート
    settingSheetName = '設定事項'
    params = readParams.readParams(conditionExcelFilename, settingSheetName)

    # 対象河川名のリストを取得
    targetRiverSheetName = '処理対象河川'
    targetRiverNames = readParams.getTargetRiver(conditionExcelFilename, targetRiverSheetName)

    # 負荷量対象期間の流量を取得
    targetFlow = calNormalFlow.getTargetFlowForCalRainLQ(params)

    # 平常流量の読み取り
    if not params['stdFlow'] == 'obsMax':
        stdFlows = calNormalFlow.execCalStdFlow(conditionExcelFilename, settingSheetName)

    # 負荷量を取得
    ## kg/日の単位で読み込むこと
    print("注意！ファイル中の負荷量の単位は【kg/日】！")
    nutLoadFilename = params['nutLoadFilename']
    nutLoadSheetname = params['nutLoadSheetname']
    # index=0は河川名
    # 使用する列は河川名、COD、TN、TPの4つの列とする
    nutLoadPd = pd.read_excel(nutLoadFilename,index_col=0, 
            usecols=[0,1,2,3], sheet_name = nutLoadSheetname)


    ## for debug
    #targetRiverNames = ['多摩川']
    #targetRiverNames = targetRiverNames[1:]
    #nutList = ['COD']

    # 結果格納用
    res = []

    for thisTargetRiver in targetRiverNames:

        # この河川の流量
        thisTargetFlow = targetFlow[thisTargetRiver]
        ## 最大流量・最小流量・切り替え流量を取得 
        maxFlow = thisTargetFlow.max()          # 最大流量
        minFlow = thisTargetFlow.min()          # 最小流量
        if not params['stdFlow'] == 'obsMax':
            changeFlow = stdFlows[thisTargetRiver]  # 切り替え流量
        else:
        ## 切り替え流量を観測値における最大流量とする方法
            changeFlow = getObsMaxFlow(params, thisTargetRiver)


        ## 辞書型にしてパック
        thresFlows = {'maxFlow':maxFlow,'minFlow':minFlow,'changeFlow':changeFlow}

        for thisNut in nutList:


            # 平常時LQの作成
            normalLQCoef,flowAndNutObs = calNormalLQ(params, thisTargetRiver, thisNut, thresFlows['changeFlow'])

            # 出水時LQの作成
            rainLQCoef,allLoadSum = calRainLQ(thisNut, thisTargetRiver, thisTargetFlow, 
                    params, nutLoadPd, thresFlows['changeFlow'], normalLQCoef)

            # 結果表示
            print(f'河川名:{thisTargetRiver}、項目:{thisNut}')
            print('平常時')
            pprint.pprint(normalLQCoef)
            print('出水時')
            pprint.pprint(rainLQCoef)
    
            # LQから算出した負荷量と設定値が合致するかの確認
            check_LQ(thisTargetFlow, thresFlows['changeFlow'], 
                    normalLQCoef, rainLQCoef, allLoadSum)

            # 描画
            drawLQ(thresFlows, normalLQCoef, rainLQCoef,flowAndNutObs,
                    thisTargetRiver, thisNut)

            # 結果をファイルに格納
            thisRiverRes = {'河川名':thisTargetRiver,
                    '項目':thisNut,'平常時a':normalLQCoef['a'],
                    '平常時b':normalLQCoef['b'],
                    '出水時a':rainLQCoef['a'],
                    '出水時b':rainLQCoef['b'],
                    '切替流量':changeFlow}
            res.append(thisRiverRes)    # 出力

    output_res(targetRiverNames, nutList, res)
    





if __name__ == '__main__':

    ## L-Qの条件ファイル
    conditionExcelFilename = './data/L-Q計算設定ファイル.xlsx'

    main(conditionExcelFilename)
