import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import xlrd
import readParams

def getTargetFlow(flowFilename:str, thisUsecols:str,
        startDayStr:str, endDayStr:str):
    '''
    使用する流量のdataFrameを返す
    '''
    flowData = pd.read_excel(flowFilename,usecols=thisUsecols)
    # Dateの作成
    ## 何故かapply関数の中では整数が少数に変換されてしまうので、intを付ける
    makeDate = lambda thisRow : dt.datetime(
            int(thisRow.年),int(thisRow.月),int(thisRow.日), int(thisRow.時間))
    flowData['Date'] = flowData.apply(makeDate,axis=1)
    ## indexをdateにする
    flowData.index=flowData['Date']
    flowData.drop(['年','月','日','時間','Date'],inplace=True,axis=1)

    # 対象期間のみに限定
    startDay = dt.datetime.strptime(startDayStr, '%Y/%m/%d %H:%M')
    endDay = dt.datetime.strptime(endDayStr, '%Y/%m/%d %H:%M')
    ## queryメソッド内で変数を使う時は@を付与する
    targetFlow = flowData.query(" index >= @startDay and index <= @endDay")

    return targetFlow

def calStdFlow(targetFlow, HeisuiOrHousui,makeFlowGraphFlag ):
    '''
    平水/豊水流量を算出する
    '''
    # 日平均流量の作成
    dayFlow = targetFlow.resample('D').mean()

    # 平水/豊水で日々を変更
    heisuiHousuiDays = {"Heisui":185,"Housui":95}

    try:
        stdDay = heisuiHousuiDays[HeisuiOrHousui]
    except:
        print('Heisui(平水流量)かHousui(豊水流量)で指定してください')

    stdFlows = {}
    # 河川毎に実行

    for thisRiver in dayFlow.columns:

        thisFlow = dayFlow[thisRiver]

        # 平水流量は一年を通じて185日はこれを下回らない流量なので、
        # 上から並べて185番目の流量となる
        sortedFlow = thisFlow.sort_values(ascending=False)
        sortedFlow.index = range(1,len(sortedFlow)+1)

        print("{}:{}".format(thisRiver, round(sortedFlow[stdDay],4)))
        stdFlows[thisRiver] = round(sortedFlow[stdDay],4)

        if makeFlowGraphFlag == True:
            makeFlowGraph(thisRiver, sortedFlow, stdDay)

    return stdFlows

    
def makeFlowGraph(thisRiver, sortedFlow,stdDay):
    '''
    日流量を大きい順に並べ変えたグラフを作る
    '''
    flowmin = sortedFlow.iloc[0]
    flowmax = sortedFlow.iloc[-1]
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(sortedFlow)
    ax.vlines([stdDay],flowmin,flowmax,ls="--",lw=3,color="red")
    ax.set_title(thisRiver)
    ax.grid()
    fig.savefig('./res/flow/{}.png'.format(thisRiver),bbox_inches='tight')
    plt.clf()

    return 


def execCalStdFlow(conditionExcelFilename, settingSheetName):
    '''
    実行
    '''

    params = readParams.readParams(conditionExcelFilename, settingSheetName)

    flowFilename = params['flowFilename']
    flowUseCols = params['flowUseCols']
    HeisuiOrHousui = params['HeisuiOrHousui']
    startDate = params['startDate']
    endDate = params['endDate']
    makeFlowGraphFlag = params['makeFlowGraphFlag']

    targetFlow = getTargetFlow(flowFilename,flowUseCols, startDate, endDate)

    stdFlows = calStdFlow(targetFlow, HeisuiOrHousui, makeFlowGraphFlag)
    
    return stdFlows


if __name__ == '__main__':
    
    conditionExcelFilename = './data/L-Q計算設定ファイル.xlsx'
    settingSheetName = '設定事項'
    stdFlows = execCalStdFlow(conditionExcelFilename, settingSheetName)

    





