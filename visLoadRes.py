import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# 出力されたres.csvのL-Q平常時・出水時負荷量と定量化した負荷量の比較を
# 棒グラフで表示させる
# COD/TN/TP_負荷量比較.pngとして出力される

resFile = './res/res.csv'
resDf = pd.read_csv(resFile,encoding='shift-jis',index_col=0)

resDir = './res/LQ'

mpl.rcParams['xtick.labelsize'] = 'small'

for item in ['COD','TN','TP']:

    thisItemDf = resDf[resDf['項目'] == item ]

    rivers = thisItemDf.index
    x_position = np.arange(len(rivers))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.bar(x_position,thisItemDf['LQ平常時負荷量(kg/日)'],width=0.4,label='平常時')
    ax.bar(x_position,thisItemDf['LQ出水時負荷量(kg/日)'],bottom=thisItemDf['LQ平常時負荷量(kg/日)'],width=0.4,label='出水時')
    ax.bar(x_position+0.4,thisItemDf['定量化負荷量(kg/日)'],width=0.4,color='red', label='定量化結果')
    ax.set_xticks(x_position+0.2)
    ax.set_xticklabels(rivers)
    ax.grid(True)
    ax.legend()
    #ax.set_xlim(-1,10)
    plt.savefig(resDir + '/' + item + '_負荷量比較.png')




