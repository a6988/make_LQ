import xlrd

def readParams(conditionExcelFilename:str,settingSheetName:str)->str:
    '''
    必要な設定を設定ファイルから読み込む
    設定ファイルの形式は、Aに説明、Bに変数名、Cに内容、Dに備考
    '''
    ## 設定事項のシートをオープン
    setting_data = xlrd.open_workbook(conditionExcelFilename).\
            sheet_by_name(settingSheetName)
    params = {}

    # 読み込んだデータの変数名に格納されているものの取得
    paramNamesCol = 1
    paramNamesInSettingData = setting_data.col_values(paramNamesCol)

    paramsNames = ['WQFilename','normalLQTargetDate','flowFilename','flowUseCols','HeisuiOrHousui',
            'makeFlowGraphFlag','loadFlowTargetDate','nutLoadFilename']

    ## パラメータの取得
    ### パラメータの値が格納されている列
    setting_data_paramsCol = 2
    for thisParam in paramsNames:

        ## 重複してパラメータを設定している場合のエラー
        if paramNamesInSettingData.count(thisParam) == 0:
            raise Exception('パラメータ{}がシートで定義されていません'.format(thisParam))
        elif not paramNamesInSettingData.count(thisParam) == 1 :

            raise Exception('複数行で{}が定義されています'.format(thisParam))

        ## 設定事項のシートからパラメータの取得
        thisRow = paramNamesInSettingData.index(thisParam)
        params[thisParam] = setting_data.cell(thisRow, setting_data_paramsCol).value

    ## startDate, endDateを算出
    params['startDate'] = params['loadFlowTargetDate'].split('-')[0].strip()
    params['endDate'] = params['loadFlowTargetDate'].split('-')[1].strip()

    return params

def getTargetRiver(conditionExcelFilename, targetRiverSheetName):
    '''
    対象河川名のリストを取得
    '''
    
    riverData = xlrd.open_workbook(conditionExcelFilename).\
            sheet_by_name(targetRiverSheetName)

    # 名前の取得。一行目は対象河川というタイトルが入っているので除外
    targetRiverNames = riverData.col_values(0)[1:]

    # 空白要素の削除
    targetRiverNames = [thisItem for thisItem in targetRiverNames if thisItem != '']

    # 名前の重複がないかの判定
    if len(targetRiverNames) != len(set(targetRiverNames)):

        raise Exception('同じ河川名が定義されています')

    return targetRiverNames

if __name__ == '__main__':

    conditionExcelFilename = './data/L-Q計算設定ファイル.xlsx'
    targetRiverSheetName = '処理対象河川'
    targetRiverNames = getTargetRiver(conditionExcelFilename, targetRiverSheetName)
