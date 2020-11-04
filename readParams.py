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
            'makeFlowGraphFlag','loadFlowTargetDate']

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
