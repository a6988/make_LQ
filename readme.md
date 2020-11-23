# 概要

* 平常時・出水時の二段階に分けたL-Q式を算出するためのモジュール。
* $L=aQ^x$で示されるL-Q式の係数の設定方法は以下の通り。

	* 平常時は観測された公共用水域水質測定結果から最小二乗法で算出。
	* 出水時は、定量化された流域負荷量と一致するように係数を定める。

* 平常時と出水時の境目となる流量を切替流量と呼び、平水流量、豊水流量
及び使用する公共用水域水質測定の最大観測流量のいずれかを選択・算出する。

# 使い方

* `data`フォルダに以下のファイルを置く

|ファイル名|概要|
|:--------:|:---:|
|L-Q計算設定ファイル.xlsx|L-Qを算出するために必要なパラメータを設定するエクセルファイル|
|公共用水域水質測定結果.xlsx|L-Qを求めたい項目の公共用水域水質測定結果と流量が記載されたエクセルファイル|
|現況負荷量.xlsx|負荷量定量化結果が記載されたエクセルファイル|
|河川流量.xlsx|河川流量が1時間毎に記載されたエクセルファイル|

* L-Q計算設定ファイル.xlsxの設定を行う(設定方法は後述)

* 以上のファイル名は任意の名前を付けられる。以下に名前の記載を行う。

	* L-Q計算設定ファイル.xlsxは、`make_LQ.py`の`main()`の引数に指定する。

	* その他のファイルは、L-Q計算設定ファイル.xlsxの中のパラメータで指定する。

* `./res/LQ`フォルダを作成する。

* `python make_LQ.py`を実行する。

# L-Q計算設定ファイル.xlsxのパラメータの説明

## シート「設定項目」

|パラメータ名|説明|
|:----------:|:-----:|
|負荷合計/平水流量対象期間|負荷量を合計する期間。この期間の流量がピックアップされる|
|水質ファイル|公共用水域水質測定のデータファイル名|
|平常時L-Q作成対象期間|公共用水域水質測定の対象期間|
|河川流量ファイル|河川流量が格納されたファイル|
|流量の使用する列|河川流量で流量が入っている列名を指定|
|平水流量か豊水流量|切替流量を平水流量、豊水流量及び平常時観測の最大値の中から選択|
|流量のグラフを作成するか|上で平水流量、豊水流量を選択した時に日流量を上から順に並び換えたグラフを作成するかどうか|
|河川毎の負荷量(kg/日)の定量化結果ファイル|河川毎に纏められた定量化結果のファイル名
。単位に注意|
|河川毎の負荷量(kg/日)の定量化結果ファイルのシート名|上のファイルのシート名の指定|

## シート「処理対象河川」

* 処理対象となる河川名を2行目以降に記載する。
* この河川名と一致する名前で他のファイルから情報を取得するので、他の全てのファイルはこのシートの河川名と一致させること
