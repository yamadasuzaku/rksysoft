# heates_make_grouptrigger_from_hdf5

HDF5形式のイベントデータから

* エネルギースペクトル
* チャンネルごとのスペクトル
* チャンネルごとのイベント数ヒストグラム
* 生きているピクセルのみを用いた **group-trigger JSON**

を自動生成する Python スクリプトです。

TES / µmux / X線検出器の **多チャンネル検出器データの quick look 解析**と **group trigger 設定生成**を目的としています。

---

# 概要

HDF5 ファイル内の

```
chan0
chan1
chan2
...
```

というグループを読み込み、各チャンネルの

```
energy
good
beam
```

データセットを用いてイベント選択を行います。

基本選択条件

```
good == True
```

さらに以下のオプションが利用できます。

* beam on / off 選択
* energy 範囲カット

これらの条件を適用した後、

* merged spectrum
* channel spectrum
* event count histogram
* grouptrigger JSON

を生成します。

---

# 特徴

このスクリプトの設計方針は次の通りです。

### 1. デフォルトではカットをかけない

以下は **すべてオプション**です。

| カット        | デフォルト |
| ---------- | ----- |
| beam cut   | 無効    |
| energy cut | 無効    |

つまり

```
good == True
```

のみでスペクトルを作成します。

---

### 2. beam on / off を選択可能

```
--beam-on
--beam-off
```

を指定すると

```
beam == 1
beam == 0
```

を選択します。

指定しない場合は beam カットは行いません。

---

### 3. beam on / off のイベント数を表示

スクリプト実行時に

```
Beam ON events
Beam OFF events
```

が自動的に print されます。

これにより

* beam 条件の統計
* beam cut の影響

を簡単に確認できます。

---

### 4. energy カットもオプション

energy カットを使う場合

```
--emin
--emax
```

を **両方指定**します。

例

```
--emin 5000 --emax 9000
```

指定しない場合は energy cut は行いません。

---

# 入力データ形式

HDF5 内に以下の構造を想定しています。

```
chan0
 ├ energy
 ├ good
 └ beam

chan1
 ├ energy
 ├ good
 └ beam
```

### energy

```
float array
```

イベントエネルギー

---

### good

```
bool array
```

good event flag

---

### beam

```
0 or 1
```

beam 状態

| 値 | 意味       |
| - | -------- |
| 1 | beam on  |
| 0 | beam off |

beam データセットが存在しない場合も動作します。

---

# 出力

スクリプトは以下のファイルを生成します。

```
output_energy_hist/
```

以下の内容が生成されます。

---

## 1 merged spectrum

```
*_merged_spectrum.png
```

全チャンネルのスペクトル

---

## 2 per-channel spectra

```
*_per_channel_spectra.png
```

チャンネルごとのスペクトルを重ね描き

---

## 3 channel event histogram

```
*_channel_count_histogram.png
```

チャンネルごとのイベント数

---

## 4 grouptrigger JSON

```
*_grouptrigger.json
```

生きているピクセルのみから生成

---

# Alive pixel の定義

以下の条件を満たすチャンネル

```
selected event > 0
```

つまり

```
good
beam cut
energy cut
```

すべて適用後に

```
event が1つ以上ある
```

チャンネルを **alive pixel** と定義します。

---

# grouptrigger JSON

group trigger は

```
{pixel : [left_neighbor, right_neighbor]}
```

の形式で生成されます。

例

```
{
  "0": [1,2],
  "1": [0,2],
  "2": [1,3],
  "3": [2,4],
  "4": [2,3]
}
```

ルール

### 内部ピクセル

```
i → [i-1 , i+1]
```

---

### 端のピクセル

内側に丸めます

```
0 → [1,2]
4 → [2,3]
```

---

### alive pixel のみ使用

dead pixel は自動的に除外されます。

---

# インストール

必要なパッケージ

```
numpy
matplotlib
h5py
```

インストール

```
pip install numpy matplotlib h5py
```

または

```
conda install numpy matplotlib h5py
```

---

# 使い方

## 基本

beam cut なし
energy cut なし

```
python heates_make_grouptrigger_from_hdf5.py data.h5
```

---

## beam ON

```
python heates_make_grouptrigger_from_hdf5.py data.h5 --beam-on
```

---

## beam OFF

```
python heates_make_grouptrigger_from_hdf5.py data.h5 --beam-off
```

---

## energy cut

```
python heates_make_grouptrigger_from_hdf5.py data.h5 \
--emin 5000 \
--emax 9000
```

---

## beam + energy

```
python heates_make_grouptrigger_from_hdf5.py data.h5 \
--beam-on \
--emin 5000 \
--emax 9000
```

---

# 出力例

実行すると以下のような summary が表示されます。

```
File               : data.h5
Beam mode          : none
Energy cut         : none

Beam ON events     : 14523
Beam OFF events    : 2211

Alive channels     : 28
Alive channel list : [0,1,2,3,...]

Selected events    : 16734
```

---

# 想定用途

このスクリプトは以下の用途を想定しています。

* TES detector
* µmux readout
* X-ray detector arrays
* quick look spectral analysis
* group trigger configuration
