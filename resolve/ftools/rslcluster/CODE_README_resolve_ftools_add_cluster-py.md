# CODE_README_resolve_ftools_add_cluster-py.md

XRISM/Resolve 擬似イベント・クラスタリングスクリプト
[resolve_ftools_add_cluster.py](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/resolve_ftools_add_cluster.py)

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/resolve_ftools_add_cluster.py

---

## 概要（Overview）

[resolve_ftools_add_cluster.py](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/resolve_ftools_add_cluster.py) は、XRISM/Resolve のイベント FITS ファイルに対して
**時間的に近接した擬似イベント（pseudo events）をクラスタとして同定し、
その情報をイベントごとのカラムとして付加する Python スクリプト**である。

本ツールは特に、

* 宇宙線などによる **大きな擬似イベント（large cluster）**
* クロストーク等に起因する **小さな擬似イベント（small cluster）**

を **時間相関・イベント特性（ITYPE, LO_RES_PH, RISE_TIME）に基づいて検出**することを目的としている。

スクリプトは **イベントを削除しない**。
あくまでクラスタ情報を **注釈（annotation）として付加**し、
後続の科学解析やフィルタ条件選択をユーザーに委ねる設計となっている。

---

## 主な機能

* ピクセルごとのイベント列に対するクラスタ検出
* large / small の 2 モードによるクラスタリング
* クラスタ ID・クラスタ内メンバー番号の付加
* 直前イベントの情報（LO_RES_PH, ITYPE）の保存
* 診断用プロット（時間 vs LO_RES_PH、NEXT_INTERVAL vs LO_RES_PH）の生成
* 元の FITS カラムを保持したまま安全に上書き出力

---

## 入力データの前提条件

本スクリプトは、以下の条件を満たすイベント FITS を入力として想定している。

### 必須カラム

* `TIME`
* `PIXEL`
* `ITYPE`

  * `3` : Lp イベント
  * `4` : Ls イベント
* `LO_RES_PH`
* `PREV_INTERVAL`
* `NEXT_INTERVAL`
* `RISE_TIME`

⚠️ `PREV_INTERVAL` および `NEXT_INTERVAL` は
`resolve_ftools_add_prevnext.sh` によって **事前に追加されている必要がある**。

---

## クラスタリングの考え方（アルゴリズム概要）

### 1. ピクセルごとの独立処理

クラスタリングは **ピクセル単位で独立**に行われる。
異なるピクセル間での直接的なイベント連結は行わない。

---

### 2. クラスタ開始条件

#### large モード

以下を満たすイベントを **クラスタ開始点**とする：

* `ITYPE == 3`（Lp）
* `LO_RES_PH > threshold_large`

→ 宇宙線などによる **高エネルギー擬似イベント**を想定。

---

#### small モード

以下をすべて満たすイベントを **クラスタ開始点**とする：

* `ITYPE == 3`
* `LO_RES_PH <= threshold_large`
* `NEXT_INTERVAL < interval_limit`
  または `NEXT_INTERVAL == SECOND_THRES_USE_LEN`
* `RISE_TIME < rt_min` または `RISE_TIME > rt_max`

→ クロストーク・遅延信号などによる **小規模擬似イベント**を想定。

---

### 3. クラスタ継続条件

クラスタ開始後、以下を満たす限りイベントを同一クラスタに含める：

* `ITYPE ∈ {3, 4}`（Lp または Ls）
* `PREV_INTERVAL < interval_limit`
  または `PREV_INTERVAL == SECOND_THRES_USE_LEN`

---

## 追加されるカラム

出力 FITS には以下のカラムが追加される。

| カラム名             | 型   | 意味                 |
| ---------------- | --- | ------------------ |
| `ICLUSTER`（任意名）  | int | クラスタ ID（0 = 非クラスタ） |
| `IMEMBER`（任意名）   | int | クラスタ内の順序番号         |
| `PREV_LO_RES_PH` | int | 直前イベントの LO_RES_PH  |
| `PREV_ITYPE`     | int | 直前イベントの ITYPE      |

※ 既存の同名カラムがある場合は **安全に置き換え**られる。

---

## 使用方法（Usage）

```bash
resolve_ftools_add_cluster.py input.evt [options]
```

### 基本例

```bash
resolve_ftools_add_cluster.py clean.evt \
  --mode large \
  --col_cluster ICLUSTERL \
  --col_member IMEMBERL \
  --outname large_ \
  -d
```

---

## コマンドライン引数

### 必須引数

| 引数           | 説明          |
| ------------ | ----------- |
| `input_fits` | 入力イベント FITS |

---

### 主なオプション

| オプション                      | 説明                   |
| -------------------------- | -------------------- |
| `-m, --mode {large,small}` | クラスタリングモード           |
| `-p, --usepixels`          | 解析するピクセル（例: `0,1,2`） |
| `--col_cluster`            | クラスタ ID カラム名         |
| `--col_member`             | メンバー番号カラム名           |
| `-o, --outname`            | 出力ファイル接頭辞            |
| `-f, --figdir`             | 図の保存先ディレクトリ          |
| `-d, --debug`              | デバッグ出力               |
| `-s, --show`               | プロット表示               |

---

### クラスタ判定パラメータ

| パラメータ                    | デフォルト | 意味                  |
| ------------------------ | ----- | ------------------- |
| `--threshold_large`      | 12235 | large 判定用 LO_RES_PH |
| `--threshold_small`      | 3000  | small 判定用目安         |
| `--interval_limit`       | 40    | クラスタ継続時間            |
| `--SECOND_THRES_USE_LEN` | 75    | PSP 仕様値             |
| `--rt_min`               | 32    | RISE_TIME 下限        |
| `--rt_max`               | 58    | RISE_TIME 上限        |
| `--mjdref`               | 58484 | MJD 基準時刻            |

---

## 出力ファイル

* 出力 FITS

  ```text
  <outname><input_fits>
  ```

* 診断図（ピクセルごと）

  ```text
  fig_cluster/cluster_summary_<outname>_pixel<pixel>.png
  ```

---

## 診断プロットの内容

1. **TIME vs LO_RES_PH**

   * Lp / Ls イベント
   * クラスタ化されたイベントを強調表示

2. **NEXT_INTERVAL vs LO_RES_PH**

   * 時間間隔によるクラスタの視覚確認

統計情報（Lp/Ls 数、クラスタ数）も図中に表示される。

---

## 設計上の注意・制限事項

* 本ツールは **注釈付けのみ**を行う
* 高輝度天体では真の同時 X 線が誤検出される可能性あり
* 親イベントが欠落した場合、クラスタ検出に失敗する可能性あり
* パラメータは **PSP の既知挙動に基づく経験的値**

---

## 想定される利用フロー

1. `resolve_ftools_add_prevnext.sh`
2. `resolve_ftools_add_cluster.py`（large）
3. `resolve_ftools_add_cluster.py`（small）
4. `resolve_ftools_qlcheck_cluster.py`
5. 科学解析用のイベント選別

---

## 関連ツール

* `resolve_ftools_add_prevnext.sh`
* `resolve_ftools_cluster_pipeline.sh`
* `resolve_ftools_qlcheck_cluster.py`

