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

\CID{220} `PREV_INTERVAL` および `NEXT_INTERVAL` は
[resolve_ftools_add_prevnext.sh](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/resolve_ftools_add_prevnext.sh) によって **事前に追加されている必要がある**。

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

---

## `identify_clusters()` 関数の詳細解説

（クラスタリング判定ロジックの正確な理解のために）

### 目的

`identify_clusters()` は、**1 ピクセル分のイベント列（時間順に並んだ配列）** に対して、

* 擬似イベントの **クラスタ開始点** を判定し
* その後に続くイベントを **同一クラスタとしてまとめ**
* 各イベントに

  * クラスタ ID
  * クラスタ内の順序番号
  * 直前イベントの情報

を付与するための **中核ロジック**である。

 **この関数はイベントを削除しない**
→ あくまで「注釈（annotation）」を付けるだけである。

---

## 入力引数

```python
identify_clusters(
    events,
    mode,
    threshold_large,
    threshold_small,
    interval_limit,
    rt_min,
    rt_max,
    SECOND_THRES_USE_LEN,
    debug=False
)
```

| 引数                     | 型                      | 意味                               |
| ---------------------- | ---------------------- | -------------------------------- |
| `events`               | FITS table / ndarray   | 単一ピクセルのイベント列（時間順）                |
| `mode`                 | `"large"` or `"small"` | クラスタ判定モード                        |
| `threshold_large`      | int                    | large 擬似イベントの LO_RES_PH 閾値       |
| `threshold_small`      | int                    | small 擬似イベントの目安値（※開始条件には直接使われない） |
| `interval_limit`       | int                    | クラスタ継続とみなす時間間隔                   |
| `rt_min`, `rt_max`     | int                    | RISE_TIME の正常範囲                  |
| `SECOND_THRES_USE_LEN` | int                    | PSP 定義の特殊値（通常 75）                |
| `debug`                | bool                   | デバッグ出力                           |

---

## 出力

```python
return (
    cluster_indices,
    member_indices,
    prev_lo_res_ph,
    prev_itype
)
```

| 配列                | 内容                 |
| ----------------- | ------------------ |
| `cluster_indices` | クラスタ ID（0 = 非クラスタ） |
| `member_indices`  | クラスタ内の順序番号（1 始まり）  |
| `prev_lo_res_ph`  | 直前イベントの LO_RES_PH  |
| `prev_itype`      | 直前イベントの ITYPE      |

---

## 処理の全体構造

```text
イベント列を先頭から順にスキャン
  ↓
クラスタ開始条件を満たすか？
  ├─ YES → クラスタ開始 → 継続条件を満たす限り連結
  └─ NO  → 非クラスタとして 0 を付与
```

重要なのは：

> **「クラスタ開始」と「クラスタ継続」は別条件**
> であり、非常に意図的に分離されている点である。

---

## クラスタ開始条件（最重要）

### `mode = "large"` の場合

```python
events["ITYPE"][i] == 3 and
events["LO_RES_PH"][i] > threshold_large
```

#### 意味

* **Lp イベント（ITYPE=3）**
* **非常に大きな LO_RES_PH**

→ 宇宙線や粒子ヒットなどの
**明確に異常な「大擬似イベント」** をクラスタの起点とする。

 **Ls（ITYPE=4）からクラスタは始まらない**

---

### `mode = "small"` の場合

```python
events["ITYPE"][i] == 3 and
events["LO_RES_PH"][i] <= threshold_large and
(
  events["NEXT_INTERVAL"][i] < interval_limit
  or
  events["NEXT_INTERVAL"][i] == SECOND_THRES_USE_LEN
) and
(
  events["RISE_TIME"][i] < rt_min
  or
  events["RISE_TIME"][i] > rt_max
)
```

#### 意味（1 条件ずつ）

1. **Lp イベントである**
2. **large ほど大きくない**
3. **直後にイベントが非常に近接している**
4. **RISE_TIME が異常（速すぎる or 遅すぎる）**

→ クロストーク・微小擬似イベント・遅延応答などを想定。

 **small モードは「単に小さいイベント」を拾っているのではない**
 
-->  **「時間相関 × 波形異常」を同時に要求している**

---

## クラスタ継続条件（開始後）

```python
(events["ITYPE"][i] in (3, 4)) and
(
  events["PREV_INTERVAL"][i] < interval_limit
  or
  events["PREV_INTERVAL"][i] == SECOND_THRES_USE_LEN
)
```

#### 重要ポイント

* **Lp / Ls の両方を含める**
* **「前のイベントとの時間差」で判定**

つまり：

> 「クラスタは、
> 開始点から **時間的に連続している限り** Lp/Ls を区別せず含める」

---

## よくある誤解 

### X 誤解 1

> `cluster_indices` にはクラスタ番号が入っている

#### 正解

* `cluster_indices[i] == i`（開始点のインデックス）
* 非クラスタは `0`

→ **「クラスタ ID = 開始イベントの行番号」**

---

### X 誤解 2

> `threshold_small` が small 判定に使われている

#### 正解

* **現在のコードでは使われていない**
* small 判定は `threshold_large` を境にしている

（※ 将来拡張用・パラメータ互換のために残されている）

---

### X 誤解 3

> クラスタはピクセルをまたぐ

#### 正解

* **この関数は 1 ピクセル分のみ**
* ピクセル間相関は上位ロジックで扱う設計

---

## 直前イベント情報の付加

```python
prev_lo_res_ph[k] = events["LO_RES_PH"][k-1]
prev_itype[k]     = events["ITYPE"][k-1]
```

### 目的

* **クラスタ検出の後処理**
* 「このイベントは何の直後に来たか？」を解析可能にする

--> **クラスタ判定そのものには使っていない**

---

## この設計の哲学

* **開始条件は厳密**
* **継続条件は緩やか**
* **削除しない**
* **判断は後段に委ねる**

これは、

> *「オンボードではできなかった精密判定を、
> 地上で情報を落とさず行う」*

という XRISM/Resolve 解析思想に沿った設計である。

---

## まとめ（README 用短文）

> `identify_clusters()` は、
> **Lp イベントを起点とした時間相関に基づき、
> 擬似イベント群を「クラスタ」として注釈付けする関数である。**
>
> large / small モードは「開始条件」のみが異なり、
> クラスタ継続条件は共通である。
> 本関数はイベントの削除を行わず、
> 後続解析のための判断材料を提供することを目的としている。


