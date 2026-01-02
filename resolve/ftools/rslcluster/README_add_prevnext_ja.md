# XRISM/Resolve 用スクリプト：

イベントファイルに PREV/NEXT インターバルカラムを追加する前処理スクリプト

[https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/resolve_ftools_add_prevnext.sh](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/resolve_ftools_add_prevnext.sh)

本 Bash スクリプトは、XRISM/Resolve のイベントファイルに対して、

* イベントフィルタリング
* カラム追加
* GTI（Good Time Interval）の生成
* GTI 適用による最終フィルタリング

といった一連の前処理を **自動化**するものである。
国際共同研究における解析を想定し、**再現性（reproducibility）** と **堅牢性（robustness）** を重視した設計となっている。

---

## 概要（Overview）

本スクリプトは、以下の処理を順に実行する。

1. **前提条件のチェック**
   必要なコマンドおよび入力ファイルが存在するかを確認する。

2. **BL（BaseLine）イベントの除去**
   `_uf.evt` ファイルに対して `ITYPE < 5` の条件を用い、
   ベースライン（BL）イベントを除去する。

3. **前後インターバルカラムの追加**
   フィルタ後のイベントファイルに対し、
   各イベントの前後時間差を表すカラムを追加する。

   * `PREV_INTERVAL`
   * `NEXT_INTERVAL`

4. **GTI（Good Time Interval）の生成**
   対応する `_cl.evt` ファイルから GTI を生成する。

5. **GTI フィルタリングの適用**
   生成した GTI を用いて、処理済みイベントファイルを最終的にフィルタリングする。

---

## 必要条件（Requirements）

以下のコマンドラインツールが、システムの `$PATH` 上で利用可能である必要がある。

* [`resolve_util_ftselect.sh`](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_util_ftselect.sh)
* [`resolve_tool_addcol_prev_next_interval.py`](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_tool_addcol_prev_next_interval.py)
* [`resolve_util_ftmgtime.py`](resolve_util_ftmgtime.py)

これらのツールがインストールされている、もしくはシェルから直接呼び出せる状態であることを確認すること。

---

## 使用方法（Usage）

```bash
./resolve_ftools_add_prevnext.sh <input_file_uf.evt>
```

### 実行例

```bash
resolve_ftools_add_prevnext.sh xa000114000rsl_p0px1000_uf.evt
```

---

## 入力ファイル（Input Files）

以下のファイルが **事前に存在している必要がある**。

* `xa<obsid>_uf.evt`
  未フィルタのイベントファイル（必須）

* `xa<obsid>_cl.evt`
  GTI 生成に用いるクリーンイベントファイル（必須）

---

## 出力ファイル（Output Files）

本スクリプトの実行により、以下のファイルが生成される。

* `<obsid>_noBL.evt`
  BL イベント（`ITYPE < 5`）を除去したイベントファイル

* `<obsid>_noBL_prevnext.evt`
  `PREV_INTERVAL` および `NEXT_INTERVAL` カラムを追加したイベントファイル

* `<obsid>_cl.gti`
  `_cl.evt` ファイルから生成された GTI ファイル

* `cutclgti.evt`
  GTI を適用した最終的なフィルタ済みイベントファイル

---

## 注意事項（Notes）

* 必要なコマンドや入力ファイルが 1 つでも欠けている場合、
  スクリプトは **即座に終了**する。
* 入力ファイルが存在する作業ディレクトリで実行するか、
  もしくは入力ファイルの **フルパスを指定**すること。
* 本スクリプトは解析の前処理段階を厳密に揃えることを目的としており、
  以降のクラスタリング解析や科学解析の **前提条件**として用いられる。