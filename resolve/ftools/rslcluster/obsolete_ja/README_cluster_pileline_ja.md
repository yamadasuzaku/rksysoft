# README：クラスタリング・パイプラインスクリプト

（Clustering Pipeline Script）

本スクリプトは、指定された `.evt` ファイルに対して、**擬似イベント（pseudo event）クラスタ**を同定・解析するための **3 段階の処理を自動化**するものである。

---

## スクリプト名

```bash
resolve_ftools_cluster_pileline.sh
```

* [resolve_ftools_cluster_pileline.sh](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/resolve_ftools_cluster_pileline.sh)

---

## 使用方法（Usage）

[resolve_ftools_cluster_pileline.sh](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/resolve_ftools_cluster_pileline.sh) の使用方法は以下の通り。

```bash
./run_cluster_pipeline.sh <input_file_uf.evt>
```

* `<input_file_uf.evt>`
  `_uf.evt` というサフィックスを持つ入力イベントファイル。

---

## 処理ステップの概要（Overview of Steps）

### 1. Prev/Next Interval カラムの追加

* [resolve_ftools_add_prevnext.sh](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/resolve_ftools_add_prevnext.sh) を呼び出し、
  入力 FITS ファイルに以下のカラムを追加する。

  * `PREV_INTERVAL`
  * `NEXT_INTERVAL`

これらはイベントの時間的前後関係を表す情報であり、その後のクラスタリング処理に必須である。

---

### 2. クラスタ検出の実行

* [resolve_ftools_add_cluster.py](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/resolve_ftools_add_cluster.py) を **2 回実行**する。

  1. **`large` モード**
     宇宙線などによる **大きな信号**に起因するクラスタを検出する。

  2. **`small` モード**
     小さく・遅い信号（クロストークなど）によるクラスタを検出する。

* クラスタリングアルゴリズムの詳細については、以下の README を参照すること。

  * [XRISM Resolve Pseudo-Event Clustering Tool](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/README_add_cluster.md)

---

### 3. 診断チェックの実行

* [resolve_ftools_qlcheck_cluster.py](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/resolve_ftools_qlcheck_cluster.py) を使用し、
  クラスタリング結果の **検証および可視化（診断プロット生成）**を行う。

---

## 出力ファイル（Output Files）

本スクリプトは、入力ファイル名にプレフィックスおよびサフィックスを付加することで、
中間ファイルおよび最終ファイルを生成する。

| 処理段階          | 出力ファイル例                                              |
| ------------- | ---------------------------------------------------- |
| Step 1 後      | `xa000000000_noBL_prevnext_cutclgti.evt`             |
| Step 2（large） | `large_xa000000000_noBL_prevnext_cutclgti.evt`       |
| Step 2（small） | `small_large_xa000000000_noBL_prevnext_cutclgti.evt` |
| 最終チェック        | QL ツールによって生成される診断プロットおよびログ                           |

---

## 必要条件（Requirements）

本スクリプトを実行するには、以下のツールが **システムの `PATH` 上に存在すること**が必要である。

* [`resolve_ftools_add_prevnext.sh`](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/resolve_ftools_add_prevnext.sh)
* [`resolve_ftools_add_cluster.py`](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/resolve_ftools_add_cluster.py)
* [`resolve_ftools_qlcheck_cluster.py`](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/resolve_ftools_qlcheck_cluster.py)

これらのツールは、`rslcluster` あるいは同等の解析パッケージの一部として提供されていることを想定している。

---

## 注意事項（Notes）

* 必要なファイルが 1 つでも存在しない場合、スクリプトは **即座に停止**する。
* 中間生成ファイルは **確認なしで上書き**される。
* ファイル名の衝突を避けるため、
  **クリーンな作業ディレクトリでの実行を推奨**する。
