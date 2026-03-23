# HDF5 `energy` データのゼロ値診断ツール

## 概要

本ツールは、HDF5 ファイル内の各チャネル (`chan*`) に格納された
`energy` データについて、

* **全要素が 0 かどうか**
* **0 ではない値が含まれているか**
* **ゼロの割合や統計量**

を一括で確認するための **デバッグ用スクリプト**です。

特に TES / μmux 系の解析において、

> **calibration（cal）が成功していれば energy は 0 以外の値を持つはず**

という前提に基づき、

* ✔️ **0 以外の値を持つ pixel（= cal 成功の可能性が高い）**
* ❌ **すべて 0 の pixel（= cal 失敗 or 未処理の可能性）**

を即座に判別できます。

---

## ✨ 主な機能

### 1. Summary Report（最重要）

スクリプト冒頭で以下を表示：

* 🔢 **0 ではない値を含む pixel 数**
* 📋 **その pixel 番号一覧**
* 🔢 **全て 0 の pixel 数**
* 📋 **その pixel 番号一覧**

👉 calibration 成功・失敗の切り分けに使う

### 2. チャネルごとの詳細統計

各 `chan*` について：

* 総イベント数
* 0 の数
* 非ゼロ数
* ゼロ割合
* NaN 数
* min / max / mean / std

### 3. CSV 出力（デフォルト）

自動で以下のファイルを生成：

```
debug_count_zero_energy_<入力ファイル名>.csv
```

例：

```
debug_count_zero_energy_20260323_run0013_mass_trans11.csv
```

---

## インストール

依存ライブラリ：

```bash
pip install h5py numpy
```

---

## 使い方

例：

```bash
python heates_debug_count_zero_energy.py 20260323_run0013_mass_trans11.hdf5
```
### オプション

```bash
--output <file.csv>
```

出力 CSV を指定したい場合：

```bash
python debug_count_zero_energy.py input.hdf5 --output result.csv
```

---

```bash
--dataset <name>
```

デフォルトは `energy`：

```bash
python debug_count_zero_energy.py input.hdf5 --dataset filt_value
```

---

## 解釈ガイド

### 正常（cal 成功の可能性が高い）

* `nonzero_count > 0`
* `zero_fraction < 1.0`

👉 energy に物理的な値が入っている

---

### ❌ 異常（要調査）

* `zero_count == total`

考えうる原因：

* calibration 未実行
* calibration 失敗
* 初期値（0）のまま保存
* energy 書き込み処理がスキップされている
* 上書きバグ

## 拡張アイデア

必要に応じて以下へ拡張可能：

* `filt_value`, `peak_value` も同時解析
* 時系列依存チェック
* pixel マップ可視化
* good/bad フラグとの相関

## ライセンス

自由に改変・再配布可能（研究用途・教育用途を想定）