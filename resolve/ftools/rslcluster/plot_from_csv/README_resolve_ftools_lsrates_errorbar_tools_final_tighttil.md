# Resolve FTOOLS lsrates Errorbar Plot Tools

`resolve_ftools_lsrates_errorbar_tools_final_tighttile.py` は、XRISM/Resolve の `lsrates` CSV ファイルを読み込み、`no_stdcut` と `stdcut` の整合性を確認したうえで、誤差棒付きの論文用プロットとデバッグ用 interactive プロットを生成する Python ツールです。

主な用途は、Resolve 各 pixel におけるイベントレート、standard cut 前後の変化、small/large pseudo event 関連レートなどを、観測ごと・pixel ごとに比較することです。

## Features

- `no_stdcut` / `stdcut` ファイルペアの事前チェック
- `(obs_id, pixel)` 単位での行ペア整合性チェック
- 問題のある OBSID/pixel をデフォルトで plot から除外
- Poisson 誤差付き rate plot の生成
- `t_exp_s` を用いた rate error 計算
- 短すぎる露光時間の観測を screening
- `object` 名による include / exclude selection
- `obs_id` による include / exclude selection
- Resolve 6x6 physical layout に対応した tile plot
- 論文向け tight-packed tile plot
- 全 pixel overlay plot
- Plotly による interactive debug plot
- Plotly hover 情報に `ObsDate`, `obs_id`, `pixel`, `object`, `t_exp_s` などを表示
- 各種チェック・フィルタ結果を CSV log として保存

## Requirements

Python 3.9 以上を推奨します。

必要な Python package は以下です。

```bash
pip install numpy pandas matplotlib plotly
````

## Input files

入力は Resolve `lsrates` の CSV ファイルです。

ファイル名から以下を判定します。

* `*_lsrates.csv`
  → `no_stdcut`
* `*_stdcut_lsrates.csv`
  → `stdcut`

例:

```text
xa001008010_lsrates.csv
xa001008010_stdcut_lsrates.csv
```

各 CSV には少なくとも以下の列が必要です。

```text
obs_id
pixel
t_exp_s
rate_all
n_all
rate_small_only
n_small_only
```

`object`, `time_start`, `time_end`, `time_span`, `event_file` などの列がある場合は、ログや Plotly hover 情報に利用されます。

## Basic usage

最も基本的な実行例です。

```bash
python resolve_ftools_lsrates_errorbar_tools_final_tighttile.py \
  "p0px1000/*lsrates.csv"
```

デフォルトでは、出力は以下のディレクトリに保存されます。

```text
resolve_ftools_errorbar_outputs/
```

## Recommended usage

`no_stdcut` と `stdcut` を比較し、論文向け tight tile plot も作る例です。

```bash
python resolve_ftools_lsrates_errorbar_tools_final_tighttile.py \
  "p0px1000/*lsrates.csv" \
  --compare-stdcut \
  --y-col rate_small_only \
  --paper-tight-tile \
  --write-csv
```

出力先を指定する場合:

```bash
python resolve_ftools_lsrates_errorbar_tools_final_tighttile.py \
  "p0px1000/*lsrates.csv" \
  --outdir outputs_abell2319 \
  --compare-stdcut \
  --y-col rate_small_only \
  --paper-tight-tile \
  --write-csv
```

## Typical examples

### 1. `stdcut` 前後の small-only rate を比較する

```bash
python resolve_ftools_lsrates_errorbar_tools_final_tighttile.py \
  "p0px1000/*lsrates.csv" \
  --compare-stdcut \
  --y-col rate_small_only
```

### 2. 論文用の tight-packed 6x6 tile plot を生成する

```bash
python resolve_ftools_lsrates_errorbar_tools_final_tighttile.py \
  "p0px1000/*lsrates.csv" \
  --compare-stdcut \
  --y-col rate_small_only \
  --paper-tight-tile \
  --paper-xlim 0.01 10 \
  --paper-ylim 1e-5 1
```

### 3. 特定の object だけを選ぶ

複数 object はセミコロン区切りで指定できます。

```bash
python resolve_ftools_lsrates_errorbar_tools_final_tighttile.py \
  "p0px1000/*lsrates.csv" \
  --object-include "Abell2319;Abell2319_BS;Abell2319_BS2" \
  --compare-stdcut \
  --y-col rate_small_only
```

完全一致にしたい場合:

```bash
python resolve_ftools_lsrates_errorbar_tools_final_tighttile.py \
  "p0px1000/*lsrates.csv" \
  --object-include "Abell2319" \
  --object-include-perfectmatch \
  --compare-stdcut
```

### 4. 特定の OBSID だけを選ぶ

```bash
python resolve_ftools_lsrates_errorbar_tools_final_tighttile.py \
  "p0px1000/*lsrates.csv" \
  --obsid-include "001008010;001008020" \
  --compare-stdcut
```

### 5. 特定の OBSID を除外する

```bash
python resolve_ftools_lsrates_errorbar_tools_final_tighttile.py \
  "p0px1000/*lsrates.csv" \
  --obsid-exclude "000104100" \
  --compare-stdcut
```

### 6. 短露光 screening の閾値を変更する

デフォルトでは `t_exp_s <= 6.5 ks` の行を除外します。

```bash
python resolve_ftools_lsrates_errorbar_tools_final_tighttile.py \
  "p0px1000/*lsrates.csv" \
  --min-exposure-ks 20 \
  --compare-stdcut
```

短露光 screening を無効化する場合:

```bash
python resolve_ftools_lsrates_errorbar_tools_final_tighttile.py \
  "p0px1000/*lsrates.csv" \
  --no-exposure-screen \
  --compare-stdcut
```

### 7. Plotly debug plot で表示 pixel を制限する

```bash
python resolve_ftools_lsrates_errorbar_tools_final_tighttile.py \
  "p0px1000/*lsrates.csv" \
  --pixels 0 1 2 3 \
  --initial-visible selected \
  --compare-stdcut
```

全 pixel を Plotly に含める場合:

```bash
python resolve_ftools_lsrates_errorbar_tools_final_tighttile.py \
  "p0px1000/*lsrates.csv" \
  --all-pixels \
  --compare-stdcut
```

## Output files

デフォルトの出力ディレクトリは以下です。

```text
resolve_ftools_errorbar_outputs/
```

主な出力ファイルは以下です。

### Pair check logs

```text
lsrates_pair_check_file_pairs.csv
lsrates_pair_check_obsid_summary.csv
lsrates_pair_check_missing_rows.csv
lsrates_pair_check_clean_keys.csv
```

これらは、`no_stdcut` / `stdcut` のファイルペアや `(obs_id, pixel)` ペアが正しく揃っているかを確認するためのログです。

### Screening / filtering logs

```text
lsrates_exposure_screening_dropped.csv
lsrates_object_filter_kept.csv
lsrates_object_exclude_dropped.csv
lsrates_obsid_filter_kept.csv
lsrates_obsid_exclude_dropped.csv
```

露光時間 screening、object selection、OBSID selection の結果が保存されます。

### Plot files

例:

```text
resolve_ftools_lsrates_paper_tile_errorbar_compare_stdcut_rate_small_only.png
resolve_ftools_lsrates_paper_tile_errorbar_compare_stdcut_rate_small_only_tightpacked.png
resolve_ftools_lsrates_paper_overlay_errorbar_compare_stdcut_rate_small_only.png
resolve_ftools_lsrates_debug_errorbar_compare_stdcut_rate_small_only.html
```

### Optional plot table

`--write-csv` を指定すると、plot に使った table も保存されます。

```text
resolve_ftools_lsrates_plot_table_compare_stdcut_rate_small_only.csv
```

## Important behavior

### Pair check

このツールは、plot を作る前に必ず `no_stdcut` / `stdcut` のペアチェックを行います。

デフォルトでは、問題のある OBSID/pixel は plot から除外されます。

問題があった時点で停止したい場合は、以下を指定します。

```bash
--require-clean-pairs
```

問題があるペアも含めて plot したい場合は、以下を指定します。

```bash
--no-pair-filter
```

ただし、最終図では `--no-pair-filter` は推奨されません。

### Poisson error

rate の誤差は、対応する count 列と露光時間列から計算されます。

例えば、

```text
rate_small_only
```

に対しては、

```text
n_small_only
```

が必要です。

誤差は以下で計算されます。

```text
sqrt(N) / t_exp_s
```

デフォルトでは露光時間として `t_exp_s` を使います。

別の時間列を使う場合:

```bash
--time-col time_span
```

### X-axis reference

X 軸はデフォルトで `rate_all` を使いますが、実際には `no_stdcut` 側の `rate_all` が基準値として使われます。

つまり、`stdcut` の点も、同じ `(obs_id, pixel)` に対応する `no_stdcut` の `rate_all` を X 軸に置いて表示されます。

## Main command-line options

| Option                   | Description                     |
| ------------------------ | ------------------------------- |
| `pattern`                | 入力 CSV の glob pattern           |
| `--outdir`               | 出力ディレクトリ                        |
| `--time-col`             | Poisson error 計算に使う露光時間列        |
| `--min-exposure-ks`      | 最小露光時間 screening の閾値            |
| `--no-exposure-screen`   | 短露光 screening を無効化              |
| `--x-rate-col`           | X 軸に使う rate 列                   |
| `--x-count-col`          | X 軸 rate に対応する count 列          |
| `--y-col`                | Y 軸に使う rate 列                   |
| `--compare-stdcut`       | `no_stdcut` と `stdcut` を重ねて表示   |
| `--cut-type`             | 比較しない場合に使う cut type             |
| `--object-include`       | object 名で include selection     |
| `--object-exclude`       | object 名で exclude selection     |
| `--obsid-include`        | OBSID で include selection       |
| `--obsid-exclude`        | OBSID で exclude selection       |
| `--paper-tight-tile`     | 論文向け tight-packed tile plot を生成 |
| `--paper-xlim`           | 論文用 plot の X 軸範囲                |
| `--paper-ylim`           | 論文用 plot の Y 軸範囲                |
| `--paper-xscale`         | 論文用 plot の X 軸 scale            |
| `--paper-yscale`         | 論文用 plot の Y 軸 scale            |
| `--paper-color-by-pixel` | cut type ではなく pixel ごとに色分け      |
| `--no-paper-overlay`     | overlay plot を生成しない             |
| `--pixels`               | Plotly debug plot に含める pixel    |
| `--all-pixels`           | Plotly debug plot に全 pixel を含める |
| `--initial-visible`      | Plotly trace の初期表示              |
| `--debug-linear-x`       | Plotly の X 軸を linear にする        |
| `--debug-linear-y`       | Plotly の Y 軸を linear にする        |
| `--debug-xlim`           | Plotly の初期 X 軸範囲                |
| `--debug-ylim`           | Plotly の初期 Y 軸範囲                |
| `--write-csv`            | plot table を CSV 出力             |
| `--output-prefix`        | 出力ファイル名の prefix                 |
| `--require-clean-pairs`  | ペアチェックに問題があれば停止                 |
| `--no-pair-filter`       | 問題ペアを除外せず plot に使う              |

## Notes

* `no_stdcut` と `stdcut` のファイル名判定では、`no_stdcut` が `stdcut` を文字列として含むため、`no_stdcut` が優先的に判定されます。
* `ObsDate` は `time_start` から計算されます。
* デフォルトの MJD reference day は `58484.0` です。
* Resolve の 6x6 physical layout に従って pixel が配置されます。
* 論文用 plot では、デフォルトで cut type を色で区別します。
* `--paper-color-by-pixel` を使うと、従来の pixel ごとの色分けにできます。

## License

Please add your license here.

Example:

```text
MIT License
```

## Author

Please add author information here.