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