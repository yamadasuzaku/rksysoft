# resolve_ftools_lsrates_errorbar_tools_final_tighttile.py

Resolve の `lsrates` CSV から、paircheck 済みの論文向け matplotlib 図と Plotly debug plot を生成するスクリプトである。

この版では、matplotlib の 6x6 tile plot に対して、論文向けの tight-packed 表示を option で追加した。

## 今回の変更点

### tight-packed tile plot

以下の option を追加した。

```bash
--paper-tight-tile
```

これを指定すると、matplotlib の 6x6 tile plot が次のような見た目になる。

- panel 間の隙間を **ゼロ**
- 6x6 を **tight packing**
- 各 panel の pixel 表示は、**左上の数字だけ**
- 既存の `pix 0` のような panel title は出さない
- 共有軸を使い、外周だけ tick label を残す
- 共通の x / y label は figure 全体に付ける

この tight-packed 版の出力ファイル名には `_tightpacked` が付く。

例：

```text
resolve_ftools_lsrates_paper_tile_errorbar_compare_stdcut_rate_small_only_tightpacked.png
```

## 使い方

通常の tile plot と同じ条件で、tight-packed 版を作る例：

```bash
python resolve_ftools_lsrates_errorbar_tools_final_tighttile.py "p0px1000/*lsrates.csv"     --outdir paper_outputs     --compare-stdcut     --y-col rate_small_only     --paper-xlim 0.05 20     --paper-ylim 1e-6 1     --paper-tight-tile     --all-pixels     --write-csv
```

従来どおりの pixel 色分けを使いたい場合は、併用できる。

```bash
python resolve_ftools_lsrates_errorbar_tools_final_tighttile.py "p0px1000/*lsrates.csv"     --outdir paper_outputs_pixel_color     --compare-stdcut     --y-col rate_small_only     --paper-tight-tile     --paper-color-by-pixel     --paper-xlim 0.05 20     --paper-ylim 1e-6 1     --all-pixels     --write-csv
```

この場合でも、デフォルトの推奨は

- no_stdcut: black
- stdcut: blue

である。

## 既存機能

以下は維持している。

- デフォルトの tile plot 色: no_stdcut=black, stdcut=blue
- `--paper-color-by-pixel` による従来の pixel 色分け
- all-pixel matplotlib overlay plot
- `t_exp_s` を用いた Poisson 誤差計算
- `ObsDate` の出力
- no_stdcut / stdcut paircheck
- `t_exp_s <= --min-exposure-ks` の短露光 screening
- `--object-include`
- `--object-exclude`
- `--obsid-include`
- `--obsid-exclude`
- `--debug-xlim` / `--debug-ylim`
- `--paper-xlim` / `--paper-ylim` の Plotly への流用
