# Resolve データをクイックチェックするスクリプトの説明

必要なプログラムに `PATH` が通っていることを前提にした書き方をしています。`PATH` を通さずに使う場合は、`chmod +x` で実行権限を与えてからの `./プログラム` で実行ください。ただし、`.sh` については `python` のプログラムへの `PATH` が通ってることが前提の書き方をしています。

## ライトカーブの作り方

[resolve_ana_pixel_ql_mklc_binned_sorted_itype_v1.py](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ana/pixel/resolve_ana_pixel_ql_mklc_binned_sorted_itype_v1.py) のコマンドを使用して、ライトカーブを作成します：

```sh
resolve_ana_pixel_ql_mklc_binned_sorted_itype_v1.py xa000138000rsl_p0px1000_cl.evt 
```

`--timebinsize TIMEBINSIZE` オプションを使用して、時間ビン（秒）を変更できます。

## スペクトルの作成方法

[resolve_ana_pixel_ql_plotspec_v1.py](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ana/pixel/resolve_ana_pixel_ql_plotspec_v1.py) のコマンドを使用して、スペクトルを作成します：

```sh
resolve_ana_pixel_ql_plotspec_v1.py xa000138000rsl_p0px1000_cl.evt
```

このコマンドは、pixel ごと、itype ごとにスペクトルをプロットします。オプションはありません。

## delta T 分布のチェック

[resolve_ana_pixel_deltat_distribution.py](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ana/pixel/resolve_ana_pixel_deltat_distribution.py) のコマンドを使用して、delta T 分布をチェックします：

```sh
resolve_ana_pixel_deltat_distribution.py --fname xa000138000rsl_p0px1000_cl.evt
```

## delta T vs risetime の分布のチェック

[resolve_ana_pixel_deltat_risetime_distribution.py](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ana/pixel/resolve_ana_pixel_deltat_risetime_distribution.py) のコマンドを使用して、delta T vs risetime の分布をチェックします：

```sh
resolve_ana_pixel_deltat_risetime_distribution.py --fname xa000138000rsl_p0px1000_cl.evt
```

## pixel 毎、grade 毎のカウント数のチェック

[resolve_util_check_pixelrate.sh](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_util_check_pixelrate_plot.sh) のコマンドを使用して、pixel 毎、grade 毎のカウント数をチェックします：


```sh
resolve_util_check_pixelrate.sh xa000138000rsl_p0px1000_cl.evt
```

## grade 毎の pixel map を生成します。

[resolve_plot_detxdety_v1_alltypes.py](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_plot_detxdety_v1_alltypes.py) を利用して、カウント数が map の上に表示されます。

``` sh:
resolve_plot_detxdety_v1_alltypes.py xa300049010rsl_p0px3000_cl.evt
```

# cal用のツール

## prev/next interval をつけて、Ls の分類

cl.evt と uf.evt の２つが同じディレクトリにある状態、例えば、

- xa300049010rsl_p0px3000_cl.evt
- xa300049010rsl_p0px3000_cl.evt

の２つのファイルがある状態で、

``` bash:
resolve_ana_run_addprevnext_Lscheck.sh xa300049010rsl_p0px3000_uf.evt
``` 

と実行すると、prev/next interval をつけて、Ls の quick check をしてくれる。
