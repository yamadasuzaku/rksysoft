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


別のファイルですが、試験的に、

```sh
resolve_ana_pixel_ql_plotspec.py xa000138000rsl_p0px1000_cl.evt --rebin 250
```

のように、energy の bin 幅を変えることができるのもあります。


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


### itype 毎、pixel 毎のライトカーブを event list のファイルから生成する

eve.list に、ファイル名を1行ずつ書いておき、-y で itype, -p で pixel を指定してライトカーブを生成する。
時間ビンや出力するファイル名もオプションで指定できる。

``` bash:
resolve_ana_pixel_ql_mklc_binned_sorted_grade_itypes.py eve.list -y 0 -p 0,17,18,35,5,11,23,30
```

### 温度のプロット

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/hk/resolve_hk_plot_temptrend.sh

``` bash:
resolve_hk_plot_temptrend.sh xa300065010rsl_a0.hk1
```

これで、XBOXA_TEMP3_CAL,HE_TANK2,JT_SHLD2,CAMC_CT1,ADRC_CT_MON_FLUC のプロットを生成する。

中身は、

``` bash:
resolve_util_fplot.py $input_file TIME 7,6,6,6,6 XBOXA_TEMP3_CAL,HE_TANK2,JT_SHLD2,CAMC_CT1,ADRC_CT_MON_FLUC 7,6,6,6,6  -p -m . -s linear,linear,linear,log,log
```

を動かしているだけです。


# 論文執筆前のデータチェック用ツールのルーティン

## ghf のチェック

```bash:
resolve_ecal_plot_ghf.py xa000126000rsl_000_fe55.ghf
resolve_ecal_plot_ghf_with_FWE.py xa000126000rsl_000_fe55.ghf --hk1 xa000126000rsl_a0.hk1
```

- resolve_ecal_plot_ghf.py でフィット結果が正しいことを目視で確認する。
- resolve_ecal_plot_ghf_with_FWE.py で、FWEの時刻とGHFがあってることを確認する。

## ghf のチェック