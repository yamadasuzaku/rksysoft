# Resolve データをクイックチェックするスクリプトの説明

必要なプログラムに `PATH` が通っていることを前提にした書き方をしています。`PATH` を通さずに使う場合は、`chmod +x` で実行権限を与えてからの `./プログラム` で実行ください。ただし、`.sh` については `python` のプログラムへの `PATH` が通ってることが前提の書き方をしています。

## ライトカーブの作り方

[resolve_ana_pixel_ql_mklc_binned_sorted_itype_v1.py](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ana/pixel/resolve_ana_pixel_ql_mklc_binned_sorted_itype_v1.py) のコマンドを使用して、ライトカーブを作成します：

```sh
resolve_ana_pixel_ql_mklc_binned_sorted_itype_v1.py xa000138000rsl_p0px1000_cl.evt 
```

`--timebinsize TIMEBINSIZE` オプションを使用して、時間ビン（秒）を変更できます。

## スペクトルの作成方法 (非タイルプロット)

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


## スペクトルの作成方法 (タイルプロット)

イベントファイル、エネルギーの下限、上限、ビン幅、を指定すると、タイルプロットが生成されます。

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ana/pixel/resolve_ana_pixel_plot_6x6_energyspectrum_by_itype.py


``` python:
python resolve_ana_pixel_plot_6x6_energyspectrum_by_itype.py xa900001010rsl_p0px1000_cl.evt  -l 0 -x 10000 -b 20 -c
```

広域のスペクトルを見る場合。`-r` オプションをつけると、スペクトルの比をだしてくれる。 

``` bash:
Example (1): plot spectra from 2keV to 20 keV witn 400 eV bin, Hp only 
  python resolve_ana_pixel_plot_6x6_energyspectrum_by_itype.py xa300049010rsl_p0px3000_cl.evt -b 400 -l 2000 -x 20000 -y 0
Example (2): plot spectral ratios from 2keV to 20 keV witn 400 eV bin, Hp only 
  python resolve_ana_pixel_plot_6x6_energyspectrum_by_itype.py xa300049010rsl_p0px3000_cl.evt -r -b 400 -l 2000 -x 20000 -y 0 -c -g
```

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

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ana/pixel/resolve_ana_run_addprevnext_Lscheck.sh

``` bash:
resolve_ana_run_addprevnext_Lscheck.sh xa300049010rsl_p0px3000_uf.evt
``` 

と実行すると、prev/next interval をつけて、Ls の quick check (`tolerance=100`(default)以下の連続したイベントの数の分布を計算)をしてくれる。


## uf.evt と pr.evt に prev/next interval をつける方法

uf.evt, pr.evt, cl.evt の3つが同じディレクトリにある状態、例えば、

- xa000114000rsl_p0px1000_uf.evt
- xa000114000rsl_a0pxpr_uf.evt
- xa000114000rsl_p0px1000_cl.evt

の3つのファイルがある状態で、

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_tool_pr_prevnextadd_cutcl.sh

``` bash:
resolve_tool_pr_prevnextadd_cutcl.sh xa000114000rsl_p0px1000_uf.evt xa000114000rsl_a0pxpr_uf.evt xa000114000rsl_p0px1000_cl.evt
``` 

を実行する。


## イベントファイルを、QUICK_DOUBLE と SLOPE_DIFFER で場合わけする方法


カットしたいファイル (例として `xa000114000rsl_a0pxpr_uf_fillprenext_cutclgti.fits`) に対して、下記を実行する。

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_util_ftselect_split_file_quickdouble_slopediffer.sh

``` bash:
resolve_util_ftselect_split_file_quickdouble_slopediffer.sh xa000114000rsl_a0pxpr_uf_fillprenext_cutclgti.fits
```

指定したいファイルを指定するだけ。

## イベントのカットをかける

望月くんのカットをかける。

``` bash:
resolve_util_screen_20240508.sh xa300049010rsl_a0pxpr_uf_fillprenext_cutclgti.fits xa300049010rsl_a0pxpr_uf_fillprenext_cutclgti_screenmochi.fits
```


## 波形 pr.evt の 6x6 のプロット方法

``` bash:
# 生波形
resolve_ana_pixel_pr_plot.py xa000114000rsl_a0pxpr_uf_fillprenext_cutclgti_itype_all_slope_b01_quick_b01_pixel_all.fits
# boxcar後の微分波形
resolve_ana_pixel_pr_plot.py xa000114000rsl_a0pxpr_uf_fillprenext_cutclgti_itype_all_slope_b01_quick_b01_pixel_all.fits --prevflag 
``` 

## 微分波形 pr.evt の 6x6 のプロット方法 + quick double で step を変更して確認する方法

``` bash:
resolve_ana_pixel_pr_plot.py xa300049010rsl_a0pxpr_uf_fillprenext_cutclgti_itype_all_slope_b00_quick_b00_pixel_all_mochicut.fits --xlims 100,300 -p -dr -c -s 1 -pr 
```


### itype 毎、pixel 毎のライトカーブを event list のファイルから生成する

eve.list に、ファイル名を1行ずつ書いておき、-y で itype, -p で pixel を指定してライトカーブを生成する。
時間ビンや出力するファイル名もオプションで指定できる。

``` bash:
resolve_ana_pixel_ql_mklc_binned_sorted_grade_itypes.py eve.list -y 0 -p 0,17,18,35,5,11,23,30
```

## 波形の確認方法


単純に波形をプロットする方法

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ana/pixel/resolve_ana_pixel_pr_plot.py


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
resolve_ecal_plot_ghf_detail.py xa000126000rsl_000_fe55.ghf
resolve_ecal_plot_ghf_with_FWE.py xa000126000rsl_000_fe55.ghf --hk1 xa000126000rsl_a0.hk1
```

- resolve_ecal_plot_ghf_detail.py でフィット結果が正しいことを目視で確認する。
- resolve_ecal_plot_ghf_with_FWE.py で、FWEの時刻とGHFがあってることを確認する。

## ghf の詳細チェック

```bash:
resolve_ecal_plot_ghf_detail.py xa000126000rsl_000_fe55.ghf  --detail
resolve_ecal_plot_ghf_detail.py xa000126000rsl_000_fe55.ghf --pixels 3,4,19,20,27,32 --detail
```

`--detail` というオプションで、一つのフィットごとに詳細な図を出力してくれる。
--pixels を指定しないと全ピクセルで、カンマ区切りで整数を指定するとピクセル毎にプロットする


## GTI の確認

``` bash:
resolve_util_gtiplot.py xa300036010rsl_tel.gti,xa300036010rsl_p0px1000_uf.evt -p -e xa300036010rsl_p0px1000_cl.evt,xa300036010rsl_p0px1000_uf.evt
``` 

コラムに `GTI` を含むファイルをカンマ区切り(or @ファイル)で指定するとGTIを表示する。-e オプションでイベントファイルを指定すると100秒ごとのライトカーブを生成して右y軸に表示してくれる。


# リプロセスの自動化に向けたスクリプト

## 暫定版のイベントスクリーニング、arf, rmf 生成のスクリプト

### Resolve

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ana/pixel/resolve_ana_runproc.py

``` bash:
resolve_ana_runproc.py
```

### Xtend

https://github.com/yamadasuzaku/rksysoft/blob/main/xtend/resolve_xtend_runproc.py

``` bash:
resolve_xtend_runproc.py
```


# GTIを丁寧に取り扱うスクリプト

## GTIごとにスペクトルを生成する

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ana/pixel/resolve_ana_pixel_mkspec_eachgti.py

``` bash:
resolve_ana_pixel_mkspec_eachgti.py xa300036010rsl_p0px1000_cl2_filt.evt -i 6600 -x 7200 -y 0 -m 5 -r 4 -t -v 0.002 --ymin 0.002 --ymax 0.017
```

## GTIごとにライトカーブを生成する

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ana/pixel/resolve_ana_pixel_mklc_branch.py

を使う。

### ライトカーブを生成する方法

`-l` オプションを使う。


``` bash:
resolve_ana_pixel_mklc_branch.py f.list -l -u -y 0 -p 0,17,18,35 -t 256 -o p0_17_18_35 -s
```

### Branching ratio を生成する方法

`-g` オプションを使う。

``` bash:
resolve_ana_pixel_mklc_branch.py noscreen.list -g -u -t 256 
 ```


# 波形処理関係

## テンプレート、平均波形、平均微分波形のプロット

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/plot/resolve_plot_templ_avgpulse.py

``` bash:
  python resolve_plot_templ_avgpulse.py xa035315064rsl_a0.hk2 -p
```        