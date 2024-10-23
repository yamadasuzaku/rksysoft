# Resolve データをクイックチェックするスクリプトの説明

必要なプログラムに `PATH` が通っていることを前提にした書き方をしています。`PATH` を通さずに使う場合は、`chmod +x` で実行権限を与えてからの `./プログラム` で実行ください。ただし、`.sh` については `python` のプログラムへの `PATH` が通ってることが前提の書き方をしています。

## ライトカーブの作り方

[resolve_ana_pixel_ql_mklc_binned_sorted_itype_v1.py](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ana/pixel/resolve_ana_pixel_ql_mklc_binned_sorted_itype_v1.py) のコマンドを使用して、ライトカーブを作成します：

```sh
resolve_ana_pixel_ql_mklc_binned_sorted_itype_v1.py xa000138000rsl_p0px1000_cl.evt 
```

`--timebinsize TIMEBINSIZE` オプションを使用して、時間ビン（秒）を変更できます。

## スペクトルの作成方法 (非タイルプロット)

[resolve_ana_pixel_ql_plotspec.py](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ana/pixel/resolve_ana_pixel_ql_plotspec.py) のコマンドを使用して、スペクトルを作成します：

```sh
resolve_ana_pixel_ql_plotspec.py xa000138000rsl_p0px1000_cl.evt
```

このコマンドは、pixel ごと、itype ごとにスペクトルをプロットします。

binsize を 4 (PI空間なので、2eVに相当)、emin 0 eV から emax 20000 eV までプロットを指定する場合は、

```sh
resolve_ana_pixel_ql_plotspec.py xa300049010rsl_p0px3000_cl.evt --rebin 4 --emin 0 --emax 20000
```

となります。

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
resolve_ana_pixel_deltat_distribution.py xa000138000rsl_p0px1000_cl.evt
```

下記のように delta T 分布と、TIME vs. PHA のグラフが生成される。

```
deltaT_histogram_linear_linear_xa000109000rsl_p0px1000_cl_TIME.png
deltaT_histogram_linear_linear_xa000109000rsl_p0px1000_cl_TIME_nearzero.png
lightcurve_pha_all_PIXEL_Hp_xa000109000rsl_p0px1000_cl_TIME.png
lightcurve_pha_all_PIXEL_Lp_xa000109000rsl_p0px1000_cl_TIME.png
lightcurve_pha_all_PIXEL_Ls_xa000109000rsl_p0px1000_cl_TIME.png
lightcurve_pha_all_PIXEL_Mp_xa000109000rsl_p0px1000_cl_TIME.png
lightcurve_pha_all_PIXEL_Ms_xa000109000rsl_p0px1000_cl_TIME.png
lightcurve_pha_all_PIXEL_all_types_xa000109000rsl_p0px1000_cl_TIME.png
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

## 広帯域でのスペクトルの比を作成

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ana/pixel/resolve_ana_pixel_plot_6x6_energyspectrum_by_itype.py

で、`-r` オプションをつけると、中心４画素の pixel 平均との比を作ってくれる。`-l 2000 -x 12000` で 2000 - 12000 eV の範囲で、`-b 250` で 250 eV ビンまとめ。`-c` で pixel ごとに autoscale、`-g` で縦軸をリニア表示する。

``` bash:
resolve_ana_pixel_plot_6x6_energyspectrum_by_itype.py xa000102000rsl_p0px1000_cl.evt  -r -y 0 -l 2000 -x 12000 -b 250 -c -g
```


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


## NEXT_INTERVAL 以下の分布と、clipped event or saturated event の関係の調査

NEXT_INTERVAL が設定されているのが前提条件。

[resolve_ana_pixel_Ls_mksubgroup_using_saturatedflags.py](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ana/pixel/resolve_ana_pixel_Ls_mksubgroup_using_saturatedflags.py) に、NEXT_INTERVAL を含む event ファイルを与える。


``` bash:
resolve_ana_pixel_Ls_mksubgroup_using_saturatedflags.py xa097093500rsl_a0pxpr_uf_fillprenext.evt TIME NEXT_INTERVAL -f "PIXEL==23"  -p 
``` 

pixel23に関して、TIMEを横軸、NEXT_INTERVAL が 100 以下のイベントが連続したかどうかでクラスタリング判定を行い、クラスタリング上限を満たしたイベントの中で、      

``` bash:
    parser.add_argument("--trigcol1name", type=str, default="LO_RES_PH", help="The colomn used for trrigger condition, default is LO_RES_PH > CLIPTHRES")
    parser.add_argument("--trigcol2name", type=str, default="ITYPE", help="trigger column 2")
    parser.add_argument("--trigcol3name", type=str, default="RISE_TIME", help="trigger column 3")
    parser.add_argument("--trigcol4name", type=str, default="PHA", help="trigger column 4")
```

の4つのコラムの相関図を自動で生成する。

``` bash:
subset_xa097093500rsl_a0pxpr_uf_fillprenext_PIXEL23.png
comptcols_subset_xa097093500rsl_a0pxpr_uf_fillprenext_PIXEL23.png
comp_subset_xa097093500rsl_a0pxpr_uf_fillprenext_PIXEL23.png
```

`subset_xa097093500rsl_a0pxpr_uf_fillprenext_PIXEL23.png` は、grouping を行い、

- grouping に適合した event を、TIME vs. PREV_INTERVAL で表示
- grouping の長さ
- grouping に適合した NEXT_INTERVAL のヒストグラムの生成

の図を生成する。

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ana/pixel/resolve_run_ana_pixel_Ls_mksubgroup_using_saturatedflags.sh

は全pixelに対して、`resolve_ana_pixel_Ls_mksubgroup_using_saturatedflags.py evtfile LO_RES_PH NEXT_INTERVAL` を一気に行う。

``` bash: 
resolve_run_ana_pixel_Ls_mksubgroup_using_saturatedflags.sh xa097093500rsl_a0pxpr_uf_fillprenext.evt
``` 

### negative lo_res_ph の凝った解析

primary の pulse を引いて、そこからは、25 tick ずつ差分のpeakを取ってくるもの。。いずれにせよ、prelimary version ...

``` bash
resolve_ana_pixel_pr_plot_derivsub_negativeloresph.py xa097093500rsl_a0pxpr_uf_fillprenext_oneclip_p23_1Lp_4Ls.evt -p
```


これは、かなり絞った波形を入れると、コラム分割して、一つ一つのパルスをプロットしてくれる。


``` bash:
resolve_pr_plot_timeseries_negativeloresph.py xa097093500rsl_a0pxpr_uf_fillprenext_oneclip_p23_1Lp_4Ls.evt -p23 -s -c 
``` 


## uf.evt と pr.evt に prev/next interval をつけて、cl.gti でカットする方法

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

### uf.evt と pr.evt に prev/next interval だけつける方法

``` bash:
resolve_tool_pr_prevnextadd.sh xa097093120rsl_a0px_uf.evt xa097093120rsl_a0pxpr_uf.evt  
```



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
resolve_ana_pixel_pr_plot.py xa000114000rsl_a0pxpr_uf_fillprenext_cutclgti_itype_all_slope_b01_quick_b01_pixel_all.fits --deriv
# prev_interval を表示する(波形の数が少ない時のみ使う)
resolve_ana_pixel_pr_plot.py xa000114000rsl_a0pxpr_uf_fillprenext_cutclgti_itype_all_slope_b01_quick_b01_pixel_all.fits --prevflag
# 範囲を絞る場合
resolve_ana_pixel_pr_plot.py xa000114000rsl_a0pxpr_uf_fillprenext_below30.evt --prevflag --xlims 0,500 --ylims=-5000,-2000      
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

## fplot の使い方

- NEXT/PREV_INTERVAL の確認の例

``` bash:
resolve_util_fplot.py xa097093500rsl_a0pxpr_uf_fillprenext_p13.evt ITYPE 1,1 PREV_INTERVAL,NEXT_INTERVAL 1,1 -p
```


# 論文執筆前のデータチェック用ツールのルーティン

## ghf のチェック


https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ecal/resolve_ecal_plot_ghf_with_FWE.py

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ecal/resolve_ecal_plot_ghf_detail.py

の２つのスクリプトと使います。

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

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_util_gtiplot.py

- tel.gti の抜けが気になる時


``` bash:
resolve_util_gtiplot.py xa300036010rsl_tel.gti,xa300036010rsl_p0px1000_uf.evt -p -e xa300036010rsl_p0px1000_cl.evt,xa300036010rsl_p0px1000_uf.evt
``` 

- PSP limit が気になる時


``` bash:
resolve_util_gtiplot.py xa300036010rsl_p0px1000_cl.evt,../xa300036010rsl_el.gti -p -e xa300036010rsl_a0ac_uf.evt -c r
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

## 波形処理のパラメータの確認方法

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_util_dumphk1.py

``` bash:
python resolve_util_dumphk1.py xa300049010rsl_a0.hk1
```

## 時間に沿った波形のプロット

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/psp/pulse/resolve_pr_plot_timeseries.py

``` bash:
resolve_pr_plot_timeseries.py xa097093500rsl_a0pxpr_uf_fillprenext_oneclip.evt -s -p 23 -c
``` 

`-s`　でプロットを表示する。`-p` でピクセルを指定できる。`-c` で色をパルスごとに変える。


## 微分波形を差し引く

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ana/pixel/resolve_ana_pixel_pr_plot_derivsub.py

``` bash:
resolve_ana_pixel_pr_plot_derivsub.py xa097093500rsl_a0pxpr_uf_fillprenext_itype_4_slope_b01_quick_b00_pixel_all.fits -p
```

平均微分波形を差し引いてプロットする。


## slopediffer, quickdouble, itype で場合分けしたイベントファイルを生成する

``` bash:
resolve_util_ftselect_split_file_quickdouble_slopediffer_itype.sh
``` 


## イベントファイルが閾値を満たしたときに、その前後の時間のイベントファイルを保存する方法

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_util_cut_eventfile.py

https://qiita.com/yamadasuzaku/items/08ab56fc8d5549089c78

``` bash:
python resolve_util_cut_eventfile.py mock_fits.evt --threshold 60000 --timewindow 2
```


# reprocess 関係

## rslpipeline の使い方

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_util_run_rslpipeline.sh


``` bash
resolve_util_run_rslpipeline.sh 300049010 20241023
```

とすると、2024.10.23現在での pipepline のオプションでリプロセスを実行して、
`repro_300049010_20241023` のようなディレクトリに生成物を全部生成し、`/cleanup_output` 以下に中間生成物を全て入れる(debug用です)。


