# Resolve データをクイックチェックするスクリプトの説明

必要なプログラムに `PATH` が通っていることを前提にした書き方をしています。`PATH` を通さずに使う場合は、`chmod +x` で実行権限を与えてからの `./プログラム` で実行ください。ただし、`.sh` については `python` のプログラムへの `PATH` が通ってることが前提の書き方をしています。

- PATH の設定方法

rksysoft の場所を `RESOLVETOOLS` の環境変数にセットして、下記を `.bashrc` や `.zshrc` に書いておくと、resolve, xtend 関係のプログラムに PATH が通ります。これが設定されている状態で動作させることを想定しています。


``` bash 
# add for resolve tool                                                                                                          
export RESOLVETOOLS=/Users/syamada/work/software/mytool/gitsoft/rksysoft
# RESOLVETOOLS/resolve以下のすべてのディレクトリをPATHに追加                                                                    
for dir in $(find "$RESOLVETOOLS/resolve" -type d); do
  PATH="$dir:$PATH"
done
export PATH

# RESOLVETOOLS/xtend以下のすべてのディレクトリをPATHに追加                                                                      
for dir in $(find "$RESOLVETOOLS/xtend" -type d); do
  PATH="$dir:$PATH"
done
export PATH

# RESOLVETOOLS/xrism以下のすべてのディレクトリをPATHに追加                                                                      
for dir in $(find "$RESOLVETOOLS/xrism" -type d); do
  PATH="$dir:$PATH"
done
export PATH
```

- ファイルの初期状態

現時点では、ファイルは、gunzip してあるファイルをみることを前提としています。


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

ここでの delta T は、全pixelを使った deltaT なので、pixel 毎ではないことに注意。frame event のような多画素同時イベントの解析が主目的です。
(pixel毎はPREV/NEXT_INTERVALを使うことにする。)

[resolve_ana_pixel_deltat_risetime_distribution.py](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ana/pixel/resolve_ana_pixel_deltat_risetime_distribution.py) のコマンドを使用して、delta T vs risetime の分布をチェックします：

```sh
resolve_ana_pixel_deltat_risetime_distribution.py --fname xa000138000rsl_p0px1000_cl.evt
```

``` bash:
..... deltaT_risetime_linear_linear_xa000114000rsl_p0px1000_cl_TIME.png is created.
..... deltaT_pha_wide_linear_linear_xa000114000rsl_p0px1000_cl_TIME.png is created.
..... deltaT_pha_narrow_linear_linear_xa000114000rsl_p0px1000_cl_TIME.png is created.
..... pha_risetime_linear_linear_xa000114000rsl_p0px1000_cl_TIME.png is created.
..... lightcurve_dt_all_PIXEL_all_types_xa000114000rsl_p0px1000_cl_TIME.png is created.
..... lightcurve_rt_all_PIXEL_all_types_xa000114000rsl_p0px1000_cl_TIME.png is created.
``` 


## pixel 毎、grade 毎のカウント数のチェック

[resolve_util_check_pixelrate.sh](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_util_check_pixelrate_plot.sh) のコマンドを使用して、pixel 毎、grade 毎のカウント数をチェックします：


```sh
resolve_util_check_pixelrate.sh xa000138000rsl_p0px1000_cl.evt
```


## pixel毎、全gradeの数と割合を表示する

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_util_stat_itype.py

``` bash:
resolve_util_stat_itype.py xa300049010rsl_p0px5000_uf.evt
```

## grade 毎の pixel map と、pixel ごとのカウント数の  1D plot を生成する

[resolve_plot_detxdety.py](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_plot_detxdety.py) を利用して、カウント数が map の上に表示されます。

``` sh:
resolve_plot_detxdety.py xa300049010rsl_p0px3000_cl.evt
```

PHA のカットをかけられるようになっています。

``` sh:
resolve_plot_detxdety.py xa300049010rsl_p0px3000_cl.evt -min 60000 -max 65537
```

二つの event ファイルを引数に与えると、それぞれの pixel map、1D plot、に加えて、最初のファイル - ２つ目のファイル (data - backgroundのイメージ) の差分をとった map  と 1D plot も生成します。

``` sh:
resolve_plot_detxdety_test.py addcluster_xa000114000rsl_p0px1000_uf_noBL_prevnext_cutclgti.evt im0_addcluster_xa000114000rsl_p0px1000_uf_noBL_prevnext_cutclgti.evt
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
- xa300049010rsl_p0px3000_uf.evt

の２つのファイルがある状態で、

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ana/pixel/resolve_ana_run_addprevnext_Lscheck.sh

``` bash:
resolve_ana_run_addprevnext_Lscheck.sh xa300049010rsl_p0px3000_uf.evt
``` 

と実行すると、prev/next interval をつけて、Ls の quick check (`tolerance=100`(default)以下の連続したイベントの数の分布を計算)をしてくれる。
(2025.01.28) これはちょっと古いので、clustering は下記に改訂

### Ls の clustering 

**clustering** を実行するスクリプト

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ana/pixel/resolve_ana_pixel_Ls_define_cluster.py

ファイル名に従って、**clustering** を実行し、ftselect で event ファイルの分割まで実行する script 

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ana/pixel/resolve_ana_pixel_run_Ls_define_cluster.sh


- (時間がある時) 全 pixel に対して clustering を実行する。`-a` をつけて、`-f` で evt ファイルを指定する。


```
resolve_ana_pixel_run_Ls_define_cluster.sh -f xa097093500rsl_a0pxpr_uf_fillprenext.evt -a
```


- (時間がない時) pixel 13 だけ、fselect でカットして、その evt file 対して clustering を実行する。`-p` で pixel 番号を指定する。

```
resolve_ana_pixel_run_Ls_define_cluster.sh -f xa097093500rsl_a0pxpr_uf_fillprenext.evt -p 33 
```

- 結果の確認方法

DERIV_MAX,IMEMBER,ICLUSTER vs. TIME の分布を見て、正常動作かを確認する。


```
resolve_util_fplot.py addcluster_xa097093500rsl_a0pxpr_uf_fillprenext.evt TIME 1,1,1 DERIV_MAX,IMEMBER,ICLUSTER 1,1,1 -p --filter "PIXEL==12,IMEMBER>0" -o pixelall -k 3 
```


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

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_util_fplot.py

を、

``` bash:
resolve_util_fplot.py $input_file TIME 7,6,6,6,6 XBOXA_TEMP3_CAL,HE_TANK2,JT_SHLD2,CAMC_CT1,ADRC_CT_MON_FLUC 7,6,6,6,6  -p -m . -s linear,linear,linear,log,log
```

のようにして使ってます。


## fplot の使い方

条件分岐の方法については、下記の記事を参考にしてください。

- PythonでFITSファイルを効率的にフィルタリングする方法（不等号条件にも対応）

https://qiita.com/yamadasuzaku/items/a4a4343698d90ab6ff20

- numpy.isin でデータを一気にフィルタリングする方法

https://qiita.com/yamadasuzaku/items/03310c063a1675abab25

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

``` bash:
resolve_ecal_plot_ghf_with_FWE.py min60_dshift12_xa300049010rsl_p0px5000_uf.ghf --hk1 xa300049010rsl_a0.hk1 -p -s
``` 

`-p` をつけると、paper 用のスタイルになる(現状、Cyg X-1 の専用にチューニングしてる。)。

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

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ana/pixel/resolve_ana_pixel_mkspec_eachgti_EPI2.py

``` bash:
resolve_ana_pixel_mkspec_eachgti.py xa300036010rsl_p0px1000_cl2_filt.evt -i 6600 -x 7200 -y 0 -m 5 -r 4 -t -v 0.002 --ymin 0.002 --ymax 0.017
```

- "-i 6600 -x 7200" : 6600 eV から 7200 eV まで表示
- "-y 0"    ITYPE==0 (=Hp) のみ表示
- " -m 5"   GTIを５つ分積算する
- "-r 4"     エネルギーのビンまとめと 4eV にする
- "-t"        スペクトルの縦軸に offset を用いて表示する
- "-v 0.002"  オフセット量
- "--ymin 0.002 --ymax 0.017"  y軸の範囲



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


# utility 関係

## ftselect 

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_util_ftselect.sh

``` bash
resolve_util_ftselect.sh xa000114000rsl_p0px1000_uf_prevnext_cutclgti.fits "ITYPE==0" itype0
```

## quick に あるファイルの GTI を別のファイルに適用したい場合

引数の一つ目のファイルの GTI を２つ目のファイルに ftselect でカットをかける。xselect ではないので、exposureなどは正しく設定されないことに注意。

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_util_screen_ufcl_quick.sh


```
resolve_util_screen_ufcl_quick.sh addcluster_xa000114000rsl_p0px1000_uf_noBL_prevnext_cutclgti.evt xa000114000rsl_a0pxpr_uf_fillprenext.evt

```

## STATUS を grade ごとに分離してプロットする


https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_util_stat_status_itype.py

```
resolve_util_stat_status_itype.py xa300049010rsl_p0px5000_uf.evt 
```

- 高速化したバージョン

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_util_stat_status_itype_fast.py


```
resolve_util_stat_status_itype_fast.py xa300049010rsl_p0px5000_uf.evt 
```


## uf イベントから、イベントセレクションをかけて cl イベントに徐々に近づけたイベントを生成する

`xa300049010rsl_p0px3000_uf.evt` と `xa300049010rsl_p0px3000_cl.evt` の２つが存在する状態で、uf.evt の方を指定するだけでよい。


https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_util_screen_ufcl_std.sh


``` bash
resolve_util_screen_ufcl_std.sh xa300049010rsl_p0px3000_uf.evt
```

を実行する。ただし、gti カットは ftselect を使っているので、exposure は補正されてないことに注意。cal 用である。


## イベントのファイルのリストから、1D ヒストグラムを生成する方法


https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ana/pixel/resolve_ana_pixel_hist1d_many_eventfiles.py


``` bash
resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI -p --xmin 0 --xmax 20000 --rebin 250 -i 0 --filters "PIXEL==0" -o pixel0pi
``` 

f.list にイベントファイル名のリストを入れて、コラムと範囲を指定すると、そのヒストグラムと、全subとの比をプロットする。



## xspec の bug 対策

xspec は、PIの範囲、数、がビシッと一致してないと表示できないので、
それを補正するスクリプト。

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_util_mod_TLMIN1_TLMAX1_DETCHANS.sh


```
resolve_util_mod_TLMIN1_TLMAX1_DETCHANS.sh eventファイル
```

中身は、

```
fparkey 0 ${pha} TLMIN1 
fparkey 59999 ${pha} TLMAX1 
fparkey 60000 ${pha} DETCHANS
ftselect ${pha} cut_${pha} '0<=CHANNEL&&CHANNEL<60000'
```

この４つの編集をするだけ。


## SAAの時期だけカットする

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_util_extract_saa_from_ufevt.sh

`gtiinvert infile=${clgti}[STDGTI] outfile=${clinvgti}` を用いて、GTIの invert 条件の時期を取得する。

``` bash:
reseolve_util_extract_saa_from_ufevt.sh xa000114000rsl_p0px1000_cl.evt xa000114000rsl_p0px1000_uf.evt
```

## uf.evt に cl.evt の GTI のカットだけかける。

簡易的な ftselect を使ってるので、exposure を正しくつけたい場合は、xselect を使うこと。

``` bash:
resolve_util_extract_clgti_from_ufevt.sh xa000114000rsl_p0px1000_cl.evt xa000114000rsl_p0px1000_uf.evt
```


## Mn Ka のフィットをするスクリプト

xrism_sdc_tutorial_20240306_v4.pdf のやり方に従って、
- 対角レスポンスを生成する
- Holzer でフィット、gain, gsmooth の sigma (FWHMではない)、norm、の３つの free parameter でフィットする。

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ecal/run_ecal_xspec_fitMnKa.sh

``` bash:
run_ecal_xspec_fitMnKa.sh xa300049010rsl_p0px5000_cl.evt 
```

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ecal/resolve_ana_pixel_ql_fit_MnKa_v2_EPI2.py

``` bash:
resolve_ana_pixel_ql_fit_MnKa_v2_EPI2.py xa300049010rsl_p0px5000_uf_gcor.evt --paper -n timeave_epi2
```

## RVを計算するスクリプト

栗原くん製のプログラムを整理したもの。

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_util_RV_calc.py

```
resolve_util_RV_calc.py 
# ex) Cyg X-1 
$ python resolve_util_RV_calc.py
max of projected_velocity = 6.72, min=-6.71
rv.png is created.
LSR velocity projection: 11.32 km / s
Solar motion projection: 18.19 km / s
Earth orbit velocity projection: 3.55 km / s
``` 

- 地球の公転運動 (30km/s)  (季節依存は含んでない適当なもの)
- 太陽の天の川銀河の公転運動 (220km/s)  https://astro-dic.jp/kinematic-distance/ を使う。
- 太陽の固有運動 (20km/s)

の３つの効果を計算する。

ただし、厳密な計算ではなく、オーダーの目安を与えるものと思うとよい。



## event ファイルの event by event plot の生成方法

- resolve_util_fploteve.py
- run_resolve_util_fploteve_many.sh 

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/run_resolve_util_fploteve_many.sh

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_util_fploteve.py

```
 resolve_util_fploteve.py xa000114000rsl_p0px1000_uf_noBL_prevnext_cutclgti.evt PHA 1,1,1,1 DERIV_MAX,RISE_TIME,EPI,LO_RES_PH 1,1,1,1 --filters "ITYPE==3" -p -o _pixel00_
 ```

 
