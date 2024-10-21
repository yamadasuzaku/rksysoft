# Resolve Ls Event の cal 関係

下記は 2024.9月以前のcal関係で、将来的には obsolete する予定です。

## Ls の PREV_INTERVALの分布を作成する

[resolve_run_ana_pixel_Ls_mksubgroup.sh](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ana/pixel/resolve_run_ana_pixel_Ls_mksubgroup.sh) に、PREV_INTERVAL を含むイベントファイルを与えると、Ls の grouping をして、

- grouping に適合した event を、time vs. PREV_INTERVAL で表示
- grouping の長さ
- grouping に適合した PREV_INTERVAL のヒストグラムの生成

をやります。


```sh
resolve_run_ana_pixel_Ls_mksubgroup.sh xa000114000rsl_p0px1000_uf_prevnext_cutclgti.fits
```

## PREV_INTERVAL を pulse record につける方法

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/README_add_prevnext_interval.md

を参考に、

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_tool_pr_prevnextadd.sh

``` bash:
resolve_tool_pr_prevnextadd.py xa000114000rsl_p0px1000_uf.evt xa000114000rsl_a0pxpr_uf.evt
```

のように、PREV_INTERVAL を生成するイベント(uf.evt)、PREV_INTERVAL を map するイベント(pxpr_uf.evt) を指定するだけ。
このプログラムでは、PREV_INTERVAL はつけるが、gti カットはしない。


## cl.evt の GTI をかける方法


[resolve_util_ahscreen_clgti.sh](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_util_ahscreen_clgti.sh) に、カットをかけたイベントデータ、"GTI"を含むevtファイルを与える

``` bash:
resolve_util_ahscreen_clgti.sh xa000114000rsl_a0pxpr_uf_fillprenext.evt xa000114000rsl_p0px1000_cl.evt
```

## PREV_INTERVAL のカット

ftselect を使ったので、Exposureは全く操作されないことに注意が必要。

``` bash:
ftselect infile=xa000114000rsl_a0pxpr_uf_fillprenext.evt outfile=xa000114000rsl_a0pxpr_uf_fillprenext_below30.evt expr="(PREV_INTERVAL<30)&&(PREV_INTERVAL>0)" chatter=5 clobber=yes
```

## pulserecord のプロット

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ana/pixel/resolve_ana_pixel_pr_plot.py

を使う。


単純なプロットは、

``` bash:
resolve_ana_pixel_pr_plot.py xa000114000rsl_a0pxpr_uf_fillprenext.evt
```

とするだけ。


PREV_INTERVAL を表示したい場合は、

``` bash:
#Example 2) Create a new file:
resolve_ana_pixel_pr_plot.py xa000114000rsl_a0pxpr_uf_fillprenext_below30.evt --prevflag --xlims 0,500 --ylims=-5000,-2000      
#[Note] --ylims=-5000,-2000 should work but --ylims -5000,-2000 NOT work. 
```

のように、--prevflag が必要。マイナスの範囲をしている場合は、`--ylims=-5000,-2000` のように `=` で指定する必要がある。