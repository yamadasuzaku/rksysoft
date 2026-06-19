# README_resolve_autorun_docal.md

# Resolve autorun QL / calibration workflow

`resolve_autorun_docal.py` は、XRISM Resolve の観測データについて、QL 確認、calibration 確認、簡易スペクトル生成、簡易フィット、HTML レポート生成を OBSID 単位で一括実行するための上位スクリプトです。

この README は、`resolve/autorun/resolve_autorun_docal.py` の使い方と、内部で呼び出される補助スクリプトとの対応関係を説明するためのものです。

---

## 1. このスクリプトの目的

Resolve の観測データでは、いきなり論文用のスペクトル解析に入るのではなく、まず次のような確認が必要です。

- event file が正しく存在しているか
- light curve に異常な jump や gap がないか
- pixel ごとの spectrum が大きくずれていないか
- 6x6 pixel 配置で見たときに異常 pixel がないか
- GTI の切れ方が妥当か
- Fe-55 / Mn Kα 較正線が安定しているか
- ITYPE / STATUS / anti-co の分布に異常がないか
- PHA, RMF, ARF が生成できるか
- 簡易 fit が問題なく走るか

`resolve_autorun_docal.py` は、これらの確認を個別に手で実行する代わりに、OBSID とフラグ指定だけでまとめて実行するための driver script です。

---

## 2. 前提条件

### 2.1 rksysoft に PATH が通っていること

このスクリプトは、内部で多数の補助スクリプトを `subprocess.run()` で呼び出します。

例えば、

- `resolve_ana_pixel_ql_mklc_binned_sorted_itype_v1.py`
- `resolve_ana_pixel_ql_plotspec.py`
- `resolve_ana_pixel_plot_6x6_energyspectrum_by_itype.py`
- `resolve_auto_gen_phaarfrmf.py`
- `resolve_spec_qlfit.py`
- `xrism_autorun_png2html.py`

などです。

そのため、`rksysoft/resolve`, `rksysoft/xtend`, `rksysoft/xrism` 以下のスクリプト群に PATH が通っている必要があります。

例：

```bash
export RESOLVETOOLS=$HOME/work/software/rksysoft

for dir in $(find "$RESOLVETOOLS/resolve" -type d); do
  PATH="$dir:$PATH"
done

for dir in $(find "$RESOLVETOOLS/xtend" -type d); do
  PATH="$dir:$PATH"
done

for dir in $(find "$RESOLVETOOLS/xrism" -type d); do
  PATH="$dir:$PATH"
done

export PATH
```

確認例：

```bash
which resolve_autorun_docal.py
which resolve_ana_pixel_ql_plotspec.py
which xrism_autorun_png2html.py
```

---

### 2.2 FITS ファイルが gunzip されていること

現状のスクリプトは、`.evt`, `.hk1`, `.gti`, `.ehk` などが gunzip 済みで存在することを前提にしています。

カレントディレクトリ以下の `.gz` を一括で展開するには、例えば次のようにします。

```bash
find ./ -type f -name "*.gz" -exec gunzip {} \;
```

---

### 2.3 OBSID ディレクトリが見える場所で実行すること

`resolve_autorun_docal.py` は、実行時のカレントディレクトリを `topdir` として扱います。

```python
topdir = os.getcwd()
```

そのため、OBSID ディレクトリの中ではなく、**OBSID ディレクトリが見える一つ上のディレクトリ**で実行します。

例えば、データが次のようにある場合、

```text
/home/test/data/201068010/
├── auxil/
├── resolve/
│   ├── event_cl/
│   ├── event_uf/
│   └── hk/
└── ...
```

実行場所は次です。

```bash
cd /home/test/data
```

正しい実行場所では、次のようなパスが見えるはずです。

```text
201068010/resolve/event_cl/
201068010/resolve/event_uf/
201068010/resolve/hk/
201068010/auxil/
```

---

## 3. 基本的な実行例

OPEN filter の一般的な例です。

```bash
resolve_autorun_docal.py 201068010 \
  -b 20 -l 2000 -x 9000 \
  --progflags 1,1,1,1,1,1,1,1,1,1,1,1 \
  --calflags 0,0,1,0,0,1,1 \
  --anaflags 1,1,1 \
  --deeplsflags 0,0
```

Cyg X-1 など、ND filter の観測では `--fwe ND` を付けます。

```bash
resolve_autorun_docal.py 300049010 \
  -b 20 -l 2000 -x 9000 \
  --progflags 1,1,1,1,1,1,1,1,1,1,1,1 \
  --calflags 1,1,1,1,1,1,1 \
  --anaflags 1,1,1 \
  --deeplsflags 0,0 \
  --fwe ND
```

---

## 4. 主なコマンドライン引数

| 引数 | 省略形 | デフォルト | 意味 |
|---|---:|---:|---|
| `obsid` | なし | 必須 | 解析対象の OBSID |
| `--timebinsize` | `-t` | `100.0` | light curve の時間 bin 幅 |
| `--itypenames` | `-y` | `0,1,2,3,4` | 対象にする ITYPE |
| `--plotpixels` | `-p` | `0,1,...,35` | plot 対象 pixel |
| `--output` | `-o` | `mklc` | 出力 prefix |
| `--debug` | `-d` | off | debug mode |
| `--show` | `-s` | off | `plt.show()` を呼ぶ |
| `--bin_width` | `-b` | `4` | spectrum / histogram の bin 幅 |
| `--ene_min` | `-l` | `6300` | 表示する最小 energy |
| `--ene_max` | `-x` | `6900` | 表示する最大 energy |
| `--gmin` | なし | `30` | grppha で使う minimum count |
| `--fwe` | なし | `OPEN` | `OPEN` または `ND` |
| `--progflags` | なし | 省略時は全 OFF | 通常 QL 確認の ON/OFF |
| `--calflags` | なし | 省略時は全 OFF | calibration 確認の ON/OFF |
| `--deeplsflags` | なし | 省略時は全 OFF | deep Ls 解析の ON/OFF |
| `--anaflags` | なし | 省略時は全 OFF | 解析 products 生成の ON/OFF |
| `--genhtml` | `-html` | HTML 生成 ON | 現状では HTML 生成を止めるオプション |

---

## 5. 注意：`--genhtml` は名前と動作が逆

現在の実装では、

```python
parser.add_argument('--genhtml', '-html', action='store_false', help='stop generate html')
```

となっています。

つまり、デフォルトでは HTML を生成します。

一方で、

```bash
--genhtml
```

または

```bash
-html
```

を付けると、HTML 生成を止めます。

オプション名だけを見ると「HTML を生成する」ように見えるため、将来的には `--no-html` などに改名した方が分かりやすいです。

---

## 6. FWE と入力 event file 名

`--fwe` は、FWE filter の状態に応じて event file 名の `px` 番号を切り替えるためのオプションです。

| `--fwe` | 内部値 | 代表的な cleaned event file |
|---|---:|---|
| `OPEN` | `1000` | `xa{obsid}rsl_p0px1000_cl.evt` |
| `ND` | `3000` | `xa{obsid}rsl_p0px3000_cl.evt` |

例えば、OBSID が `201068010` で OPEN filter の場合、

```text
xa201068010rsl_p0px1000_cl.evt
```

を探します。

OBSID が `300049010` で ND filter の場合、

```text
xa300049010rsl_p0px3000_cl.evt
```

を探します。

---

## 7. スクリプト内部で使う主なファイル名

`resolve_autorun_docal.py` は OBSID と FWE から、次のようなファイル名を自動生成します。

| 変数名 | ファイル名の例 | 意味 |
|---|---|---|
| `clevt` | `xa201068010rsl_p0px1000_cl.evt` | cleaned event file |
| `clgcorevt` | `xa201068010rsl_p0px1000_clgcor.evt` | rslgain 補正後の cleaned event file |
| `ufevt` | `xa201068010rsl_p0px1000_uf.evt` | unscreened / unfiltered event file |
| `rsla0hk1` | `xa201068010rsl_a0.hk1` | Resolve HK file |
| `ghf` | `xa201068010rsl_000_fe55.ghf` | Fe-55 gain history file |
| `telgti` | `xa201068010rsl_tel.gti` | telescope GTI |
| `uf50evt` | `xa201068010rsl_p0px5000_uf.evt` | Fe-55 などで使う UF event |
| `cl50evt` | `xa201068010rsl_p0px5000_cl.evt` | Fe-55 などで使う CL event |
| `ufacevt` | `xa201068010rsl_a0ac_uf.evt` | anti-co 側 event file |
| `elgti` | `xa201068010rsl_el.gti` | EL GTI |
| `expgti` | `xa201068010rsl_px1000_exp.gti` | exposure GTI |
| `ehk` | `xa201068010.ehk` | enhanced HK file |

---

## 8. `flag_configs` の意味

このスクリプトの中心は、次の辞書です。

```python
flag_configs = {
    "progflags": [
        "qlmklc", "qlmkspec", "spec6x6", "deltat", "deltat-rt-pha", "detxdety",
        "temptrend", "plotghf", "plotgti", "spec-eachgti", "lc-eachgti", "mkbratio"
    ],
    "calflags": [
        "lsdist", "lsdetail", "specratio6x6", "statusitype", "statitype", "antico", "fe55fit"
    ],
    "deeplsflags": ["addprevnext", "defcluster"],
    "anaflags": ["genpharmfarf", "qlfit", "compcluf"]
}
```

これは、`--progflags`, `--calflags`, `--deeplsflags`, `--anaflags` の **カンマ区切り 0/1 列の順番表**です。

例えば、

```bash
--calflags 0,0,1,0,0,1,1
```

は、次の意味です。

```text
lsdist        OFF
lsdetail      OFF
specratio6x6  ON
statusitype   OFF
statitype     OFF
antico        ON
fe55fit       ON
```

内部では、文字列を整数 list に変換し、

```python
[0, 0, 1, 0, 0, 1, 1]
```

さらに、名前付き辞書に変換しています。

```python
{
    "lsdist": False,
    "lsdetail": False,
    "specratio6x6": True,
    "statusitype": False,
    "statitype": False,
    "antico": True,
    "fe55fit": True
}
```

その後、例えば次のように実行分岐します。

```python
if flag_dicts["calflags"]["fe55fit"]:
    ...
```

---

## 9. `progflags`: 通常 QL 確認

`progflags` は、通常の quick look 確認を制御します。

| 番号 | flag | 主に呼ぶスクリプト | 出力ディレクトリ | 内容 |
|---:|---|---|---|---|
| 1 | `qlmklc` | `resolve_ana_pixel_ql_mklc_binned_sorted_itype_v1.py` | `check_qlmklc` | light curve 作成 |
| 2 | `qlmkspec` | `resolve_ana_pixel_ql_plotspec.py` | `check_qlmkspec` | spectrum 作成 |
| 3 | `spec6x6` | `resolve_ana_pixel_plot_6x6_energyspectrum_by_itype.py` | `check_spec6x6` | 6x6 pixel spectrum |
| 4 | `deltat` | `resolve_ana_pixel_deltat_distribution.py` | `check_deltat` | event 間隔分布 |
| 5 | `deltat-rt-pha` | `resolve_ana_pixel_deltat_risetime_distribution.py` | `check_deltat-rt-pha` | event 間隔、rise time、PHA |
| 6 | `detxdety` | `resolve_plot_detxdety.py` | `check_detxdety` | DETX/DETY 分布 |
| 7 | `temptrend` | `resolve_hk_plot_temptrend.sh` | `check_temptrend` | temperature trend |
| 8 | `plotghf` | `resolve_ecal_plot_ghf_detail.py`, `resolve_ecal_plot_ghf_with_FWE.py` | `check_plotghf` | Fe-55 gain history |
| 9 | `plotgti` | `resolve_util_gtiplot.py` | `check_plotgti` | GTI 比較 |
| 10 | `spec-eachgti` | `resolve_ana_pixel_mkspec_eachgti.py` | `check_spec-eachgti` | GTI ごとの spectrum |
| 11 | `lc-eachgti` | `resolve_ana_pixel_mklc_branch.py` | `check_lc-eachgti` | GTI ごとの light curve |
| 12 | `mkbratio` | `resolve_ana_pixel_mklc_branch.py` | `check_mkbratio` | branch / grade 的な比の確認 |

すべて ON にする例：

```bash
--progflags 1,1,1,1,1,1,1,1,1,1,1,1
```

---

## 10. `calflags`: calibration / event classification 確認

`calflags` は、calibration や event classification に近い確認を制御します。

| 番号 | flag | 主に呼ぶスクリプト | 出力ディレクトリ | 内容 |
|---:|---|---|---|---|
| 1 | `lsdist` | `resolve_ana_run_addprevnext_Lscheck.sh` | `checkcal_lsdist` | Ls event と前後 event の関係 |
| 2 | `lsdetail` | `resolve_run_ana_pixel_Ls_mksubgroup_using_saturatedflags.sh` | `checkcal_lsdetail` | Ls event の詳細分類 |
| 3 | `specratio6x6` | `resolve_ana_pixel_plot_6x6_energyspectrum_by_itype.py` | `checkcal_specratio6x6` | 6x6 spectrum ratio |
| 4 | `statusitype` | `resolve_util_stat_status_itype_fast.py` | `checkcal_statusitype` | STATUS と ITYPE |
| 5 | `statitype` | `resolve_util_stat_itype.py` | `checkcal_statitype` | ITYPE 統計 |
| 6 | `antico` | `resolve_ana_antico_comp_ELVhilo.sh` | `checkcal_antico` | anti-co と ELV 条件の比較 |
| 7 | `fe55fit` | `resolve_ana_pixel_ql_fit_MnKa_v2_EPI2.py` | `checkcal_fe55fit` | Fe-55 / Mn Kα fit |

例：

```bash
--calflags 0,0,1,0,0,1,1
```

この場合、次だけが実行されます。

```text
specratio6x6
antico
fe55fit
```

---

## 11. `deeplsflags`: deep Ls / pseudo event 関連

`deeplsflags` は、Ls event や pseudo event を深く調べるための処理を制御します。

| 番号 | flag | 主に呼ぶスクリプト | 出力ディレクトリ | 内容 |
|---:|---|---|---|---|
| 1 | `addprevnext` | `resolve_ana_run_addprevnext_Lscheck_for_deepls.sh` | `checkdeepls_addprevnext` | 前後 event 情報を追加 |
| 2 | `defcluster` | `resolve_ana_pixel_Ls_define_cluster.py` | `checkdeepls_defcluster` | Ls cluster 定義 |

通常の QL 確認では OFF でよいです。

```bash
--deeplsflags 0,0
```

Ls / pseudo event の詳細調査を行う場合は ON にします。

```bash
--deeplsflags 1,1
```

---

## 12. `anaflags`: analysis products 生成

`anaflags` は、簡易解析用の products 作成と確認を制御します。

| 番号 | flag | 主に呼ぶスクリプト | 出力ディレクトリ | 内容 |
|---:|---|---|---|---|
| 1 | `genpharmfarf` | `resolve_auto_gen_phaarfrmf.py` | `checkana_genpharmfarf` | PHA, RMF, ARF 生成 |
| 2 | `qlfit` | `resolve_spec_qlfit.py`, `xrism_spec_qlfit_many.py`, `xrism_util_plot_arf.py` | `checkana_qlfit` | 簡易 spectral fit |
| 3 | `compcluf` | `resolve_util_screen_ufcl_std.sh`, `run_resolve_ana_pixel_hist1d_many_eventfiles.sh` | `checkana_compcluf` | UF / CL 比較 |

すべて ON にする例：

```bash
--anaflags 1,1,1
```

---

## 13. `genpharmfarf` で作る products

`anaflags` の `genpharmfarf` を ON にすると、`resolve_auto_gen_phaarfrmf.py` を使って PHA, RMF, ARF を生成します。

生成対象は次の 3 種類です。

| 名前 | pixel | 用途 |
|---|---|---|
| `all` | 全 pixel | 全体 spectrum |
| `inner` | `0,17,18,35` | 中心 4 pixel |
| `outer` | outer pixel 群 | 外側 pixel |

`outer` では、現在のスクリプトでは次の pixel を使います。

```text
1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34
```

pixel 12 は除外されています。

代表的な出力ファイルは次のようになります。

```text
rsl_source_Hp_all_gopt.pha
rsl_source_Hp_all.rmf
rsl_source_Hp_all.arf

rsl_source_Hp_inner_gopt.pha
rsl_source_Hp_inner.rmf
rsl_source_Hp_inner.arf

rsl_source_Hp_outer_gopt.pha
rsl_source_Hp_outer.rmf
rsl_source_Hp_outer.arf
```

---

## 14. Cyg X-1 用の special case

スクリプト内では、OBSID が `300049010` の場合だけ特別扱いがあります。

```python
special_case = (obsid == "300049010")
```

この場合、

```text
event_cl/
```

ではなく、

```text
event_cl_rslgain/
```

を使う処理があります。

また、cleaned event file として、

```text
xa300049010rsl_p0px3000_clgcor.evt
```

のような gain-corrected file を使う分岐があります。

この処理は Cyg X-1 SWG 解析向けの特別対応です。一般の OBSID では通常の `event_cl/` を使います。

---

## 15. `dojob()` の動作

各処理は、最終的に `dojob()` から実行されます。

`dojob()` は次のことを行います。

1. 実行する補助スクリプトが PATH にあるか確認する
2. `gdir` で指定された作業ディレクトリへ移動する
3. 必要なら `subdir` を作る
4. 必要な入力ファイルへの symbolic link を作る
5. 補助スクリプトを `subprocess.run()` で実行する
6. 元の top directory に戻る
7. 開始時刻と終了時刻を表示する

例えば `qlmklc` では、概念的には次のような処理を行います。

```bash
cd 201068010/resolve/event_cl/
mkdir -p check_qlmklc
cd check_qlmklc
ln -s ../xa201068010rsl_p0px1000_cl.evt .
resolve_ana_pixel_ql_mklc_binned_sorted_itype_v1.py \
  xa201068010rsl_p0px1000_cl.evt \
  --timebinsize 100.0
cd <topdir>
```

このように、処理ごとに `check_*` ディレクトリが作られ、その中に結果が保存されます。

---

## 16. 出力ディレクトリの命名規則

出力ディレクトリは、おおむね次の prefix で分類されます。

| prefix | 内容 |
|---|---|
| `check_` | 通常の QL 確認 |
| `checkcal_` | calibration / event classification 関連 |
| `checkdeepls_` | deep Ls / pseudo event 関連 |
| `checkana_` | analysis products / 簡易 fit 関連 |

例：

```text
201068010/resolve/event_cl/check_qlmklc/
201068010/resolve/event_cl/check_qlmkspec/
201068010/resolve/event_cl/check_spec6x6/
201068010/resolve/event_uf/checkcal_antico/
201068010/resolve/event_uf/checkcal_fe55fit/
201068010/resolve/event_cl/checkana_genpharmfarf/
```

---

## 17. HTML report の生成

デフォルトでは、最後に `xrism_autorun_png2html.py` を使って、png を HTML にまとめます。

内部では次のような処理を実行します。

```bash
xrism_autorun_png2html.py 201068010 --keyword check_       --ver v0
xrism_autorun_png2html.py 201068010 --keyword checkcal_    --ver v0
xrism_autorun_png2html.py 201068010 --keyword checkdeepls_ --ver v0
xrism_autorun_png2html.py 201068010 --keyword checkana_    --ver v0
```

HTML を探すには、実行後に次のようにします。

```bash
find 201068010 -name "*.html"
```

---

## 18. 実行ログを残す推奨 wrapper

このスクリプトは多数の補助スクリプトを順に実行するため、標準出力が長くなります。

実運用では、次のような wrapper script を作って、`tee` でログを残すことを推奨します。

```bash
#!/bin/bash

set -u

obsid=201068010

logdir=logs
mkdir -p "$logdir"

logfile="${logdir}/resolve_autorun_docal_${obsid}_$(date +%Y%m%d_%H%M%S).log"

resolve_autorun_docal.py "$obsid" \
  -b 20 -l 2000 -x 9000 \
  --progflags 1,1,1,1,1,1,1,1,1,1,1,1 \
  --calflags 0,0,1,0,0,1,1 \
  --anaflags 1,1,1 \
  --deeplsflags 0,0 \
  2>&1 | tee "$logfile"
```

実行権限を付けます。

```bash
chmod +x auto.sh
```

実行します。

```bash
./auto.sh
```

ログ確認例：

```bash
grep -i error logs/resolve_autorun_docal_201068010_*.log
grep -i "not found" logs/resolve_autorun_docal_201068010_*.log
```

---

## 19. 典型的な使い方

### 19.1 通常の QL 確認を一通り行う

```bash
resolve_autorun_docal.py 201068010 \
  -b 20 -l 2000 -x 9000 \
  --progflags 1,1,1,1,1,1,1,1,1,1,1,1 \
  --calflags 0,0,1,0,0,1,1 \
  --anaflags 1,1,1 \
  --deeplsflags 0,0
```

まずはこの程度を基本設定にするとよいです。

### 19.2 Ls / pseudo event まで詳しく見る

```bash
resolve_autorun_docal.py 201068010 \
  -b 20 -l 2000 -x 9000 \
  --progflags 1,1,1,1,1,1,1,1,1,1,1,1 \
  --calflags 1,1,1,1,1,1,1 \
  --anaflags 1,1,1 \
  --deeplsflags 1,1
```

Ls event や pseudo event の混入が疑われる場合の重めの設定です。

### 19.3 light curve と spectrum だけ軽く見る

```bash
resolve_autorun_docal.py 201068010 \
  -b 20 -l 2000 -x 9000 \
  --progflags 1,1,1,0,0,0,0,0,0,0,0,0 \
  --calflags 0,0,0,0,0,0,0 \
  --anaflags 0,0,0 \
  --deeplsflags 0,0
```

最初に軽く event file の様子だけ見たい場合に使えます。

### 19.4 PHA/RMF/ARF と簡易 fit だけやり直す

```bash
resolve_autorun_docal.py 201068010 \
  --progflags 0,0,0,0,0,0,0,0,0,0,0,0 \
  --calflags 0,0,0,0,0,0,0 \
  --anaflags 1,1,0 \
  --deeplsflags 0,0
```

QL 図はすでに作成済みで、response と quick fit だけ再実行したい場合に使えます。

---

## 20. 最初に見るべき結果

大量の図が生成されるため、最初は次の順番で確認するとよいです。

### 20.1 `check_qlmklc`

light curve を確認します。

見るべき点：

- 急激な count rate jump がないか
- GTI gap と light curve の欠損が対応しているか
- pixel ごとの rate が極端に違わないか
- ITYPE の比率が時間的に大きく変わっていないか

### 20.2 `check_spec6x6`

6x6 pixel 配置の spectrum を確認します。

見るべき点：

- 特定 pixel だけ spectrum が異常でないか
- 中心 pixel と外側 pixel に大きな違いがあるか
- Fe-K 周辺に構造があるか
- 低 energy / 高 energy 側で異常 pixel がないか

### 20.3 `check_plotgti`

GTI の関係を確認します。

見るべき点：

- UF と CL の違い
- telescope GTI と event GTI の対応
- exposure GTI の妥当性
- 極端に短い GTI が大量にないか

### 20.4 `check_plotghf` と `checkcal_fe55fit`

Fe-55 / Mn Kα 較正線を確認します。

見るべき点：

- gain history に jump がないか
- Mn Kα fit が通っているか
- pixel ごとに中心値や幅が大きくずれていないか
- 温度変化と相関するような変動がないか

### 20.5 `checkana_genpharmfarf` と `checkana_qlfit`

PHA/RMF/ARF と quick fit を確認します。

見るべき点：

- all / inner / outer の products が生成されているか
- RMF / ARF が PHA と対応しているか
- quick fit が失敗していないか
- Fe-K 周辺の狭帯域 fit で明らかな異常がないか

---

## 21. よくあるエラーと対処

### 21.1 `not found in $PATH`

例：

```text
Error: resolve_ana_pixel_ql_plotspec.py not found in $PATH.
```

補助スクリプトに PATH が通っていません。

確認：

```bash
echo $RESOLVETOOLS
which resolve_ana_pixel_ql_plotspec.py
```

PATH 設定を見直してください。

---

### 21.2 `The directory '201068010/resolve/event_cl/' does not exist.`

実行場所が間違っている可能性があります。

正しい例：

```bash
cd /home/test/data
resolve_autorun_docal.py 201068010 ...
```

間違った例：

```bash
cd /home/test/data/201068010
resolve_autorun_docal.py 201068010 ...
```

このスクリプトは、OBSID ディレクトリの一つ上で実行する想定です。

---

### 21.3 event file が見つからない

`--fwe` の指定が合っていない可能性があります。

OPEN filter なら、

```text
xa{obsid}rsl_p0px1000_cl.evt
```

ND filter なら、

```text
xa{obsid}rsl_p0px3000_cl.evt
```

を探します。

ND filter 観測なのに `--fwe ND` を付け忘れると、`px1000` 側を探しに行ってしまいます。

---

### 21.4 `.gz` のままで見つからない

現状では gunzip 済みを前提にしています。

```bash
find ./ -type f -name "*.gz" -exec gunzip {} \;
```

で展開してください。

---

## 22. 開発上の注意点

### 22.1 フラグ列は便利だが読みにくい

例えば、

```bash
--progflags 1,1,1,1,1,1,1,1,1,1,1,1
```

は短く書けますが、何が ON なのか一目では分かりません。

運用上は、

- README の対応表を見る
- 目的別の `auto.sh` を用意する
- 実行ログに flag dictionary の出力を残す

ことを推奨します。

### 22.2 補助スクリプトの失敗後も次へ進む可能性がある

`dojob()` では、`subprocess.CalledProcessError` を捕まえて表示しています。

```python
except subprocess.CalledProcessError as e:
    print(f"Error occurred: {e}")
```

そのため、ある補助スクリプトが失敗しても、処理全体がそこで止まらず、次に進む場合があります。

ログを必ず確認してください。

### 22.3 出力の上書きに注意

一部の処理では、同じ `check_*` ディレクトリ内で複数回コマンドを実行します。

同名の出力ファイルがある場合、補助スクリプト側の実装によっては上書きされる可能性があります。

重要な結果は、ログや HTML と合わせて保存しておくことを推奨します。

---

## 23. このスクリプトがやること、やらないこと

### やること

- OBSID から入力 file 名を組み立てる
- QL 確認用 plot を作る
- calibration / event classification 確認を行う
- Ls / pseudo event 関連の補助解析を走らせる
- PHA/RMF/ARF を作る
- quick spectral fit を行う
- png を HTML にまとめる

### やらないこと

- 論文用の最終 spectrum 解析を完成させる
- science case に応じた最終 GTI 選択を自動判断する
- background / systematic uncertainty を完全に評価する
- calibration document との詳細照合を自動化する
- すべての天体に最適な pixel selection を自動決定する

このスクリプトは、最終解析を自動で完成させるものではありません。

あくまで、Resolve データの状態を把握し、次に進むための「健康診断」を一括で行うためのものです。

---

## 24. まとめ

`resolve_autorun_docal.py` は、XRISM Resolve の QL 解析と calibration 確認を OBSID 単位で一括実行する driver script です。

重要な点は次です。

- OBSID ディレクトリが見える一つ上で実行する
- OPEN filter は `px1000`、ND filter は `px3000` を見る
- `--progflags`, `--calflags`, `--deeplsflags`, `--anaflags` は `flag_configs` の順番に対応する
- 出力は `check_`, `checkcal_`, `checkdeepls_`, `checkana_` に分類される
- デフォルトでは HTML report を生成する
- `--genhtml` は現状では HTML 生成を止めるオプションである
- ログを残して、エラーを必ず確認する
- このスクリプトは最終 science analysis ではなく、QL / calibration の健康診断用である

まずはこのスクリプトで全体像を確認し、その後、science case に応じて GTI、pixel、energy band、response、spectral model を改めて選ぶ、という流れで使うのがよいです。
