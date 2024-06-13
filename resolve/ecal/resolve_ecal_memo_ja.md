# Resolve データの energy cal の fine tuming 方法

## ピクセル毎、Hp only の TIME, PHA, PI を csv ファイルに保存する

``` bash:
./resolve_ecal_pha_pi.py xa000126000rsl_p0px1000_cl.evt TIME 1,1 PHA,PI 1,1 --filters PIXEL==0,ITYPE==0 -o 00
```

- `TIME 1,1` TIMEコラムを FITS の EXTENSION 1 から取得する。`1,1`  は次の引数で別々の extention の場合に適用できるようにしたため。
- `PHA,PI 1,1`  PHA, PI をFITS の EXTENSION 1 から取得する
- `--filters PIXEL==0,ITYPE==0` PIXEL 0 と Hp だけから取得する。
- `-o 00`  ファイル名に、00 をつける。pixel 0 なので 00 をつけるようにしただけ。

## 多項式でフィットして、残差の時間変化を調べる

``` bash:
python resolve_ecal_fitpoly_csv.py fplot_xa000126000rsl_p0px1000_cl_p00.csv PHA PI 10000 4 4 fitpoly_p00.png
```

- `PHA PI 10000 4 4 fitpoly_p00.png` PHA と PI を csv ファイルから抜き出し、PHA=10000の前後で２つの多項式に分けて、それぞれ４次で近似する。

