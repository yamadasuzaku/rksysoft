# resolve_util_screen_for_lscluster.sh README（日本語）

## これは何？
XRISM/Resolve の **ufevt（unfiltered event）** を入力として、

1. **xselect** を使って GTI をマージ・整形したベースイベント（`*_clgti.evt`）を作成し  
2. そのベースに対して **ftcopy の式フィルタ**を複数通り適用して  
3. 条件ごとに別々の event file（`.evt`）を一括生成する

ための Bash スクリプトです。

## 目的（なぜ xselect → ftcopy なのか？）
- **xselect は「GTI をうまく整形・マージしてくれる」**（解析パイプライン上、必要な所作がある）
- 一方で条件を少しずつ変えて複数ファイルを作りたいとき、毎回 xselect を回すのは面倒
- そこで、
  - まず xselect で「ベース（GTI 整形済み）」を1つ作り
  - その後は ftcopy で条件違いのファイルを量産する
という設計になっています。

## 必要環境
- bash（`set -Eeuo pipefail` が使えること）
- HEASOFT / ftools が使える環境
- 以下のコマンドが `$PATH` にあること：
  - `ftlist`
  - `ftcopy`
  - `xselect`

※冒頭で `heasoft_env.sh` を source しています：
```bash
. "$(dirname "$0")/heasoft_env.sh"
````

ここで HEASOFT の初期化をする想定です。

## 使い方

### 実行

```bash
./resolve_util_screen_for_lscluster.sh <input_ufevt.evt>
```

### 例

```bash
./resolve_util_screen_for_lscluster.sh small_large_xa000154000rsl_p0px1000_uf_noBL_prevnext_cutclgti.evt
```

## 生成されるファイル

入力が `AAA.evt` のとき：

### 1) ベース（GTI 整形済み）

* `AAA_clgti.evt`

これは **xselect** が生成します。

### 2) フィルタ適用後の出力（ftcopy）

* `AAA_clgti_stdcut.evt`
* `AAA_clgti_clustercut.evt`
* `AAA_clgti_clustercutstdcut.evt`

※最後の `clustercutstdcut` は「標準カット AND クラスタカット」です。

## スクリプトの処理フロー（ざっくり）

1. **環境チェック**

   * `ftlist`, `ftcopy`, `xselect` が PATH にあるか確認（なければ即終了）

2. **入力ファイル確認**

   * 引数が1つか
   * ファイルが存在するか

3. **xselect でベースイベント生成**

   * `run_xselect_make_clgti <in> <out>`
   * 出力：`*_clgti.evt`
   * ここは **対話コマンド xselect に必要最低限の所作だけ**流し込み

4. **ftlist でヘッダ表示**

   * 入力・出力それぞれで `ftlist <file> H`

5. **定義済みフィルタを順番に ftcopy**

   * `OUT_FILES[]` と `OUT_FILTERS[]` を「同じ index」で対応付け
   * ループで順に生成


## フィルタの書き方ルール（超重要）

このスクリプトではフィルタを「人間が読みやすい複数行」で書き、
最終的に **1行化 + 空白除去**して ftcopy に渡します。

### なぜ空白を消すの？

HEASOFT の PIL（パラメータ解釈）やシェルクォート周りで、
フィルタ式が意図せず分割・誤解釈されて事故ることがあるためです。

### 実装

* `normalize_filter_expr` が担当します：

  * 改行 → スペース
  * 連続スペースを畳む
  * さらに最後に **全ての空白を削除**

つまり、見た目はこう書いても：

```bash
local FILTER_STDCUT='
(ITYPE < 5) &&
(PI >= 600)
'
```

最終的に ftcopy に渡るのはこういう形になります：

```
(ITYPE<5)&&(... )&&(PI>=600)
```


## フィルタ定義の構造

### 1) 標準カット（例）

```bash
local FILTER_STDCUT='
...
'
```

### 2) クラスタ系追加カット（例）

```bash
local FILTER_CLUSTERCUT='
(ITYPE < 5) &&
(ICLUSTERL == 0) &&
(ICLUSTERS == 0)
'
```

### 3) AND 合成（標準カット && クラスタカット）

```bash
local FILTER_CLUSTERCUTSTDCUT="
(
${FILTER_STDCUT}
)
&&
(
${FILTER_CLUSTERCUT}
)
"
```

**ポイント**

* 合成するときは `(...) && (...)` のように **括弧を付けて安全に**する
* FILTER を文字列として埋め込むので、見落としバグ防止のためにも括弧推奨


## 出力ファイルとフィルタの対応（ここを編集する）

このスクリプトは「出力ファイル名」と「使うフィルタ」を配列で管理しています。

```bash
declare -a OUT_FILES=(
  "..._stdcut.evt"
  "..._clustercut.evt"
  "..._clustercutstdcut.evt"
)

declare -a OUT_FILTERS=(
  "$FILTER_STDCUT"
  "$FILTER_CLUSTERCUT"
  "$FILTER_CLUSTERCUTSTDCUT"
)
```

### 追加したい場合

1. `local FILTER_XXXX=' ... '` を増やす
2. `OUT_FILES` に出力名を追加
3. `OUT_FILTERS` に対応するフィルタ変数を追加
   ※配列の順番（index）がズレると事故るので注意


## よくある落とし穴

* **ftcopy の infile 指定が分割される**

  * このスクリプトは `"infile=... [EVENTS][filter]"` を **1引数として渡す**ことで回避しています

* **フィルタ式の括弧不足**

  * `A && (B || C)` のつもりが `A && B || C` になって意味が変わる
  * 合成時は必ず括弧で守る

* **xselect は対話コマンド**

  * 入力ファイルやカレントディレクトリの扱いが環境依存になりやすい
  * このスクリプトでは `./` を明示している（必要なら調整）


## ログ表示について

* INFO（シアン）
* OK（グリーン）
* WARN/ERROR（赤）

エラー時は `die` で即終了します。


## 変更メモ欄（運用ログ用）

* いつ・誰が・どのフィルタを変えたかを書くと、後で「なぜこうした」を思い出せます。

例：

* 2026-01-01: FILTER_CLUSTERCUT に (ICLUSTERL==0) を追加（理由：Ls cluster 除去を強化）
* 2026-01-0X: FILTER_STDCUT の設定変更（理由：低エネルギー偽イベントが増えたため）

## 連絡事項（自分用）

* 「この出力は何のために作った？」を忘れがちなので、

  * 出力ファイル名に目的を入れる
  * README の変更メモに根拠を書く
    を強く推奨。

```