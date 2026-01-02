# XRISM Resolve 擬似イベント・クラスタリングツール

（XRISM Resolve Pseudo-Event Clustering Tool）

## 概要（Overview）

[`resolve_ftools_add_cluster.py`](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/resolve_ftools_add_cluster.py) は、XRISM 衛星の **Resolve** 装置で取得された X 線イベントデータに対して、**擬似イベント（pseudo events）** を同定するための Python 製ツールである。
ここでいう **擬似イベント** とは、天体起源の実 X 線光子によるものではなく、**宇宙線や装置由来の効果**によって引き起こされる誤トリガーを指す。

本ツールは、
[run_cluster_pipeline.sh](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/resolve_ftools_cluster_pileline.sh)
から呼び出されることを想定しており、その詳細は以下の README に記載されている。

* [README_cluster_pileline.md](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/README_cluster_pileline.md)

既知のクロストーク（cross-talk）イベントに加えて、本ツールは **複数ピクセルがほぼ同時に反応するクラスタ化イベント**（例：宇宙線が検出器に入射した場合）も検出できる。
イベントを時間方向にクラスタリングすることで、`resolve_ftools_add_cluster.py` は「ほぼ同時に発生したイベント群」を体系的に同定し、それらにフラグを付与する。

出力は **クラスタ情報が付加されたイベントファイル**であり、

* 孤立した（本物の）X 線イベント
* 同時発生グループに属するイベント（擬似イベントや粒子起源イベント候補）

を明確に区別できるようになる。
このツールは、XRISM Resolve データ解析において擬似イベントを検出・可視化し、**科学解析に真の X 線イベントのみを用いるための前処理**を支援する。

---

## 依存関係（Dependencies）

[`resolve_ftools_add_cluster.py`](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/resolve_ftools_add_cluster.py) を実行するには、以下の Python 環境およびライブラリが必要である。

* **Python 3.x**
  本スクリプトは Python 3 系で書かれている。

* **Astropy**
  FITS ファイルの読み書きに使用（`astropy.io.fits` など）。

* **NumPy**
  数値計算および配列処理（イベント時刻配列の操作など）。

* **Matplotlib**（任意）
  イベントクラスタの診断用プロットを生成する場合に使用。

* **標準ライブラリ**
  `argparse` などの標準ライブラリも使用するが、これらは Python に標準で含まれている。

Anaconda 環境や pip を使用している場合、不足しているライブラリは通常の方法でインストール可能である（例：`pip install astropy numpy matplotlib`）。
XRISM 専用やプロプライエタリなライブラリは不要であり、**標準的な FITS ファイルと Python 科学計算環境のみで動作する**。

---

## 使用方法（Usage）

Resolve のイベント FITS ファイルに対して、コマンドラインから本ツールを実行できる。
基本的には **パイプラインスクリプトを使用することが推奨される**。

```bash
./run_cluster_pipeline.sh <input_file_uf.evt>
```

実行前に、必ず以下の README を熟読すること。

* [README_cluster_pileline.md](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/README_cluster_pileline.md)

### パイプライン内での使用例

[run_cluster_pipeline.sh](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/resolve_ftools_cluster_pileline.sh) 内では、
`resolve_ftools_add_cluster.py` は以下のように使用されている。

```bash
# --- Step 1: prev/next interval カラムを追加 ---
echo ">>> Running resolve_ftools_add_prevnext.sh on $input_file"
resolve_ftools_add_prevnext.sh "$input_file"

# --- Step 2: large / small 擬似イベントのクラスタリング ---
base_name="${input_file%.evt}_noBL_prevnext_cutclgti.evt"
check_file_exists "$base_name"

echo ">>> Running large cluster detection on $base_name"
resolve_ftools_add_cluster.py "$base_name" \
    --mode large \
    --col_cluster ICLUSTERL \
    --col_member IMEMBERL \
    --outname large_ \
    -d

echo ">>> Running small cluster detection on large_$base_name"
resolve_ftools_add_cluster.py "large_$base_name" \
    --mode small \
    --col_cluster ICLUSTERS \
    --col_member IMEMBERS \
    --outname small_ \
    -d

# --- Step 3: QL 診断ツール ---
final_file="small_large_${input_file%.evt}_noBL_prevnext_cutclgti.evt"
check_file_exists "$final_file"
```

### 単体実行例

例として、`clean.evt` というイベントファイルに対し、
**small cluster モード**で擬似イベント検出を行い、診断プロットを出力する場合：

```bash
resolve_ftools_add_cluster.py clean.evt \
  --mode small \
  --col_cluster ICLUSTERS \
  --col_member IMEMBERS \
  --outname small_ \
  -d
```

---

## 引数（Arguments）

本スクリプトは、以下のコマンドライン引数を受け付ける。

### 入力ファイル

* **`input_events.fits`（位置引数）**
  **入力イベント FITS ファイル**。
  これは非標準 XRISM パイプライン処理の出力であり、
  [resolve_ftools_add_prevnext.sh](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/resolve_ftools_add_prevnext.sh)
  を実行して `PREV_INTERVAL` および `NEXT_INTERVAL` カラムが追加されたものを想定している。
  詳細は以下を参照：

  * [README_add_prevnext.md](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/README_add_prevnext.md)

### モード指定

* **`-m, --mode {small|large}`**
  **クラスタリングモードの選択**。

  * **`small`（デフォルト）**
    **小規模クラスタモード**。
    クロストーク対や、小さなイベント群（一次の小さな擬似イベントと、付随する低エネルギー二次イベント）を検出することに最適化されている。

  * **`large`**
    **大規模クラスタモード**。
    複数ピクセルでほぼ同時に有意なイベントが発生した場合を検出する。
    宇宙線や粒子が検出器全体に影響した可能性を示唆するイベント群を対象とする。
    このモードでは時間ウィンドウがやや緩く設定され、
    **複数の高エネルギーイベントを含むクラスタ全体が「要注意イベント」として扱われる**。
    天体 X 線が同時に複数ピクセルを独立にトリガーする可能性は極めて低いため、
    large クラスタに属するイベントは原則として疑わしいとみなされる。

### ピクセル指定

* **`-p, --usepixels`**
  **解析対象とするピクセルの選択**。

（※ 現行バージョンでは上記が主要オプションである。）

---

## クラスタリングモードの違い：large と small

本ツールには **small** と **large** の 2 種類のクラスタリングモードがあり、
検出スケールと物理的解釈が異なる。

* **Small cluster モード**
  小規模な擬似イベント（特にクロストーク起源）を検出するためのモード。

* **Large cluster モード**
  極めて短い時間内に発生したイベントをまとめてクラスタ化する。
  2 つ以上のイベントがほぼ同時に発生した場合、それらは 1 つの「大クラスタ」として扱われ、
  **そのクラスタに属する全イベントが擬似イベント候補としてフラグ付けされる**。

通常の天体観測や中程度の明るさの天体では、
**small と large の両方を使用することが推奨される**。

バックグラウンドが非常に高い条件、較正データ、
あるいは非常に厳密に擬似イベントを除去したい場合には、
**large モードのみを用いる**ことで、同時事象をすべて検出できる。

---

## 出力（Output）

スクリプト実行後、以下が得られる。

### クラスタ情報付き FITS イベントファイル

出力 FITS ファイルには、元のイベントデータに加えて、
クラスタリング結果を示す新しいカラムが追加される。

主な追加カラム：

* **`ICLUSTER`**
  各イベントが属するクラスタの ID（整数）。
  時間順にクラスタが見つかるごとに連番で付与される
  （例：0, 0, 2, 3, 0, 5, 6, 7, 0 …）。

  * `ICLUSTER > 0`：クラスタに属するイベント
  * `ICLUSTER == 0`：非クラスタ（孤立）イベント

* **`IMEMBER`**
  そのクラスタに含まれるイベント数。

  * `IMEMBER > 0`：クラスタイベント
  * `IMEMBER == 0`：非クラスタイベント
  * `IMEMBER == 1`：そのクラスタ内で最初に検出されたイベント

元のイベントファイルに含まれるすべてのカラム
（TIME、PI/エネルギー、検出器 ID など）は **完全に保持**され、
新しいカラムは EVENTS 拡張に追加される。

この出力を用いて、たとえば

```text
ICLUSTER == 0 のイベントのみを抽出
```

といったフィルタを行うことで、
擬似イベント候補を除去したスペクトル解析が可能となる。

---

## 注意事項（Notes）

* **解析への適用について**
  本ツールはイベントを削除することは一切行わず、
  **あくまで注釈（annotation）を付けるのみ**である。
  どのイベントを除外するかは、ユーザー自身が判断する必要がある。
  非常に明るい天体の場合、特に small モードの適用には注意が必要である。

* **時間ウィンドウとパラメータ**
  クラスタリングには固定の時間一致ウィンドウを使用している。
  これは検出器の時間特性と既知の挙動に基づいて選ばれている。
  通常は調整不要だが、本ツールは **まだ初期的な実装段階**であることに留意すること。

* **制限事項（Limitations）**

  * **擬似イベントの取りこぼし**
    高カウント率時に検出器が飽和したり、オンボード CPU がイベントを落とした場合、
    親イベントが失われ、子イベントのみが残る可能性がある。
    この場合、本ツールは期待通りに動作しない可能性がある。

  * **誤検出（False positive）**
    通常の天体観測では、独立な X 線がほぼ同時に到来する確率は無視できるほど小さい。
    しかし、極端に明るい天体では、
    本物のイベントが誤ってクラスタ化される可能性がわずかに存在する。
    必要に応じて時間ウィンドウを緩める、あるいは手動で確認すること。