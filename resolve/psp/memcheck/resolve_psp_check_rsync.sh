#!/bin/bash

# 引数で日付を取得する
if [ -z "$1" ]; then
  echo "日付を引数として指定してください（例: 20240923）"
  exit 1
fi

DATE=$1

# rsync コマンドを実行
rsync -av --include='*/' --include='*.hk1.gz' --exclude='*' "rfxarm2:/nasA_xarm1/sot/qlff/${DATE}*/" ./
