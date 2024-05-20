#!/bin/bash

# 引数のチェック
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <source_directory> [<file_extension>]"
    echo "ex) ./resolve_util_rsync.sh /home/resolve/xrdata/NGC1365/300075010/resolve/event_cl"
  exit 1
fi

# ソースディレクトリ
SOURCE_DIR=$1

# ファイル拡張子（デフォルトは png）
FILE_EXTENSION=${2:-png}

echo $FILE_EXTENSION $SOURCE_DIR

# rsyncコマンドの実行
rsync -av --include="*.$FILE_EXTENSION" --include='*/' --exclude='*' "resolverk:$SOURCE_DIR" .

echo "Files with extension .$FILE_EXTENSION from $SOURCE_DIR have been synchronized."
