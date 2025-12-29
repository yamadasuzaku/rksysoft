#!/bin/bash

# resolve_env.sh: HEASOFT 環境の最小ブートストラップ（macOSだけ対策）
case "$(uname -s 2>/dev/null)" in
  Darwin)
    # macOS: system shell 起動時にDYLD_*が落ちることがあるので復元
    if [ -n "${HEADAS:-}" ] && [ -z "${DYLD_LIBRARY_PATH:-}" ]; then
      export DYLD_LIBRARY_PATH="$HEADAS/lib"
    fi
    ;;
  *)
    # Linux等は基本何もしない
    ;;
esac
