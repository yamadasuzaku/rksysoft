#!/bin/sh

echo "getdata"
resolve_util_getdata.py

echo "decrypt"
resolve_util_decrypt.py

echo "unzip"
resolve_util_gunzip.py
