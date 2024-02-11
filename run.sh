#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /mnt/data/anaconda3/envs/szz_backend
#/usr/sbin/nginx &&
uvicorn main:app --host 0.0.0.0 --port 8000 --proxy-headers  --workers 16
