#!/usr/bin/env bash
#
#SBATCH --job-name=blender
#SBATCH --partition=common
#SBATCH --output=./blender_test_logs.txt

mkdir official/
wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0_no_images.zip -O ./official/CLEVR_v1.0.zip
unzip ./official/CLEVR_v1.0.zip -d ./official/
rm -rf ./official/CLEVR_v1.0.zip

