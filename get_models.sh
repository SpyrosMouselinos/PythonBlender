#!/usr/bin/env bash
#
#SBATCH --job-name=get_models_fair
#SBATCH --partition=common
#SBATCH --output=./get_models_fair.txt
wget https://dl.fbaipublicfiles.com/clevr/iep/models.zip
unzip models.zip
rm models.zip

