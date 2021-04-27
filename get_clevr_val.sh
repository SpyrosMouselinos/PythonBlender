#!/usr/bin/env bash
#
#SBATCH --job-name=blender
#SBATCH --partition=common
#SBATCH --output=./blender_test_logs.txt

function gdrive_download () {
	  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
	    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
	      rm -rf /tmp/cookies.txt
      }

echo "Downloading CLEVR validation scenes/questions"
mkdir official_val
cd official_val
gdrive_download 1hFZhdzatDaRSQyAXT656m5vAevR-AcRc CLEVR_val_scenes.json
gdrive_download 1Z2mySUxzeYimGIjyYW8p4_w8y34Bmi_O CLEVR_val_questions.json
echo "Done!"
