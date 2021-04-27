#!/usr/bin/env bash
#
#SBATCH --job-name=blender
#SBATCH --partition=common
#SBATCH --output=./blender_test_logs.txt

echo "Downloading Blender 2.79b for x86 Linux"
#wget https://download.blender.org/release/Blender2.79/blender-2.79b-linux-glibc219-x86_64.tar.bz2

echo "Extracting..."
#tar -xf blender-2.79b-linux-glibc219-x86_64.tar.bz2

echo "Deleting the tar file..."
#rm -rf blender-2.79b-linux-glibc219-x86_64.tar.bz2

echo "Rename and Move"
mv  blender-2.79b-linux-glibc219-x86_64 generation/blender2.79


echo "######################################"
echo "Enabling Execution on Blender..."
chmod 555 ./generation/blender2.79/blender

echo "Adding the local .pth file to Blender"
cp  clevr_local.pth ./generation/blender2.79/2.79/python/lib/python3.5/site-packages/clevr.pth

