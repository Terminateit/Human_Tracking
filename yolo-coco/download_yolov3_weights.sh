# !/bin/bash
# This bash-script downloads weights for YOLO. Also, it creates all needed files
cd ..
echo "Creating needed folders and files... It may need SUDO priviledges. It's OK, don't worry! "
# make '/weights' directory if it does not exist and cd into it
mkdir -p weights && cd weights
echo "Copying weight files into yolo-coco/weights folder... It will take time ..."
# copy darknet weight files, continue '-c' if partially downloaded
sudo apt-get install axel
axel -a https://pjreddie.com/media/files/yolov3.weights
axel -a https://pjreddie.com/media/files/yolov3-tiny.weights
axel -a https://pjreddie.com/media/files/yolov3-spp.weights
