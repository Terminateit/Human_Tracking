# Human_Tracking
This is code implementation for human tracking on the video frames.

At first, clone the repository:
```bash
git clone https://github.com/Terminateit/Human_Tracking.git
```

then go to the folder yolo-coco in the repository:

```bash
cd Human_Tracking/yolo-coco
```

Run the bash-script to download weights for models:

```bash
sudo ./download_yolov3_weights.sh
```
*(May be, it the upper command will download the axel packet also (it's needed for boosting downloading from the net))*

In order to run, start yolov3_video.py script:

```bash
python3 yolov3_video.py
```

Also, you can assess the quality of the algorithm watching the video in folder "output".

