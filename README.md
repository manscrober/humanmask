# humanmask
Uses webcam and DeepLabv3 with mobilenetv2 backbone to remove background and only show humans.

Designed to be used with OBS Studio, its a zero-opacity filter on black pixels and and v4l2-sink extension to create a virtual projector, allowing the user to project their screen behind themselves. Works better than virtual backgrounds too.


requirements:

download tensorflow model from http://download.tensorflow.org/models/deeplabv3_mnv2_dm05_pascal_trainval_2018_10_01.tar.gz
and place in the same folder.
install  python, v4l2loopback, tensorflow-gpu, cudnn, opencv2, PIL, numpy, and pyfakewebcam


usage:

from a terminal in the directory:
sudo modprobe v4l2loopback devices=2
python test.py
