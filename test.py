### THIS FILE CAN BE RUN ANYWHERE IN A TERMINAL WRITING 'python deeplab_demo_webcam_v2.py' AS LONG
### AS THE HELPER FILE get_dataset_colormap.py IS IN THE SAME DIRECTORY AS deeplab_demo_webcam_v2.py
## adapted from antonio verdones blogpost https://averdones.github.io/real-time-semantic-image-segmentation-with-deeplab-in-tensorflow/
 
## Imports

import collections
import os
import io
import sys
import tarfile
import tempfile
import urllib
import time
import pyfakewebcam
import sys
from threading import Thread, Event
from time import sleep
import numpy as np
from PIL import Image
import cv2
# import skvideo.io

import tensorflow as tf

# Needed to show segmentation colormap labels
import get_dataset_colormap

import tensorflow as tf

## Select and download models
_MODEL_URLS = {
    'deeplabv3_mnv2_dm05_pascal_trainaug': 'http://download.tensorflow.org/models/deeplabv3_mnv2_dm05_pascal_trainaug_2018_10_01.tar.gz',
    'deeplabv3_mnv2_dm05_pascal_trainval': 'http://download.tensorflow.org/models/deeplabv3_mnv2_dm05_pascal_trainval_2018_10_01.tar.gz',
}

_TARBALL_NAME = 'deeplabv3_mnv2_dm05_pascal_trainaug_2018_10_01.tar.gz'
model_url = _MODEL_URLS['deeplabv3_mnv2_dm05_pascal_trainaug']


download_path = "deeplabv3_mnv2_dm05_pascal_trainaug_2018_10_01.tar.gz"
'''
print('downloading model to %s, this might take a while...' % download_path)
urllib.request.urlretrieve(model_url, download_path)
print('download completed!')
'''
## Load model in TensorFlow
_FROZEN_GRAPH_NAME = 'frozen_inference_graph'

class DeepLabModel(object):
    """Class to load deeplab model and run inference."""
    
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if _FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()
        
        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():      
            tf.import_graph_def(graph_def, name='')
        
        self.sess = tf.compat.v1.Session(graph=self.graph)
            
    def run(self, image):
        """Runs inference on a single image.
        
        Args:
            image: A PIL.Image object, raw input image.
            
        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map

    def run_once(self, framearr):
        frame=framearr[0]
        # From cv2 to PIL
        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        
        # Run model
        resized_im, seg_map = model.run(pil_im)
        # Adjust color of mask
        seg_image = get_dataset_colormap.label_to_color_image(
            seg_map, get_dataset_colormap.get_pascal_name()).astype(np.uint8)
        
        # Convert PIL image back to cv2 and resize
        r = seg_image.shape[1] / frame.shape[1]
        
        dim = (frame.shape[0], frame.shape[1])[::-1]
        
        #print(dim," ",frame.shape[0],"x",frame.shape[1])
        seg_image = cv2.resize(seg_image, dim, interpolation = cv2.INTER_AREA)
        # Stack horizontally color frame and mask
        (r,g,b)=cv2.split(seg_image)
        ret2, thresh2 = cv2.threshold(r, 191, 255, cv2.THRESH_BINARY)
        ret3, thresh3 = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)
        ret4, thresh4 = cv2.threshold(b, 127, 255, cv2.THRESH_BINARY)
        rgb = cv2.bitwise_and(thresh2, thresh3, thresh4)
        rgb = cv2.merge((rgb,rgb,rgb))
        return rgb
    def run_threaded(self, framearr,rgbarr,fpsarr):
        while True:
            frame=framearr[0]
            start = time.time()#time processing of the frame
            # From cv2 to PIL
            pil_im = Image.fromarray(frame)
            
            # Run model
            resized_im, seg_map = model.run(pil_im)
            # Adjust color of mask
            seg_image = get_dataset_colormap.label_to_color_image(
                seg_map, get_dataset_colormap.get_pascal_name()).astype(np.uint8)
            
            # Convert PIL image back to cv2 and resize
            r = seg_image.shape[1] / frame.shape[1]
            
            dim = (frame.shape[0], frame.shape[1])[::-1]
            
            #print(dim," ",frame.shape[0],"x",frame.shape[1])
            seg_image = cv2.resize(seg_image, dim, interpolation = cv2.INTER_AREA)
            # apply threshold mask with human label color to the frame
            (r,g,b)=cv2.split(seg_image)
            ret2, thresh2 = cv2.threshold(r, 191, 255, cv2.THRESH_BINARY)
            ret3, thresh3 = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)
            ret4, thresh4 = cv2.threshold(b, 127, 255, cv2.THRESH_BINARY)
            rgb = cv2.bitwise_and(thresh2, thresh3, thresh4)
            rgb = cv2.merge((rgb,rgb,rgb))
            rgbarr[0]=rgb
            #alpha = cv2.GaussianBlur(rgb, (7,7),0)
            end = time.time()#finish timer
            fpsarr[0]=(1/(end - start))

model = DeepLabModel(download_path)

camera=pyfakewebcam.FakeWebcam('/dev/video2',1920,1080)

## Webcam demo
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
cap.set(cv2.CAP_PROP_FPS,30)
# Next line may need adjusting depending on webcam resolution
ret, frame = cap.read()
framearr=[frame]
rgb=model.run_once(framearr)
rgbarr=[rgb]
fpsarr=[0]
t=Thread(target=model.run_threaded, args=(framearr, rgbarr,fpsarr))
t.start()
while True:
    start = time.time()#time processing of the frame
    ret, frame = cap.read()
    framearr[0]= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    color_and_mask = cv2.bitwise_and(framearr[0],rgbarr[0])
    camera.schedule_frame(color_and_mask)#output to camera
    end = time.time()#finish timer
    sys.stdout.write("\rmodel:%f fps  camera:%i fps." % (fpsarr[0], int(1/(end - start))) )
