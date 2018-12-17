import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

class ObjectDetection:

	# What model to download.
	MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
	MODEL_FILE = MODEL_NAME + '.tar.gz'
	DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

	# Path to frozen detection graph. This is the actual model that is used for the object detection.
	PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

	# List of the strings that is used to add correct label for each box.
	PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

	def __init__(self):
		#self.downloadModel()
		self.loadTensorflowModel()

	def downloadModel(self):
		opener = urllib.request.URLopener()
		opener.retrieve(self.DOWNLOAD_BASE + self.MODEL_FILE, self.MODEL_FILE)
		tar_file = tarfile.open(self.MODEL_FILE)
		for file in tar_file.getmembers():
			file_name = os.path.basename(file.name)
			if 'frozen_inference_graph.pb' in file_name:
				tar_file.extract(file, os.getcwd())

	def loadTensorflowModel(self):
		self.detection_graph = tf.Graph()
		with self.detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(self.PATH_TO_FROZEN_GRAPH, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')
	
	def boxsize_conversion(self, image, boxes):
		h, w = image.shape[0], image.shape[1]
		newboxes = np.zeros_like(boxes)
		newboxes[:, 0] = boxes[:, 0] * h
		newboxes[:, 1] = boxes[:, 1] * w
		newboxes[:, 2] = boxes[:, 2] * h
		newboxes[:, 3] = boxes[:, 3] * w

		return newboxes

	def runDetection(self, image):
		_image = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
		#return image
		tensor_dict = []
		for key in ['detection_boxes', 'detection_scores','detection_classes']:
			tensor_name = key + ':0'
			tensor_dict.append(self.detection_graph.get_tensor_by_name(tensor_name))
		image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
		with tf.Session(graph=self.detection_graph) as sess:
			(boxes, scores, classes) = sess.run(tensor_dict, feed_dict={image_tensor: _image})
			boxes = np.squeeze(boxes)
			scores = np.squeeze(scores)
			classes = np.squeeze(classes)
        # filter out wanted boxes based on classes c and threshold
		selected = []
		c = [1, 3, 10, 13] # cars
		threshold = 0.4
		for i in range(len(classes)):
			if classes[i] in c and scores[i] >= threshold:
				selected.append(i)
		boxes = boxes[selected, ...]
		scores = scores[selected, ...]
		classes = classes[selected, ...]

  		# draw the box
  		boxes = self.boxsize_conversion(image, boxes)
  		for i in range(len(boxes)):
  			bottom, left, top, right = boxes[i, ...]
  			cv2.rectangle(image,(left,top),(right,bottom),(0,0,255), 3)

		return image


