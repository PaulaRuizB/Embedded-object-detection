#import tensorflow as tf
import time
import sys
sys.path.append('../')
from pathlib import Path
import numpy as np
import torch
import os
import tensorrt as trt
from utils import common
TRT_LOGGER = trt.Logger()

def get_engine(engine_file_path="", dla=False):
	if os.path.exists(engine_file_path):
		# If a serialized engine exists, use it instead of building an engine.
		# print("Reading engine from file {}".format(engine_file_path))
		with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
			if dla:
				runtime.DLA_core=0
			return runtime.deserialize_cuda_engine(f.read())
	else:
		print("ERROR, NO TRT AVAILABLE")

class DetectionModel:
	def __init__(self, version, kind, path_trt, dla,  int8, size_rpn, end):
		self.version = version
		self.kind = kind
		self.interpreter = None
		self.input_details = None
		self.output_details = None
		self.sess = None
		self.image_tensor = None
		self.tf_scores = None
		self.tf_boxes = None
		self.tf_classes = None
		self.dla = dla
		self.end = end
		self.int8 = int8
		self.size_rpn = size_rpn

		if kind == 'xavier':
			self.init_xavier()
		elif kind == 'tflite':
			self.init_tflite()
		elif kind == 'trt':
			if self.end == False:
				self.engine = get_engine(path_trt, self.dla)
				self.context = self.engine.create_execution_context()

				self.inputs_trt, self.outputs_trt, self.bindings_trt, self.stream_trt = common.allocate_buffers(self.engine, self.size_rpn)

			if self.end ==True:
				self.engine = 0
				self.context = 0
		else:
			self.init_normal()

	def init_normal(self):
		if self.version == 'faster':
			model_path = 'detection_models/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28/frozen_inference_graph.pb'
		elif self.version == 'ssd_mobilenet_v1_fpn':
			model_path = 'detection_models/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb'
		elif self.version == 'ssd_mobilenet_v2_oid':
			model_path = 'detection_models/ssd_mobilenet_v2_oid_v4_2018_12_12/frozen_inference_graph.pb'
		elif self.version == 'ssd_resnet50':
			model_path = 'detection_models/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb'
		elif self.version == 'ssdlite':
			model_path = 'detection_models/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb'
		elif self.version == 'yolov4_resnet50':
			model_path = str(Path.cwd()) + "/pretrained_models/best.pt"
		else:
			model_path = 'detection_models/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb'

		# Load model
		self.model = torch.load(model_path)['model'].float().fuse().eval().cuda(0)

	def init_xavier(self):

		if self.int8:
			model_path = str(Path.cwd()) + "/pretrained_models/last_302.pt"
			with torch.no_grad():
				self.model = torch.load(model_path)['model'].fuse().eval()
		else:
			base_dir = Path(__file__).resolve().parent.parent
			model_path = str(base_dir / "pretrained_models/best.pt")
			# Load model
			self.model = torch.load(model_path)['model'].float().fuse().eval()

	def init_tflite(self):
		if self.version == 'faster':
			model_path = 'detection_models/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28/frozen_inference_graph.pb'
		elif self.version == 'ssd_mobilenet_v1_fpn':
			model_path = 'detection_models/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb'
		elif self.version == 'ssd_mobilenet_v2_coco':
			model_path = 'detection_models_fp16/ssd_mobilenet2/tflite_model.tflite'
		elif self.version == 'ssd_resnet50':
			model_path = 'detection_models/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb'
		elif self.version == 'ssdlite':
			model_path = 'detection_models_fp16/ssdlite/trt_graph.pb'
		else:
			model_path = 'detection_models/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb'

		# Load TFLite model and allocate tensors.
		self.interpreter = tf.lite.Interpreter(model_path=model_path)
		self.interpreter.allocate_tensors()

		# Get input and output tensors.
		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()


	def compute_detections_normal(self, image):

		pred = self.model(image.cuda(0))[0]

		return pred

	def compute_detections_trt(self, image):

		# Do inference
		# Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
		self.inputs_trt[0].host = image
		feats = common.do_inference_v2(self.context, bindings=self.bindings_trt, inputs=self.inputs_trt,
											   outputs=self.outputs_trt, stream=self.stream_trt)

		return feats


	def compute_detections_tflite(self, image):
		self.interpreter.set_tensor(self.input_details[0]['index'], np.expand_dims(np.float32(image), 0))
		self.interpreter.invoke()
		scores = np.squeeze(self.interpreter.get_tensor(self.output_details[2]['index']))
		boxes = np.squeeze(self.interpreter.get_tensor(self.output_details[0]['index']))
		classes = np.squeeze(self.interpreter.get_tensor(self.output_details[1]['index']))

		return boxes, classes, scores

	def compute_detections(self, image):
		if self.kind == 'xavier':
			pred = self.compute_detections_normal(image)
		elif self.kind == 'tflite':
			boxes_, classes_, scores_ = self.compute_detections_tflite(image)
		elif self.kind == 'trt':
			pred = self.compute_detections_trt(image)
		else:
			pred = self.compute_detections_normal(image)

		return pred
