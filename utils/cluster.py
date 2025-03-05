import numpy as np
import cv2
from sklearn.decomposition import PCA
import hdbscan
import pickle
import random
from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN
import tensorflow as tf
import tensorrt as trt
import os
from utils import common
import torch
import torch.nn as nn
import time
from pathlib import Path

TRT_LOGGER = trt.Logger()
random.seed(999)

def create_model(layers):
	dims = 128
	# with mirrored_strategy.scope():
	if layers == 3:
		model_2 = tf.keras.Sequential([
			tf.keras.layers.InputLayer(input_shape=dims),
			# tf.keras.layers.LayerNormalization(),
			tf.keras.layers.Dense(units=128, activation="relu"),
			tf.keras.layers.Dense(units=128, activation="relu"),
			tf.keras.layers.Dense(units=64),
		])
	if layers == 4:
		model_2 = tf.keras.Sequential([
			tf.keras.layers.InputLayer(input_shape=dims),
			# tf.keras.layers.LayerNormalization(),
			tf.keras.layers.Dense(units=128, activation="relu"),
			tf.keras.layers.Dense(units=128, activation="relu"),
			tf.keras.layers.Dense(units=128, activation="relu"),
			tf.keras.layers.Dense(units=64),
		])

	if layers == 5:
		model_2 = tf.keras.Sequential([
			tf.keras.layers.InputLayer(input_shape=dims),
			# tf.keras.layers.LayerNormalization(),
			tf.keras.layers.Dense(units=128, activation="relu"),
			tf.keras.layers.Dense(units=128, activation="relu"),
			tf.keras.layers.Dense(units=128, activation="relu"),
			tf.keras.layers.Dense(units=128, activation="relu"),
			tf.keras.layers.Dense(units=64),
		])

	model_2.summary()

	return model_2

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

class ClusterModel:

	def __init__(self, clustering_algorithm, compression_algorithm, n_components, path_cluster, torch_use, end):
		self.dla = False
		self.clustering_algorithm = clustering_algorithm
		self.compression_algorithm = compression_algorithm
		self.cluster = None
		self.reducer = None
		self.distances = None
		self.n_components = n_components
		self.end = end
		self.path_cluster = path_cluster
		self.torch_use = torch_use

	def load_model(self, path):
		class model_torch(torch.nn.Module):

			def __init__(self):
				super(model_torch, self).__init__()

				self.dropout_1 = nn.Dropout(0.2)
				self.linear1 = torch.nn.Linear(128, 128)
				self.activation_1 = torch.nn.ReLU()
				self.dropout_2 = nn.Dropout(0.2)
				self.linear2 = torch.nn.Linear(128, 128)
				self.activation_2 = torch.nn.ReLU()
				self.dropout_3 = nn.Dropout(0.2)
				self.linear3 = torch.nn.Linear(128, 128)
				self.activation_3 = torch.nn.ReLU()
				self.dropout_4 = nn.Dropout(0.2)
				self.linear4 = torch.nn.Linear(128, 128)
				self.activation_4 = torch.nn.ReLU()
				self.linear5 = torch.nn.Linear(128, 128)

			def forward(self, x):
				x = self.dropout_1(self.activation_1(self.linear1(x)))
				x = self.dropout_2(self.activation_2(self.linear2(x)))
				x = self.dropout_3(self.activation_3(self.linear3(x)))
				x = self.dropout_4(self.activation_4(self.linear4(x)))
				x = self.linear5(x)
				return x

		file_clusterer = open(path + 'clusterer.obj', 'rb')
		self.cluster = pickle.load(file_clusterer)
		if self.torch_use:

			model_torch = model_torch()
			base_dir = Path(__file__).resolve().parent.parent
			model_path = str(base_dir / "pretrained_models/model_cluster.pth")

			# Load model
			model_torch.load_state_dict(torch.load(model_path))
			model_torch.eval()

			self.model_torch = model_torch
			self.engine = None
		else:
			try:
				file_reducer = open(path + 'reducer.obj', 'rb')
				self.reducer = pickle.load(file_reducer)
				self.engine = None
			except:
				if self.end == False:
					self.engine = get_engine(self.path_cluster, self.dla)
					self.context = self.engine.create_execution_context()
				if self.end == True:
					self.engine = 0
					self.context = 0


	def build(self, data, bbs):
		class model_torch(torch.nn.Module):

			def __init__(self):
				super(model_torch, self).__init__()

				self.dropout_1 = nn.Dropout(0.2)
				self.linear1 = torch.nn.Linear(128, 128)
				self.activation_1 = torch.nn.ReLU()
				self.dropout_2 = nn.Dropout(0.2)
				self.linear2 = torch.nn.Linear(128, 128)
				self.activation_2 = torch.nn.ReLU()
				self.dropout_3 = nn.Dropout(0.2)
				self.linear3 = torch.nn.Linear(128, 128)
				self.activation_3 = torch.nn.ReLU()
				self.dropout_4 = nn.Dropout(0.2)
				self.linear4 = torch.nn.Linear(128, 128)
				self.activation_4 = torch.nn.ReLU()
				self.linear5 = torch.nn.Linear(128, 128)

			def forward(self, x):
				x = self.dropout_1(self.activation_1(self.linear1(x)))
				x = self.dropout_2(self.activation_2(self.linear2(x)))
				x = self.dropout_3(self.activation_3(self.linear3(x)))
				x = self.dropout_4(self.activation_4(self.linear4(x)))
				x = self.linear5(x)
				return x
		# L2 norm.
		if data.ndim == 1:
			norms = np.linalg.norm(data)
			activations = data / norms
		else:
			norms = np.linalg.norm(data, axis=1)
			activations = data / norms[:, np.newaxis]

		# Data compression
		if self.compression_algorithm == 'umap':
			reducer = umap.UMAP(n_components=self.n_components) #TODO #n_neighbors=5, n_components=2048, random_state=42
			embedding_ = reducer.fit_transform(activations)
		elif self.compression_algorithm == 'pca':
			reducer = PCA(n_components=self.n_components) #'mle'
			embedding_ = reducer.fit_transform(activations)
		elif self.compression_algorithm == 'none':
			reducer = None
			embedding_ = activations
		elif self.compression_algorithm =='model_umap':
			reducer = None
			model_torch = model_torch()
			base_dir = Path(__file__).resolve().parent.parent
			model_path = str(base_dir / "pretrained_models/model_cluster.pth")

			# Load model
			model_torch.load_state_dict(torch.load(model_path))
			model_torch.eval()

			embedding_ = model_torch(torch.Tensor(activations))
			embedding_ = embedding_.detach().numpy()


		if isinstance(embedding_, dict):
			embedding_ = embedding_['umap']

		# Include aspect ratio in the embeddings
		ar = (bbs[:, 3] - bbs[:, 1]) / np.float32(bbs[:, 2] - bbs[:, 0])
		embedding = np.zeros((embedding_.shape[0], embedding_.shape[1] + 1))
		embedding[:, :-1] = embedding_
		embedding[:, embedding_.shape[1]] = ar.reshape(-1, 1)[:, 0]

		# Clustering
		if self.clustering_algorithm == 'hdbscan':
			clusterer = hdbscan.HDBSCAN(algorithm='boruvka_balltree', core_dist_n_jobs=8,prediction_data=True, min_cluster_size=30) #min_cluster_size=5, min_samples=5, core_dist_n_jobs=-2, algorithm='boruvka_kdtree', cluster_selection_method='eom', prediction_data=True
			clusterer.fit(embedding)
		elif self.clustering_algorithm == 'optics':
			clusterer = OPTICS(min_cluster_size=50, algorithm='auto', n_jobs=8)
			clusterer.fit(embedding)
		elif self.clustering_algorithm == 'dbscan':
			clusterer = DBSCAN(eps=2, algorithm='auto', n_jobs=8, min_samples=30)
			clusterer.fit(embedding)

		self.cluster = clusterer
		self.reducer = reducer

		return clusterer, reducer

	def predict(self, data, bbs, energy_measurer_GPU, energy_measurer_CPU):

		if energy_measurer_GPU is not None:
			energy_measurer_GPU.start_measuring()

		tcluster_engine1 = time.time()

		if data.ndim == 1:
			data = np.vstack([data, data])
		norms = np.linalg.norm(data, axis=1)
		activations = data / norms[:, np.newaxis]

		ar = (bbs[:, 3] - bbs[:, 1]) / np.float32(bbs[:, 2] - bbs[:, 0])

		# Data compression
		if self.reducer is not None:
			embedding_ = self.reducer.transform(activations)
		else:
			embedding_ = activations

		if self.engine is not None:
			inputs_trt, outputs_trt, bindings_trt, stream_trt = common.allocate_buffers_descriptor(self.engine, activations.shape)
			inputs_trt[0].host = activations
			self.context.set_binding_shape(0, (activations.shape[0], 128))
			feats = common.do_inference_v2(self.context, bindings=bindings_trt, inputs=inputs_trt, outputs=outputs_trt, stream=stream_trt)
			embedding_ = feats[0].reshape((activations.shape[0], -1))

		else:
			embedding_ = self.model_torch(torch.Tensor(activations))
			embedding_ = embedding_.detach().numpy()

		if isinstance(embedding_, dict):
			embedding_ = embedding_['umap']

		embedding = np.zeros((embedding_.shape[0], embedding_.shape[1] + 1))
		embedding[:, :-1] = embedding_
		embedding[:, embedding_.shape[1]] = ar.reshape(-1, 1)[:, 0]

		tcluster_engine2 = time.time()

		if energy_measurer_GPU is not None:
			energy_measurer_GPU.stop_measuring()

		tcluster_engine = tcluster_engine2-tcluster_engine1

		if energy_measurer_CPU is not None:
			energy_measurer_CPU.start_measuring()

		tcluster_hdbscan1 = time.time()
		# Clustering
		if self.clustering_algorithm == 'hdbscan':
			probs = hdbscan.membership_vector(self.cluster, embedding)
			bbs = bbs[~np.isnan(probs).any(axis=1), :]
			probs = probs[~np.isnan(probs).any(axis=1)]
			test_labels = np.nanargmax(probs, axis=1)
			strengths = np.nanmax(probs, axis=1)

			tcluster_hdbscan2 = time.time()

		elif self.clustering_algorithm == 'kmeans':
			test_labels = self.cluster.predict(embedding)
			strengths_ = self.cluster.transform(embedding)
			strengths = np.zeros(test_labels.shape[0])
			for i in range(test_labels.shape[0]):
				strengths[i] = strengths_[i, test_labels[i]]


		if energy_measurer_CPU is not None:
			energy_measurer_CPU.stop_measuring()
			energy_hdbscan = energy_measurer_CPU.total_energy
		else:
			energy_hdbscan = 0

		tcluster_hdbscan = tcluster_hdbscan2-tcluster_hdbscan1

		if energy_measurer_GPU is not None:
			energy_model_umap = energy_measurer_GPU.total_energy
		else:
			energy_model_umap = 0

		return test_labels, strengths, bbs, tcluster_engine, energy_model_umap, tcluster_hdbscan, energy_hdbscan

	def save_clusters(self, imge_paths, boxes, labels, scores, outpath_clusters):
		ulabs = np.unique(labels)
		ulabs = ulabs[ulabs >= 0]
		print("Detections total: {}".format(len(imge_paths)))
		for i in range(ulabs.shape[0]):
			positions = np.where(ulabs[i] == labels)[0]
			positions = positions[np.argsort(scores[positions])][::-1]
			print("Detections class {} : {}".format(i, positions.shape[0]))
			if positions.shape[0]<100:
				data_ = self.load_images(positions, imge_paths, boxes, positions.shape[0])
			else:
				data_ = self.load_images(positions, imge_paths, boxes, 100)

			for j in range(len(data_)):
				imgg = data_[j]
				imgg_ = imgg[:, :, ::-1].copy()
				cv2.imwrite(outpath_clusters + str(ulabs[i]) + '_' + str(j) + '.jpg', cv2.cvtColor(imgg_, cv2.COLOR_BGR2RGB))

	def load_images(self, positions, paths, boxes, n):
		positions_ = positions[0:n]
		images = []
		for i in positions_:
			image_np = cv2.imread(paths[i])

			w = 720
			h = 405
			image_np = cv2.resize(image_np, (w, h))

			x = max(0, np.int32(boxes[i, 0])) #* image_np.shape[1]
			xe = min(image_np.shape[1], np.int32(boxes[i, 2])) #* image_np.shape[1]
			y = max(0, np.int32(boxes[i, 1])) #* image_np.shape[0]
			ye = min(image_np.shape[0], np.int32(boxes[i, 3])) # * image_np.shape[0]
			im = image_np[y:ye, x:xe, :]
			images.append(im)

		return images
