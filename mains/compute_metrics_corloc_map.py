import glob
import tensorflow as tf
from lxml import etree
import os
import numpy as np
import sys
import cv2

from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

#PARAMETERS
PATH_IMAGES = '' #path images dataset
GROUND_TRUTH_PATH = '' #path annotations dataset
LABEL_MAP_PATH = '' #label map path dataset

PREDICTION_PATH = CLUSTERING_OUTPUT = '' #npz clustering output
VERSION = 'complete' # complete version or only one part OF or RPN

PATH_MODELS_TF = '' #please install models package from tensorflow models
sys.path.append(PATH_MODELS_TF)
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

def computeIoU(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou

def parse_xml(path, label_map_dict): #, img, contador
	with tf.io.gfile.GFile(path, 'r') as fid:
		xml_str = fid.read()
	xml = etree.fromstring(xml_str)
	data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

	width = int(data['size']['width'])
	height = int(data['size']['height'])

	xmin = []
	ymin = []
	xmax = []
	ymax = []
	classes = []
	classes_text = []
	if 'object' in data:
		for obj in data['object']:
			xmin.append(max(0, float(obj['bndbox']['xmin'])))  # / width
			ymin.append(max(0, float(obj['bndbox']['ymin'])))  # / height
			xmax.append(min(float(obj['bndbox']['xmax']), width))  # / width
			ymax.append(min(float(obj['bndbox']['ymax']), height))  # / height
			classes_text.append(obj['name'].encode('utf8'))
			classes.append(label_map_dict[obj['name']])

	return xmin, ymin, xmax, ymax, classes, classes_text

def computer_corloc(label_map_dict):
	counter_of = 0
	counter_det = 0
	counter_per_class_det = np.zeros(11)
	counter_per_class_of = np.zeros(11)
	total_boxes_of = 0
	total_boxes_det = 0
	total_boxes_per_class = np.zeros(11)

	contador = 0

	for data_path in sorted(glob.glob(PREDICTION_PATH + '*.npz')):
		data = np.load(data_path)
		boxes = data['boxes_corloc']
		if VERSION == 'complete':
			scores = data['scores']

		filename = str(int(os.path.splitext(os.path.split(data_path)[1])[0])).zfill(6)

		img_original = cv2.imread(PATH_IMAGES + filename + '.jpg')
		detection = img_original.copy()
		contador = contador +1
		detection_et = img_original.copy()

		xmin, ymin, xmax, ymax, classes, classes_text = parse_xml(GROUND_TRUTH_PATH + filename + '.xml', label_map_dict) #, detection_et, contador

		# Compute CorLoc.
		selected = np.zeros(len(xmin))
		for i in range(boxes.shape[0]):

			boxes[i][0] = boxes[i][0] * 1920 / 720
			boxes[i][1] = boxes[i][1] * 1080 / 405
			boxes[i][2] = boxes[i][2] * 1920 / 720
			boxes[i][3] = boxes[i][3] * 1080 / 405

			for j in range(len(xmin)):

				if selected[j] == 0:

					box = [xmin[j], ymin[j], xmax[j], ymax[j]]
					iou = computeIoU(boxes[i, :], box)

					if VERSION == 'complete':
						if scores[i] >= 0:
							total_boxes_det = total_boxes_det + 1

							if iou > 0.5:
								counter_det = counter_det + 1
								counter_per_class_det[classes[j]-1] = counter_per_class_det[classes[j]-1] + 1
								selected[j] = 1
								break
						else:
							total_boxes_of = total_boxes_of + 1
							if iou > 0.1:
								counter_of = counter_of + 1
								counter_per_class_of[classes[j]-1] = counter_per_class_of[classes[j]-1] + 1
								selected[j] = 1
								break
					else:
						if np.int32(boxes[i][4]) == 0 or np.int32(boxes[i][4]) == 2:
							total_boxes_of = total_boxes_of + 1
							if iou > 0.1:
								counter_of = counter_of + 1
								counter_per_class_of[classes[j] - 1] = counter_per_class_of[classes[j] - 1] + 1
								selected[j] = 1
								break

		for j in range(len(xmin)):
			total_boxes_per_class[classes[j]-1] = total_boxes_per_class[classes[j]-1] + 1

	corloc_det = counter_det / np.float32(total_boxes_det)
	corloc_per_class_det = counter_per_class_det / np.float32(total_boxes_per_class)
	corloc_of = counter_of / np.float32(total_boxes_of)
	corloc_per_class_of = counter_per_class_of / np.float32(total_boxes_per_class)

	return corloc_det, corloc_per_class_det, corloc_of, corloc_per_class_of


def compute_ap(rec, prec):
	mrec = np.zeros(rec.shape[0] + 2)
	mrec[1:-1] = rec
	mrec[rec.shape[0] + 1] = 1
	mpre = np.zeros(prec.shape[0] + 2)
	mpre[1:-1] = prec
	mpre[prec.shape[0] + 1] = 0

	for i in range(mpre.shape[0]-2, 0, -1):
		mpre[i] = max(mpre[i], mpre[i + 1])

	pos = []
	for i in range(1, mrec.shape[0]):
		if mrec[i] is not mrec[i-1]:
			pos.append(i+1)
	pos = np.asarray(pos)
	if pos[pos.shape[0]-1] == mrec.shape[0]:
		pos = pos[0:-1]
	return np.sum((mrec[pos] - mrec[pos - 1]) * mpre[pos])

def compute_map(label_map_dict, labels_map):
	#labels_map = np.loadtxt(MAP_FILE)

	counter = 0
	total_boxes = 0
	gt = []
	predictions = []
	labels = []
	boxes = []
	scores = []
	gt_xmin = []
	gt_ymin = []
	gt_xmax = []
	gt_ymax = []
	gt_classes = []
	for data_path in sorted(glob.glob(CLUSTERING_OUTPUT + '*.npz')):
		data = np.load(data_path)
		labels_ = data['labels']
		boxes_ = data['boxes']
		scores_ = data['scores']

		boxes_[:,0] = boxes_[:,0] * 1920 / 720
		boxes_[:,1] = boxes_[:,1] * 1080 / 405
		boxes_[:,2] = boxes_[:,2] * 1920 / 720
		boxes_[:,3] = boxes_[:,3] * 1080 / 405

		if boxes_.shape[0] is not labels_.shape[0]:
			print(data_path)
		else:
			labels_ = labels_map[labels_]
			pos = np.where(labels_ > 0)[0]
			labels_ = labels_[pos]
			boxes_ = boxes_[pos, :]
			scores_ = scores_[pos]

			labels.append(labels_)
			boxes.append(boxes_)
			scores.append(scores_)

			# Compute new metric
			# Load ground-truth results.
			filename = str(int(os.path.splitext(os.path.split(data_path)[1])[0])).zfill(6)
			xmin, ymin, xmax, ymax, classes, classes_text = parse_xml(GROUND_TRUTH_PATH + filename + '.xml', label_map_dict)

			gt_xmin.append(np.asarray(xmin))
			gt_ymin.append(np.asarray(ymin))
			gt_xmax.append(np.asarray(xmax))
			gt_ymax.append(np.asarray(ymax))
			gt_classes.append(np.asarray(classes))

	ulabs = np.unique(np.hstack(labels))
	ap = np.zeros(ulabs.shape[0])
	for ulab_ix in range(ulabs.shape[0]):
		predictions_ = []
		tp = np.zeros(np.hstack(labels).shape[0])
		fp = np.zeros(np.hstack(labels).shape[0])
		bb_counter = 0
		gt_counter = 0
		for im_ix in range(len(labels)):
			poss = np.where(labels[im_ix] == ulabs[ulab_ix])[0]
			if poss.shape[0] > 0:
				poss = poss[np.argsort(scores[im_ix][poss])][::-1]
				boxes_ = boxes[im_ix][poss, :]
				scores_ = scores[im_ix][poss]
				poss = np.where(gt_classes[im_ix] == ulabs[ulab_ix])[0]
				if poss.shape[0] > 0:
					xmin_ = gt_xmin[im_ix][poss]
					ymin_ = gt_ymin[im_ix][poss]
					xmax_ = gt_xmax[im_ix][poss]
					ymax_ = gt_ymax[im_ix][poss]
					gt_counter = gt_counter + xmin_.shape[0]
					# Compute CorLoc.
					selected = np.zeros(len(xmin_))
					for i in range(boxes_.shape[0]):
						omax = 0
						idxx = -1
						for j in range(len(xmin_)):
							if selected[j] == 0:
								box = [xmin_[j], ymin_[j], xmax_[j], ymax_[j]]
								iou = computeIoU(boxes_[i, :], box)

								if scores_[i] >= 0:
									if iou > 0.5 and iou > omax:
										omax = iou
										idxx = j
								else:
									if iou > 0.1 and iou > omax:
										omax = iou
										idxx = j
						if idxx >= 0:
							counter = counter + 1
							selected[idxx] = 1
							tp[bb_counter + i] = 1
						else:
							fp[bb_counter + i] = 1
				else:
					fp[bb_counter:bb_counter+boxes_.shape[0]] = 1

				bb_counter = bb_counter + boxes_.shape[0]

		tp = np.cumsum(tp)
		fp = np.cumsum(fp)
		rec = tp / gt_counter
		prec = tp / (fp + tp)
		ap[ulab_ix] = compute_ap(rec, prec)

	return np.mean(ap), ap

def compute_confusion_matrix(label_map_dict, labels_map):
	#labels_map = np.loadtxt(MAP_FILE)

	confusion_matrix = np.zeros((len(label_map_dict.keys()), len(label_map_dict.keys())), dtype=np.uint)

	counter = 0
	total_boxes = 0
	gt = []
	predictions = []
	labels = []
	boxes = []
	scores = []
	gt_xmin = []
	gt_ymin = []
	gt_xmax = []
	gt_ymax = []
	gt_classes = []
	for data_path in tqdm(sorted(glob.glob(CLUSTERING_OUTPUT + '*.npz'))):
		data = np.load(data_path)
		labels_ = data['labels']
		boxes_ = data['boxes']
		scores_ = data['scores']

		boxes_[:,0] = boxes_[:,0] * 1920 / 720
		boxes_[:,1] = boxes_[:,1] * 1080 / 405
		boxes_[:,2] = boxes_[:,2] * 1920 / 720
		boxes_[:,3] = boxes_[:,3] * 1080 / 405

		if boxes_.shape[0] is not labels_.shape[0]:
			print(data_path)
		else:
			labels_ = labels_map[labels_]
			pos = np.where(labels_ > 0)[0]
			labels_ = labels_[pos]
			boxes_ = boxes_[pos, :]
			scores_ = scores_[pos]

			labels.append(labels_)
			boxes.append(boxes_)
			scores.append(scores_)

			# Compute new metric
			# Load ground-truth results.
			filename = str(int(os.path.splitext(os.path.split(data_path)[1])[0])).zfill(6)
			xmin, ymin, xmax, ymax, classes, classes_text = parse_xml(GROUND_TRUTH_PATH + filename + '.xml', label_map_dict)

			cost_matrix = np.zeros((boxes_.shape[0], len(xmin)), dtype=float)

			for pred_idx in range(boxes_.shape[0]):
				for gt_idx in range(len(xmin)):

					pred_box = boxes_[pred_idx]
					gt_box = [xmin[gt_idx], ymin[gt_idx], xmax[gt_idx], ymax[gt_idx]]

					iou = computeIoU(pred_box, gt_box)

					if iou > 0:
						cost_matrix[pred_idx, gt_idx] = 1.0 / iou
					else:
						cost_matrix[pred_idx, gt_idx] = 99999.

			row_idxs, col_idxs = linear_sum_assignment(cost_matrix)

			for r_idx, c_idx in zip(row_idxs, col_idxs):

				if scores_[r_idx] >= 0:
					if (1.0 / cost_matrix[r_idx, c_idx]) > 0.5:

						pred_class_idx = labels_[r_idx]
						gt_class_idx = classes[c_idx]
						confusion_matrix[pred_class_idx - 1, gt_class_idx - 1] += 1
				else:
					if (1.0 / cost_matrix[r_idx, c_idx]) > 0.1:
						pred_class_idx = labels_[r_idx]
						gt_class_idx = classes[c_idx]
						confusion_matrix[pred_class_idx - 1, gt_class_idx - 1] += 1

	confusion_matrix_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
	confusion_matrix_normalized = np.nan_to_num(confusion_matrix_normalized)
	sns.heatmap(confusion_matrix_normalized*100., annot=True, cmap='Blues', fmt=".1f", xticklabels=["car", "ft", "fuel", "lgm", "lg", "mb", "pe", "pl", "pb", "st", "van"], yticklabels=["car", "ft", "fuel", "lgm", "lg", "mb", "pe", "pl", "pb", "st", "van"])
	plt.show()

	return confusion_matrix

if __name__ == '__main__':
	label_map_dict = label_map_util.get_label_map_dict(LABEL_MAP_PATH)

	# Please complete this list with the clusters real labels of the dataset. Example:
	labels_map = np.array([8, 8, 8, 2, 4, 7, 7, 7, 7, 7, 3, 7, 7, 7, 4, 7, 4, 4, 7, 4, 4, 4, 8])

	corloc_detector, corloc_per_class_detector, corloc_of, corloc_per_class_of = computer_corloc(label_map_dict)

	confusion_matrix = compute_confusion_matrix(label_map_dict, labels_map)
	print("Confusion matrix:")
	print(confusion_matrix)

	print('CorLoc Detector: ' + str(corloc_detector))
	print('CorLoc per Class Detector: ' + str(corloc_per_class_detector))
	print('CorLoc OF: ' + str(corloc_of))
	print('CorLoc per Class OF: ' + str(corloc_per_class_of))
	map, ap = compute_map(label_map_dict, labels_map)
	print('mAP: ' + str(map))
	print('AP: ' + str(ap))
