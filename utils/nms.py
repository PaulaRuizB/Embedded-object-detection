import numpy as np

class NMSModel:
	def __init__(self):
		pass
	def remove_static_bbs(self, of, current_bbs, previous_bbs, width, height, min_threshold=0.1):
		bbs = []
		for i in range(previous_bbs.shape[0]):
			previous_bbs[i] = previous_bbs[i]
			# Compute mean flow.
			x = max(np.int32(previous_bbs[i][0]), 0)
			y = max(np.int32(previous_bbs[i][1]), 0)
			xe = min(np.int32(previous_bbs[i][2]), of.shape[1])
			ye = min(np.int32(previous_bbs[i][3]), of.shape[0])

			bb = of[y:ye, x:xe, :]
			mean_flow_x = np.mean(bb[:, :, 0])
			mean_flow_y = np.mean(bb[:, :, 1])

			# Future BB.
			xf = max(np.int32(x + mean_flow_x), 0)
			xef = min(np.int32(xe + mean_flow_x), width)
			yf = max(np.int32(y + mean_flow_y), 0)
			yef = min(np.int32(ye + mean_flow_y), height)

			# Check if the future BB exists.
			for j in range(current_bbs.shape[0]):
				current_bbs[j] = current_bbs[j]
				xc = max(np.int32(current_bbs[j][0]), 0)
				yc = max(np.int32(current_bbs[j][1]), 0)
				xec = min(np.int32(current_bbs[j][2]), width)
				yec = min(np.int32(current_bbs[j][3]), height)
				iou = self.computeIoU([yf, xf, yef, xef], [yc, xc, yec, xec])
				if iou >= min_threshold:
					bbs.append(previous_bbs[i])
					break

		return np.asarray(bbs)


	def computeIoU(self, boxA, boxB):
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

	def combine_detections(self, detector_bbs, of_bbs, detector_classes, of_classes, detector_scores, of_scores, width, height, th_bb=0.4, th_prop=0.5):
		bbs = []
		classes = []
		scores = []
		selected = [False] * of_bbs.shape[0]
		for i in range(detector_bbs.shape[0]):
			x = max(np.int32(detector_bbs[i][1] * width), 0)
			y = max(np.int32(detector_bbs[i][0] * height), 0)
			xe = min(np.int32(detector_bbs[i][3] * width), width)
			ye = min(np.int32(detector_bbs[i][2] * height), height)
			det_prop = (ye - y) / np.float32(xe - x)
			for j in range(of_bbs.shape[0]):
				if not selected[j]:
					x2 = max(np.int32(of_bbs[j][1] * width), 0)
					y2 = max(np.int32(of_bbs[j][0] * height), 0)
					xe2 = min(np.int32(of_bbs[j][3] * width), width)
					ye2 = min(np.int32(of_bbs[j][2] * height), height)
					iou = self.computeIoU([y, x, ye, xe], [y2, x2, ye2, xe2])
					of_prop = (ye2 -y2) / np.float32(xe2 -x2)

					a = max(det_prop, of_prop)
					b = min(det_prop, of_prop)

					if (iou > th_bb) and ((b / a) > th_prop):
						selected[j] = True
						break

			bbs.append(detector_bbs[i])
			classes.append(detector_classes[i])
			scores.append(detector_scores[i])

		# Include non selected bbs.
		for j in range(of_bbs.shape[0]):
			if not selected[j]:
				bbs.append(of_bbs[j])
				classes.append(of_classes[j])
				scores.append(of_scores[j])

		return np.asarray(bbs), classes, scores

	def combine_detections_full(self, of_bbs, previous_of, previous_bbs, detector_bbs, detector_classes, detector_scores, width, height, unknown_id, th_bb=0.4, th_prop=0.5, th_of=0.07):
		if previous_of is not None and previous_bbs is not None:
			of_bbs = self.remove_static_bbs(previous_of, of_bbs, previous_bbs, width, height, th_of)
		of_classes = np.zeros(of_bbs.shape[0], dtype='int32') + unknown_id
		of_scores = np.zeros(of_bbs.shape[0]) + -1
		bbs = []
		classes = []
		scores = []
		selected = [False] * of_bbs.shape[0]
		both = False #[]* of_bbs.shape[0]
		medir_corloc_of = False	# Only OF: True

		if not medir_corloc_of:
			for i in range(detector_bbs.shape[0]):
				if detector_scores[i] > 0:
					(x, y), (xe, ye) = (int(detector_bbs[i][0]), int(detector_bbs[i][1])), (int(detector_bbs[i][2]), int(detector_bbs[i][3]))  # xyxy
					if (ye - y) > 0 and (xe - x) > 0:
						det_prop = (ye - y) / np.float32(xe - x)
						for j in range(of_bbs.shape[0]):
							if not selected[j]:
								(x2, y2), (xe2, ye2) = (int(of_bbs[j][0]), int(of_bbs[j][1])), (int(of_bbs[j][2]), int(of_bbs[j][3]))
								iou = self.computeIoU([y, x, ye, xe], [y2, x2, ye2, xe2])
								of_prop = (ye2 - y2) / np.float32(xe2 - x2)

								a = max(det_prop, of_prop)
								b = min(det_prop, of_prop)

								if (iou > th_bb) and ((b / a) > th_prop):
									selected[j] = True
									both = True

									break

						if both == True:
							bbs.append(np.concatenate((detector_bbs[i], [2])))
							both = False
						else:
							bbs.append(np.concatenate((detector_bbs[i], [1])))
						classes.append(detector_classes[i])
						scores.append(detector_scores[i])
		else:
			for i in range(detector_bbs.shape[0]):
				bbs.append(np.concatenate((detector_bbs[i], [1])))
				classes.append(detector_classes[i])
				scores.append(detector_scores[i])

		# Include non selected bbs.
		for j in range(of_bbs.shape[0]):
			if not selected[j] and of_bbs[j, 2] > 0 and of_bbs[j, 3] > 0:
				bbs.append(np.concatenate((of_bbs[j],[0])))
				classes.append(of_classes[j])
				scores.append(of_scores[j])

		return np.asarray(bbs), classes, scores

