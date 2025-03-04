import cv2
import glob
import numpy as np
import os
import argparse
import pickle
import torch
import sys
sys.path.append('../')
from utils.detection import DetectionModel
from utils.of import OFModel
from utils.descriptor import DescriptorModel
from utils.nms import NMSModel
from utils.cluster import ClusterModel
from utils.general import non_max_suppression
from utils.general import scale_coords

################# Parameters #################
OF_MODEL_PATH = 'none'
IMAGES_SUFIX = '*.jpg'
UNKNOWN_ID = 1000
IM_SCALE = 2
##############################################



if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser(description='Evaluates a detection pipeline')
    parser.add_argument('--path_images', type=str, required=True, help="Path trt file")
    parser.add_argument('--kind_of', type=str, required=True, help="trt, tflite or normal")
    parser.add_argument('--path_trt', type=str, required=True, help="Path trt file")
    parser.add_argument('--path_trt_descriptor', type=str, required=True, help="Path trt file")
    parser.add_argument('--output_path', type=str, required=True, help="Path trt file")
    parser.add_argument('--kind_detection', type=str, required=True, help="trt, tflite or normal")
    parser.add_argument('--kind_descriptor', type=str, required=True, help="trt, tflite or normal")
    parser.add_argument('--version_detection', type=str, required=True,
                        help="faster, ssd_mobilenet_v1_fpn, ssd_mobilenet_v2_oid, ssd_resnet50 or ssdlite")
    parser.add_argument('--version_descriptor', type=str, required=True,
                        help="xception, vgg16, vgg19, resnet, resnetv2, inceptionv3, inceptionresnetv2, mobilenet, mobilenetv2, densenetor nas")
    parser.add_argument('--clustering_algorithm', type=str, required=True, help="hdbscan or kmeans")
    parser.add_argument('--compression_algorithm', type=str, required=True, help="umap or pca")
    parser.add_argument('--dla', default=False, action='store_true')
    parser.add_argument('--torch_use', default=False, action='store_true')
    parser.add_argument('--n_components', type=int, required=False, default=0, help="n_components umap")
    parser.add_argument('--torch_use_fp16', default=False, action='store_true')
    parser.add_argument('--int8', default=False, action='store_true')
    parser.add_argument('--path_trt_cluster', type=str, required=True, help="Path trt file cluster")
    parser.add_argument('--max_batch', type=int, required=False, default=16, help="batch size")

    args = parser.parse_args()

    kind_of = args.kind_of
    kind_detection = args.kind_detection
    kind_descriptor = args.kind_descriptor
    path_trt = args.path_trt
    IMAGES_PATH= args.path_images
    OUTPUT_PATH = args.output_path
    path_trt_descriptor = args.path_trt_descriptor
    version_detection = args.version_detection
    version_descriptor = args.version_descriptor
    clustering_algorithm = args.clustering_algorithm
    compression_algorithm = args.compression_algorithm
    dla = args.dla
    torch_use = args.torch_use
    n_components = args.n_components
    torch_use_fp16 = args.torch_use_fp16
    int8 = args.int8
    path_trt_cluster = args.path_trt_cluster
    max_batch = args.max_batch

    # Create output folders.
    OUTPUT_PATH = OUTPUT_PATH+'{}'.format(clustering_algorithm) + '_{}'.format(compression_algorithm)+'_{}'.format(n_components)+'/'
    try:
        os.mkdir(OUTPUT_PATH)
        os.mkdir(OUTPUT_PATH+'features/')
        os.mkdir(OUTPUT_PATH+'clusters/')
        os.mkdir(OUTPUT_PATH+'combination_train/')
        os.mkdir(OUTPUT_PATH+'test/')

    except:
        pass

    print(OUTPUT_PATH)
    size_rpn = int(np.prod(torch.Size([1, 3, 384, 640])))


    # Initialize models.
    of_model = OFModel(kind_of, OF_MODEL_PATH)
    detection_model = DetectionModel(version_detection, kind_detection, path_trt, dla, int8, size_rpn, end=False)
    descriptor_model = DescriptorModel(version_descriptor, kind_descriptor, path_trt_descriptor, dla, torch_use_fp16, end=False)
    nms_model = NMSModel()
    cluster_model = ClusterModel(clustering_algorithm, compression_algorithm, n_components, path_trt_cluster, torch_use, end=False)

    previous_image = None
    previous_boxes = None
    previous_of = None
    boxes = None
    final_features = []
    final_bbs = []
    final_scores = []
    paths = []
    images = []

    # Loop over the training images.
    for idx, image_path in enumerate(sorted(glob.glob(IMAGES_PATH + IMAGES_SUFIX))):
        if (idx % 5 == 0):
            number = idx

            ######### Preprocessing #########
            img_original = cv2.imread(image_path)
            width = 720
            height = 405
            img_original = cv2.resize(img_original, (width, height))
            img = img_original.copy()
            shape = img.shape[:2]
            new_shape = (640, 640)
            r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
            ratio = r, r  # width, height ratios
            new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
            dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
            dw, dh = np.mod(dw, 128), np.mod(dh, 128)  # wh padding
            dw /= 2  # divide padding into 2 sides
            dh /= 2

            if shape[::-1] != new_unpad:  # resize
                img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            color = (114, 114, 114)
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)

            if torch_use_fp16:
                img = img.half()
            else:
                img = img.float()

            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Detection
            if torch_use_fp16:
                detection_model.model.half()
            pred = detection_model.compute_detections(img)
            pred = pred.detach()

            # Non max suppression
            pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)

            # Scores
            scores_ = (torch.clone(pred[0][:, 4])).tolist()

            # Classes
            classes_ = torch.clone(pred[0][:, 5])

            # Bouding boxes recalculated for the original image
            boxes_ = scale_coords(img.shape[2:], torch.clone(pred[0][:, :4]), img_original.shape).round()

            # Optical flow detection
            if previous_image is not None:
                boxes, optical_flow = of_model.compute(previous_image, img_original.copy(), IM_SCALE, idx, OUTPUT_PATH + 'combination_train/')  # yxyx
                previous_of = optical_flow.copy()

            # Combine detections
            if previous_boxes is not None:
                boxes_, classes_, scores_ = nms_model.combine_detections_full(boxes, previous_of, previous_boxes, boxes_,
                                                                              classes_, scores_, img_original.shape[1],
                                                                              img_original.shape[0], UNKNOWN_ID, th_bb=0.3)
                if boxes_.shape[0]>max_batch:
                    arr = np.random.permutation(np.arange(boxes_.shape[0]))
                    boxes_ = boxes_[arr[:max_batch]]


                measure_of = False
                if measure_of:
                    boxes_of = []
                    # out_name = os.path.splitext(os.path.split(str(image_path))[1])[0]
                    # np.savez(OUTPUT_PATH + 'npz/' + out_name + '.npz', boxes=boxes_corloc)
                    for i in boxes_:
                        if i[4] == 0 or i[4] == 2:
                            boxes_of.append(i)

                    boxes_ = (np.array(boxes_of))

                measure_rpn = True
                if measure_rpn:
                    boxes_of = []
                    # out_name = os.path.splitext(os.path.split(str(image_path))[1])[0]
                    # np.savez(OUTPUT_PATH + 'npz/' + out_name + '.npz', boxes=boxes_corloc)
                    for i in boxes_:
                        if i[4] == 1 or i[4] == 2:
                            boxes_of.append(i)

                    boxes_ = (np.array(boxes_of))


                if np.size(boxes_) > 0:
                    boxes_ = boxes_[:,:-1]

                    if torch_use_fp16:
                        descriptor_model.model.half()

                    feats = descriptor_model.compute_descriptors(boxes_, img_original.copy())
                    if torch_use:
                        feats = feats.detach().cpu()

                    # Stack data.
                    paths.append([image_path] * boxes_.shape[0])
                    final_features.append(feats)
                    final_bbs.append(boxes_)
                    final_scores.append(scores_)

                    # Save current data.
                    out_name = os.path.split(image_path)[1]
                    np.savez(OUTPUT_PATH + 'features/' + out_name[0:-4] + '.npz', features=feats,
                             paths=image_path, scores=scores_, boxes=boxes_)

            if boxes is not None:
                previous_boxes = boxes.copy()

            previous_image = img_original.copy()


    # Save intermediate data
    np.savez(OUTPUT_PATH + 'features/', features=np.vstack(final_features), scores=np.hstack(final_scores), boxes=np.vstack(final_bbs))

    # Build the clusterer
    clusterer, reducer = cluster_model.build(np.vstack(final_features), np.vstack(final_bbs))
    file_clusterer = open(OUTPUT_PATH + 'clusters/' +'clusterer.obj', 'wb')
    pickle.dump(clusterer, file_clusterer)

    if reducer != None:
        file_reducer = open(OUTPUT_PATH + 'clusters/' + 'reducer.obj', 'wb')
        pickle.dump(reducer, file_reducer)

    flat_paths = [item for sublist in paths for item in sublist]
    if clustering_algorithm == 'hdbscan':
        cluster_model.save_clusters(flat_paths, np.vstack(final_bbs), clusterer.labels_, clusterer.probabilities_,
                                OUTPUT_PATH + 'clusters/')
    else:
        cluster_model.save_clusters(flat_paths, np.vstack(final_bbs), clusterer.labels_, None,
                                OUTPUT_PATH + 'clusters/')

