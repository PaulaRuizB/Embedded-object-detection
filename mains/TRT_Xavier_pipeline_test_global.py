# import umap
import cv2
import glob
import numpy as np
import os
import argparse
import time
import tensorrt as trt
from utils.detection import DetectionModel
from utils.of import OFModel
from utils.descriptor import DescriptorModel
from utils.nms import NMSModel
from utils.cluster import ClusterModel
import torch
from utils.general import non_max_suppression
from utils.general import scale_coords
from utils.energy_meter_siroco import EnergyMeter

################# Parameters #################
OF_MODEL_PATH = 'none'
IMAGES_SUFIX = '*.jpg'
UNKNOWN_ID = 1000
IM_SCALE = 3
##############################################
TRT_LOGGER = trt.Logger()

def get_engine(engine_file_path=""):
    if os.path.exists(engine_file_path):
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        print("ERROR, NO TRT AVAILABLE")


if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser(description='Evaluates a detection pipeline')
    parser.add_argument('--path_trt', type=str, required=True, help="Path trt file")
    parser.add_argument('--path_trt_descriptor', type=str, required=True, help="Path trt file")
    parser.add_argument('--path_trt_cluster', type=str, required=False, help="Path trt file cluster")
    parser.add_argument('--kind_of', type=str, required=True, help="trt, tflite or normal")
    parser.add_argument('--path_images', default='', type=str, required=True,
                        help="Path trt file")
    parser.add_argument('--output_path', type=str, required=True, help="Output path")
    parser.add_argument('--path_clusters', type=str, required=False, help="Path trt file")
    parser.add_argument('--kind_detection', type=str, required=True, help="trt, tflite or normal")
    parser.add_argument('--kind_descriptor', type=str, required=True, help="trt, tflite or normal")
    parser.add_argument('--version_detection', type=str, required=True,
                        help="faster, ssd_mobilenet_v1_fpn, ssd_mobilenet_v2_oid, ssd_resnet50, yolov4_resnet50, or ssdlite")
    parser.add_argument('--version_descriptor', type=str, required=True,
                        help="xception, vgg16, vgg19, resnet, resnetv2, inceptionv3, inceptionresnetv2, mobilenet, mobilenetv2, densenetor nas")
    parser.add_argument('--clustering_algorithm', type=str, required=True, help="hdbscan or kmeans")
    parser.add_argument('--compression_algorithm', type=str, required=True, help="umap or pca")
    parser.add_argument('--dla', default=False, action='store_true')
    parser.add_argument('--n_components', type=int, required=False, default=128, help="n_components umap")
    parser.add_argument('--torch_use', default=False, action='store_true')
    parser.add_argument('--torch_use_fp16', default=False, action='store_true')
    parser.add_argument('--int8', default=False, action='store_true')
    parser.add_argument('--measure_gpu', default=False, action='store_true')
    parser.add_argument('--measure_cpu', default=False, action='store_true')
    parser.add_argument('--measure_dla', default=False, action='store_true')
    parser.add_argument('--max_batch', type=int, required=False, default=16, help="batch size")


    args = parser.parse_args()
    path_trt = args.path_trt
    path_trt_descriptor = args.path_trt_descriptor
    path_trt_cluster = args.path_trt_cluster
    kind_of = args.kind_of
    kind_detection = args.kind_detection
    kind_descriptor = args.kind_descriptor
    version_detection = args.version_detection
    version_descriptor = args.version_descriptor
    clustering_algorithm = args.clustering_algorithm
    compression_algorithm = args.compression_algorithm
    IMAGES_PATH = args.path_images
    OUTPUT_PATH = args.output_path
    PATH_CLUSTERS = args.path_clusters
    dla = args.dla
    n_components = args.n_components
    torch_use = args.torch_use
    torch_use_fp16 = args.torch_use_fp16
    int8 = args.int8
    max_batch = args.max_batch
    measure_gpu = args.measure_gpu
    measure_cpu = args.measure_cpu
    measure_dla = args.measure_dla

    # Initialize models.
    of_model = OFModel(kind_of, OF_MODEL_PATH)

    #Size image
    size_rpn = int(np.prod(torch.Size([1, 3, 384, 640])))

    detection_model = DetectionModel(version_detection, kind_detection, path_trt, dla, int8, size_rpn, end=False)
    descriptor_model = DescriptorModel(version_descriptor, kind_descriptor, path_trt_descriptor, dla,
                                       torch_use_fp16, end=False)
    nms_model = NMSModel()

    if measure_gpu:
        energy_measurer_GPU = EnergyMeter('xavier', 2, 'GPU', 0)
        energy_measurer_GPU.start()
    if measure_cpu:
        energy_measurer_CPU = EnergyMeter('xavier', 2, 'CPU', 0)
        energy_measurer_CPU.start()
    if measure_dla:
        energy_measurer_DLA = EnergyMeter('xavier', 2, 'CV', 0)
        energy_measurer_DLA.start()


    PATH_CLUSTERS = PATH_CLUSTERS + '{}'.format(clustering_algorithm) + '_{}'.format(
        compression_algorithm) + '_{}'.format(n_components) + '/clusters/'

    print(PATH_CLUSTERS)
    cluster_model = ClusterModel(clustering_algorithm, compression_algorithm, n_components, path_trt_cluster, torch_use,
                                 end=False)
    try:
        cluster_model.load_model(PATH_CLUSTERS)
    except:
        PATH_CLUSTERS = ''
        cluster_model.load_model(PATH_CLUSTERS)

    # Create output folder.
    try:
        os.mkdir(OUTPUT_PATH + '')
        os.mkdir(OUTPUT_PATH + 'npz/')
        os.mkdir(OUTPUT_PATH + 'test/')
    except:
        pass

    previous_image = None
    previous_boxes = None
    previous_of = None
    boxes = None

    tclustermodel = []
    tclusterhdbscan = []
    eclusterhdbscan = []
    eclustermodel = []
    tprep = []
    eprep = []
    td = []
    tpostinf = []

    ed = []
    eddla = []
    tnms = []
    enms = []
    tdescriptordlaortorch = []
    tpreprenc = []
    tbuffersc = []
    tinfrenc = []

    edescriptor = []
    edescriptordla = []
    tof = []
    eof = []

    save_first = True
    save_second = False
    # Loop over the test images.
    tcompleto=time.time()
    for idx, image_path in enumerate(sorted(glob.glob(IMAGES_PATH + IMAGES_SUFIX))):
        if (idx % 5 == 0 and idx > 500): # Steps between frames and number of images
            #print(idx)
            img_original = cv2.imread(image_path)
            width = 720
            height = 405
            img_original = cv2.resize(img_original, (width, height))

            img = img_original.copy()

            ######### Preprocessing #########
            #time.sleep(5)
            if measure_cpu:
                energy_measurer_CPU.start_measuring()

            tprep1 = time.time()

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
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                     value=color)  # add border
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)

            if torch_use_fp16:
                img = img.half()
            else:
                img = img.float()  # TRT models
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            tprep2 = time.time()

            if measure_cpu:
                energy_measurer_CPU.stop_measuring()
                eprep.append(energy_measurer_CPU.total_energy)

            tprep.append(tprep2 - tprep1)

            #################################################


            ###################### RPN ######################
            #time.sleep(5)
            if measure_dla:
                energy_measurer_DLA.start_measuring()

            if measure_gpu:
                energy_measurer_GPU.start_measuring()

            if previous_image is not None:

                #Inference
                td1 = time.time()
                # Detections YOLOV4+ResNet50
                pred = detection_model.compute_detections(img)

                td2 = time.time()

                #POST inference
                tpostinf1 = time.time()

                if not torch_use:
                    pred = torch.tensor(pred[3])
                    pred = torch.reshape(pred, (1, -1, 96))

                pred = pred.detach()

                # Non max suppression
                pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)

                # Scores
                scores_ = (torch.clone(pred[0][:, 4])).tolist()

                # Clases
                classes_ = torch.clone(pred[0][:, 5])

                # Bouding boxes recalculated for the original image
                boxes_ = scale_coords(img.shape[2:], torch.clone(pred[0][:, :4]),
                                      img_original.shape).round()
                tpostinf2 = time.time()

                if measure_dla:
                    energy_measurer_DLA.stop_measuring()
                    eddla.append(energy_measurer_DLA.total_energy)

                if measure_gpu:
                    energy_measurer_GPU.stop_measuring()
                    ed.append(energy_measurer_GPU.total_energy)

                td.append(td2 - td1)
                tpostinf.append(tpostinf2-tpostinf1)

                #################################################


                ###################### OFRP ######################
                #time.sleep(5)
                if measure_cpu:
                    energy_measurer_CPU.start_measuring()
                seconds = time.time()

                # Optical flow detection
                boxes, optical_flow = of_model.compute(previous_image, img_original.copy(), IM_SCALE,
                                                       idx, OUTPUT_PATH + 'combination_train/')  # yxyx

                seconds2 = time.time()

                if measure_cpu:
                    energy_measurer_CPU.stop_measuring()
                    eof.append(energy_measurer_CPU.total_energy)
                tof.append(seconds2 - seconds)
                #################################################

                # Combine detections.
                ###################### NMS ######################
                #time.sleep(5)
                if measure_cpu:
                    energy_measurer_CPU.start_measuring()

                tnms1 = time.time()

                boxes_, classes_, scores_ = nms_model.combine_detections_full(boxes, previous_of,
                                                                              previous_boxes, boxes_, classes_,
                                                                              scores_, img_original.shape[1],
                                                                              img_original.shape[0], UNKNOWN_ID)
                tnms2 = time.time()

                if measure_cpu:
                    energy_measurer_CPU.stop_measuring()
                    enms.append(energy_measurer_CPU.total_energy)

                tnms.append(tnms2 - tnms1)
                #################################################

                boxes_corloc = boxes_

                if boxes_.shape[0] > max_batch:
                    arr = np.random.permutation(np.arange(boxes_.shape[0]))
                    boxes_ = boxes_[arr[:max_batch]]

                # Measure mAP OF measure_of=TRUE
                measure_of = False
                if measure_of:
                    boxes_of = []
                    # out_name = os.path.splitext(os.path.split(str(image_path))[1])[0]
                    # np.savez(OUTPUT_PATH + 'npz/' + out_name + '.npz', boxes=boxes_corloc)
                    for i in boxes_:
                        if i[4] == 0 or i[4] == 2:
                            boxes_of.append(i)

                    boxes_ = (np.array(boxes_of))

                measure_rpn = False
                if measure_rpn:
                    boxes_of = []
                    # out_name = os.path.splitext(os.path.split(str(image_path))[1])[0]
                    # np.savez(OUTPUT_PATH + 'npz/' + out_name + '.npz', boxes=boxes_corloc)
                    for i in boxes_:
                        if i[4] == 1 or i[4] == 2:
                            boxes_of.append(i)

                    boxes_ = (np.array(boxes_of))


                # else:
                #    boxes_ = boxes_[:, :-1]

                if np.size(boxes_) > 0:

                    boxes_ = boxes_[:, :-1]

                    if boxes is not None:
                        previous_boxes = boxes.copy()
                        previous_of = optical_flow.copy()

                    # ###################### REN ######################
                    prueba_copy = img_original.copy()

                    #time.sleep(5)
                    # Get features.
                    if measure_dla:
                        energy_measurer_DLA.start_measuring()

                    if measure_gpu:
                        energy_measurer_GPU.start_measuring()

                    #if torch_use_fp16:
                    #    descriptor_model.model.half()

                    feats, tprepren, tbuffers, tinfren = descriptor_model.compute_descriptors(boxes_, prueba_copy)

                    if dla:
                        tdescriptordlaortorch1 = time.time()
                        feats = np.float32(feats)
                        tdescriptordlaortorch2 = time.time()

                    if torch_use:
                        tdescriptordlaortorch1 = time.time()
                        feats = feats.detach().cpu()  # cpu
                        tdescriptordlaortorch2 = time.time()

                    if measure_dla:
                        energy_measurer_DLA.stop_measuring()
                        edescriptordla.append(energy_measurer_DLA.total_energy)

                    if measure_gpu:
                        energy_measurer_GPU.stop_measuring()
                        edescriptor.append(energy_measurer_GPU.total_energy)

                    if torch_use or dla:
                        tdescriptordlaortorch.append(tdescriptordlaortorch2 - tdescriptordlaortorch1)
                    tpreprenc.append(tprepren)
                    tbuffersc.append(tbuffers)
                    tinfrenc.append(tinfren)
                    # #################################################


                    # # Test
                    # ###################### CLUSTER ######################
                    #time.sleep(5)
                    energy_measurer_GPU.start_measuring()
                    tcluster1 = time.time()

                    if measure_cpu:
                        labels, scores, boxes2, tcluster_engine, energycluster_engine, tcluster_hdbscan, energycluster_hdbscan = cluster_model.predict(
                            feats, boxes_, None, energy_measurer_CPU)  # cpu
                    if measure_gpu:
                        labels, scores, boxes2, tcluster_engine, energycluster_engine, tcluster_hdbscan, energycluster_hdbscan = cluster_model.predict(
                            feats, boxes_, energy_measurer_GPU, None)  # cpu
                    if not measure_gpu and not measure_cpu:
                        labels, scores, boxes2, tcluster_engine, energycluster_engine, tcluster_hdbscan, energycluster_hdbscan = cluster_model.predict(
                            feats, boxes_, None, None)  # cpu

                    energy_measurer_GPU.stop_measuring()
                    tcluster2 = time.time()

                    tclustermodel.append(tcluster_engine)
                    tclusterhdbscan.append(tcluster_hdbscan)

                    eclustermodel.append(energycluster_engine)
                    eclusterhdbscan.append(energycluster_hdbscan)
                    # #################################################

            else:
                if measure_gpu:
                    energy_measurer_GPU.stop_measuring()

                if measure_dla:
                    energy_measurer_DLA.stop_measuring()

            previous_image = img_original.copy()

    if measure_gpu:
        energy_measurer_GPU.finish()

    if measure_dla:
        energy_measurer_DLA.finish()

    if measure_cpu:
        energy_measurer_CPU.finish()
    tcompleto2=time.time()

    print("The variance for preprocessing time is {} and for CPU energy {}".format(
        np.var(tprep[6:]), np.var(eprep[6:])))
    print("The variance for detection time is {}, for GPU energy {}, and for DLA energy {}".format(
        np.var(td[5:]), np.var(ed[5:]), np.var(eddla[5:])))
    print("The variance for optical flow time is {} and for CPU energy {}".format(np.var(tof[5:]),
                                                                                  np.var(eof[5:])))
    print("The variance for NMS time is {} and for CPU energy {}".format(np.var(tnms[5:]),
                                                                         np.var(enms[5:])))
    print("The variance for descriptor time is {}, for GPU energy {}, and for DLA energy {}".format(
        np.var(tinfrenc[5:]),
        np.var(edescriptor[5:]),
        np.var(edescriptordla[5:])))
    print("The variance for cluster model time is {} and for GPU energy {}".format(
        np.var(tclustermodel[5:]), np.var(eclustermodel[5:])))
    print("The variance for hdbscan time is {} and for CPU energy {}".format(np.var(tclusterhdbscan[5:]),
                                                                             np.var(eclusterhdbscan[5:])))
    print("-----------------------------------------------------------------------------------------------")

    tclustermodel = sum(tclustermodel[5:]) / (len(tclustermodel) - 5)
    tclusterhdbscan = sum(tclusterhdbscan[5:]) / (len(tclusterhdbscan) - 5)
    tprep = sum(tprep[6:]) / (len(tprep) - 6)
    td = sum(td[5:]) / (len(td) - 5)
    tpostinf = sum(tpostinf[5:]) / (len(tpostinf) - 5)
    tnms = sum(tnms[5:]) / (len(tnms) - 5)
    tdescriptordlaortorch = sum(tdescriptordlaortorch[5:]) / (len(tdescriptordlaortorch) - 5)
    tpreprenc = sum(tpreprenc[5:]) / (len(tpreprenc) - 5)
    tbuffersc = sum(tbuffersc[5:]) / (len(tbuffersc) - 5)
    tinfrenc = sum(tinfrenc[5:]) / (len(tinfrenc) - 5)
    tof = sum(tof[5:]) / (len(tof) - 5)

    eclustermodel = sum(eclustermodel[5:]) / (len(eclustermodel) - 5)
    eclusterhdbscan = sum(eclusterhdbscan[5:]) / (len(eclusterhdbscan) - 5)
    eprep = sum(eprep[6:]) / (len(eprep) - 6)
    ed = sum(ed[5:]) / (len(ed) - 5)
    enms = sum(enms[5:]) / (len(enms) - 5)
    edescriptor = sum(edescriptor[5:]) / (len(edescriptor) - 5)
    eof = sum(eof[5:]) / (len(eof) - 5)
    eddla = sum(eddla[5:]) / (len(eddla) - 5)
    edescriptordla = sum(edescriptordla[5:]) / (len(edescriptordla) - 5)

    print("The average preprocessing time is {} ms and CPU energy {} mJ".format(tprep * 1000.0, eprep * 5.0))
    print(
        "The average inference time for detection is {} ms, GPU energy {} mJ, and DLA energy {} mJ".format(td * 1000.0,
                                                                                                           ed * 5.0,
                                                                                                           eddla * 5.0))
    print("The average post-detection time is {} ms".format(tpostinf * 1000.0))
    print("The average optical flow time is {} ms and CPU energy {} mJ".format(tof * 1000.0, eof * 5.0))
    print("The average NMS time is {} ms and CPU energy {} mJ".format(tnms * 1000.0, enms * 5.0))
    print(
        "The extra average time for the DLA or TORCH descriptor is {} ms, total GPU energy {} mJ, and DLA energy {} mJ".format(
            tdescriptordlaortorch * 1000.0,
            edescriptor * 5.0,
            edescriptordla * 5.0))
    print("The average pre-REN time is {} ms".format(tpreprenc * 1000.0))
    print("The average buffer time in REN is {} ms".format(tbuffersc * 1000.0))
    print("The average REN inference time is {} ms".format(tinfrenc * 1000.0))

    print("The average cluster model time is {} ms and GPU energy {} mJ".format(tclustermodel * 1000.0,
                                                                                eclustermodel * 5.0))
    print("The average hdbscan time is {} ms and CPU energy {} mJ".format(tclusterhdbscan * 1000.0,
                                                                          eclusterhdbscan * 5.0))

    detection_model = DetectionModel(version_detection, kind_detection, path_trt, dla, int8, size_rpn, end=True)
    descriptor_model = DescriptorModel(version_descriptor, kind_descriptor, path_trt_descriptor, dla,
                                       torch_use_fp16, end=True)
    cluster_model = ClusterModel(clustering_algorithm, compression_algorithm, n_components, path_trt_cluster, torch_use,
                                 end=True)

