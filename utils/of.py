from __future__ import absolute_import, division, print_function
import imutils
import cv2
import numpy as np
def process_motion_vectors(mv):
    with mv.rlock():
        flow = np.float32(mv.cpu()) / (1 << 5)
    return flow

class OFModel:
    def __init__(self, kind, model_path, min_area=200, threshold=0.3):
        self.kind = kind
        self.path = model_path
        self.min_area = min_area
        self.threshold = threshold
        self.model = None
        self.gpu_flow = None

        if kind == 'trt':
            print("TRT OF")
        elif kind == 'tflite':
            self.init_tflite()
        else:
            self.init_normal()

    def init_tflite(self):
        print("WARN: This a CPU version of Farneback algorithm, not a deep learning model!")

    def init_normal(self):
        print("WARN: This a GPU version of Farneback algorithm, not a deep learning model!")

    def compute(self, image1, image2, scale, idx, path_of):
        if self.kind == 'trt':
            optical_flow = self.compute_trt(image1, image2, scale)
        elif self.kind == 'tflite':
            optical_flow = self.compute_tflite(image1, image2, scale)
        else:
            optical_flow, image2 = self.compute_normal(image1, image2, scale)

        # Compute OF module.

        if self.kind =='trt':
            min_area = 400
            threshold = 1.5
            kernel = False
            of_module = np.sqrt(optical_flow[:, :, 0] ** 2 + optical_flow[:, :, 1] ** 2)
            of_module[of_module < threshold] = 0
            of_module = np.uint8(((of_module - np.amin(of_module)) / (1e-6 + (np.amax(of_module) - np.amin(of_module)))) * 255)

        elif self.kind == 'corloc':
            min_area = 100
            threshold = 0.7
            kernel = True

            of_module = np.sqrt(optical_flow[:, :, 0] ** 2 + optical_flow[:, :, 1] ** 2)
            of_module[of_module < threshold] = 0
            of_module = np.uint8((of_module - np.amin(of_module)) / (np.amax(of_module) - np.amin(of_module)) * 255)

        else:
            min_area = 600/np.float32(scale*scale)
            threshold = 1
            kernel = True

            of_module = np.sqrt(optical_flow[:, :, 0] ** 2 + optical_flow[:, :, 1] ** 2)
            of_module[of_module < threshold] = 0
            of_module = np.uint8(((of_module - np.amin(of_module)) / (1e-6 + (np.amax(of_module) - np.amin(of_module)))) * 255)

        if kernel:
            of_module = cv2.medianBlur(of_module, 5)

        cnts = cv2.findContours(of_module.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # loop over the contours
        boxes = []
        for c in cnts:

            # if the contour is too small, ignore it
            if cv2.contourArea(c) < min_area:
                continue

            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)

            box = np.array([x, y, x + w, y + h])
            boxes.append(box)

        return np.asarray(boxes), optical_flow

    def compute_trt(self, image1, image2, scale):
        backend = vpi.Backend.NVENC
        image1 = cv2.GaussianBlur(image1, (9, 9), 0)
        image2 = cv2.GaussianBlur(image2, (9, 9), 0)

        # Generate the predictions and display them
        image1 = vpi.asimage(image1, vpi.Format.BGR8) \
            .convert(vpi.Format.NV12_ER, backend=vpi.Backend.CUDA) \
            .convert(vpi.Format.NV12_ER_BL, backend=vpi.Backend.VIC)
        with vpi.Backend.VIC:
            image1 = image1.rescale((image1.width * 4, image1.height * 4), interp=vpi.Interp.LINEAR)
        image2 = vpi.asimage(image2, vpi.Format.BGR8) \
            .convert(vpi.Format.NV12_ER, backend=vpi.Backend.CUDA) \
            .convert(vpi.Format.NV12_ER_BL, backend=vpi.Backend.VIC)
        with vpi.Backend.VIC:
            image2 = image2.rescale((image2.width * 4, image2.height * 4), interp=vpi.Interp.LINEAR)

        with backend:
            motion_vectors = vpi.optflow_dense(image1, image2, quality=vpi.OptFlowQuality.HIGH)


        # Turn motion vectors into an image
        motion_image = process_motion_vectors(motion_vectors)

        return motion_image

    def compute_tflite(self, image1, image2, scale):
        im1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        im2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        width = np.int(image1.shape[1] / np.float32(scale))
        height = np.int(image1.shape[0] / np.float32(scale))
        optical_flow = cv2.calcOpticalFlowFarneback(cv2.resize(im1, (width, height)), cv2.resize(im2, (width, height)),
                                                    None, np.sqrt(2.0) / 2.0, 5, 10, 2, 7, 1.5,
                                                    cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        return optical_flow

    def compute_normal(self, image1, image2, scale):
        # Convert frames to gray-scale
        im1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        im2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        im1 = cv2.GaussianBlur(im1, (7, 7), 0)
        im2 = cv2.GaussianBlur(im2, (7, 7), 0)

        optical_flow = cv2.calcOpticalFlowFarneback(im1, im2, None, 0.75, 1, 15, 2, 3, 0.9, 0)

        return optical_flow, im2
