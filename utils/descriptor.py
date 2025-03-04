import os
import numpy as np
import cv2
import torch
from utils import resnet_big
import tensorrt as trt
from utils import common
TRT_LOGGER = trt.Logger()
from pathlib import Path

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def get_engine(engine_file_path="", dla=False):
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        # print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            if dla:
                runtime.DLA_core = 1
            return runtime.deserialize_cuda_engine(f.read())
    else:
        print("ERROR, NO TRT AVAILABLE")

class DescriptorModel:
    def __init__(self, version, kind, path_trt, dla, torch_use_fp16, end, batch=32):
        self.version = version
        self.kind = kind
        self.batch = batch
        self.dla = dla
        self.end = end
        self.use_fp16 = torch_use_fp16
        self.input_ren = torch.Size([1, 3, 224, 224])

        if kind == 'xavier':
            self.model = None
            self.init_xavier()
        elif kind == 'tflite':
            self.interpreter = None
            self.input_details = None
            self.output_details = None
            self.init_tflite()
        elif kind =='trt':
            if self.end == False:
                self.engine = get_engine(path_trt, self.dla)
                self.context = self.engine.create_execution_context()
                if self.dla:
                    self.inputs_trt, self.outputs_trt, self.bindings_trt, self.stream_trt = common.allocate_buffers(
                        self.engine, self.input_ren)

            if self.end == True:
                self.engine = 0
                self.context = 0
        else:
            self.model = None
            self.init_normal()

    def init_xavier(self):
        base_dir = Path(__file__).resolve().parent.parent
        model_path = str(base_dir / "pretrained_models/supcon.pth")
        state_dict = torch.load(model_path)['model']
        model = resnet_big.SupConResNet()

        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' of dataparallel
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

        self.model = model.float().eval().cuda()

    def init_tflite(self):
        if self.version == 'xception':
            model = tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', pooling='avg')
        elif self.version == 'vgg16':
            model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', pooling='avg')
        elif self.version == 'vgg19':
            model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', pooling='avg')
        elif self.version == 'resnet':
            model = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', pooling='avg')
        elif self.version == 'resnetv2':
            model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet', pooling='avg')
        elif self.version == 'inceptionv3':
            model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', pooling='avg')
        elif self.version == 'inceptionresnetv2':
            model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet',
                                                                                pooling='avg')
        elif self.version == 'mobilenet':
            model = tf.keras.applications.mobilenet.MobileNet(include_top=False, weights='imagenet', pooling='avg')
        elif self.version == 'mobilenetv2':
            model_path = 'descriptor_models_fp16/mobilenet_v2/tflite_model.tflite'
        elif self.version == 'densenet':
            model = tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', pooling='avg')
        elif self.version == 'nas':
            model = tf.keras.applications.nasnet.NASNetMobile(include_top=False, weights='imagenet', pooling='avg')
        else:
            model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet', pooling='avg')

        # Load TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def init_normal(self):
        if self.version == 'xception':
            model = tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', pooling='avg')
        elif self.version == 'vgg16':
            model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', pooling='avg')
        elif self.version == 'vgg19':
            model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', pooling='avg')
        elif self.version == 'resnetv2':
            model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet', pooling='avg')
        elif self.version == 'inceptionv3':
            model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', pooling='avg')
        elif self.version == 'inceptionresnetv2':
            model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', pooling='avg')
        elif self.version == 'mobilenet':
            model = tf.keras.applications.mobilenet.MobileNet(include_top=False, weights='imagenet', pooling='avg')
        elif self.version == 'mobilenetv2':
            model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet', pooling='avg')
        elif self.version == 'densenet':
            model = tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', pooling='avg')
        elif self.version == 'nas':
            model = tf.keras.applications.nasnet.NASNetMobile(include_top=False, weights='imagenet', pooling='avg')
        else:
            model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet', pooling='avg')

        self.model = model

    def compute_descriptors_normal(self, crops):
        crops = torch.Tensor(crops).permute(0,3,1,2).cuda()
        descriptors = self.model(crops)

        return descriptors

    def compute_descriptors_xavier(self, crops):
        # Get previous behaviour.
        crops = torch.Tensor(crops).permute(0,3,1,2)
        if self.use_fp16:
            descriptors = self.model(crops.cuda().half())
        else:
            descriptors = self.model(crops.cuda())

        return descriptors

    def compute_descriptors_trt(self, crops):

        if not self.dla:

            crops = torch.Tensor(crops).permute(0, 3, 1, 2)

            inputs_trt, outputs_trt, bindings_trt, stream_trt = common.allocate_buffers_descriptor(self.engine, crops.shape)
            inputs_trt[0].host = crops
            self.context.set_binding_shape(0, (crops.shape[0], 3, 224, 224))

            feats = common.do_inference_v2(self.context, bindings=bindings_trt, inputs=inputs_trt,
                                           outputs=outputs_trt, stream=stream_trt)
            feats = feats[0].reshape((crops.shape[0], -1))

            return feats

        else:

            crops = torch.Tensor(crops).permute(0,3,1,2)
            feats_complete = np.zeros((crops.shape[0], 128))

            for i in range(crops.shape[0]):

                self.inputs_trt[0].host = crops[i:(i+1)]

                feats = common.do_inference_v2(self.context, bindings=self.bindings_trt, inputs=self.inputs_trt,
                                                   outputs=self.outputs_trt, stream=self.stream_trt)

                feats = feats[0].reshape((1, -1))
                feats_complete[i:(i + 1)] = feats

            return feats_complete[0:crops.shape[0]]

    def compute_descriptors_tflite(self, crops):
        descriptors = []
        for i in range(crops.shape[0]):
            self.interpreter.set_tensor(self.input_details[0]['index'],
                                        np.float32(np.expand_dims(crops[i, :, :, :], axis=0)))
            self.interpreter.invoke()
            descriptors_ = np.squeeze(self.interpreter.get_tensor(self.output_details[0]['index']))
            descriptors.append(descriptors_)

        descriptors = np.concatenate(descriptors)
        return descriptors

    def compute_descriptors(self, boxes, image):

        data_frame = []
        for j in boxes:
            c1, c2 = [int(j[0]), int(j[1])], [int(j[2]), int(j[3])]  # xyxy
            c1[0] = max(np.int32(c1[0]), 0)
            c1[1] = max(np.int32(c1[1]), 0)
            c2[0] = min(np.int32(c2[0]), image.shape[1])
            c2[1] = min(np.int32(c2[1]), image.shape[0])


            if (c2[1] - c1[1]) > 0 and (c2[0] - c1[0]) > 0:
                im = image[c1[1]:c2[1], c1[0]:c2[0], :]
                im = np.float32(im/255.0)
                data_frame.append(np.expand_dims(
                    (np.float32(cv2.resize(im, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)) - [0.485, 0.456,
                                                                                                   0.406])/[0.229, 0.224, 0.225], axis=0))

        data = np.concatenate(data_frame)
        if len(data.shape) == 3:
            data = np.expand_dims(data, axis=0)

        # Computation
        if self.kind == 'xavier':
            descriptors = self.compute_descriptors_xavier(data)
        elif self.kind == 'tflite':
            descriptors = self.compute_descriptors_tflite(data)
        elif self.kind == 'trt':
            descriptors = self.compute_descriptors_trt(data)
        else:
            descriptors = self.compute_descriptors_normal(data)

        return descriptors