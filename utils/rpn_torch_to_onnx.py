import argparse
import torch
import torch.nn as nn
from onnxsim import simplify
import models.common_resnext
from models.experimental import attempt_load
from utils.activations import Mish

def revert_sync_batchnorm(module):
    # this is very similar to the function that it is trying to revert:
    # https://github.com/pytorch/pytorch/blob/c8b3686a3e4ba63dc59e5dcfe5db3430df256833/torch/nn/modules/batchnorm.py#L679
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        module_output = nn.BatchNorm2d(module.num_features,
                                               module.eps, module.momentum,
                                               module.affine,
                                               module.track_running_stats)
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child))
    del module
    return module_output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='weights path')  # from yolov5/models/
    parser.add_argument('--img-size', nargs='+', type=int, default=[384, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')

    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)

    # Input
    img = torch.zeros((opt.batch_size, 3, *opt.img_size))  # image size(1,3,320,192) iDetection

    # Load PyTorch model
    model = attempt_load(opt.weights, map_location=torch.device('cpu'))  # load FP32 model

    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
        if isinstance(m, models.common_resnext.Conv) and isinstance(m.act, models.common_resnext.Mish):
            m.act = Mish()  # assign activation
        if isinstance(m, models.common_resnext.BottleneckCSP) or isinstance(m, models.common_resnext.BottleneckCSP2) \
                or isinstance(m, models.common_resnext.SPPCSP):
            if isinstance(m.bn, nn.SyncBatchNorm):
                m.bn = revert_sync_batchnorm(m.bn)
            if isinstance(m.act, models.common_resnext.Mish):
                m.act = Mish()  # assign activation
        if isinstance(m, models.common_resnext.resBottleneck):
            if isinstance(m.bn1, nn.SyncBatchNorm):
                m.bn1 = revert_sync_batchnorm(m.bn1)
            if isinstance(m.bn2, nn.SyncBatchNorm):
                m.bn2 = revert_sync_batchnorm(m.bn2)
            if isinstance(m.bn3, nn.SyncBatchNorm):
                m.bn3 = revert_sync_batchnorm(m.bn3)
            if m.downsample is not None:
                if isinstance(m.downsample[1], nn.SyncBatchNorm):
                    m.downsample[1] = revert_sync_batchnorm(m.downsample[1])
        if isinstance(m, nn.SyncBatchNorm):
            m = revert_sync_batchnorm(m)
        # if isinstance(m, models.yolo_resnext.Model):
        #    m.forward = m.forward  # assign forward (optional)

    model.eval()
    model.model[-1].export = True  # set Detect() layer export=True
    # y = model(img)  # dry run

    #model = model.half() # convert to FP16
    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
                          output_names=['output'])

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, f)
        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # Finish
    print('\nExport complete. Visualize with https://github.com/lutzroeder/netron.')