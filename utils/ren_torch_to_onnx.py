from utils import resnet_big
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='weights path')  # from yolov5/models/
    parser.add_argument('--save_path', type=str, default='', help='path to save onnx model')

    args = parser.parse_args()
    weights = args.weights
    save_path = args.save_path

    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    state_dict = torch.load(weights)['model']
    model = resnet_big.SupConResNet()
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v

    torch.onnx.export(model, dummy_input, save_path,
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})