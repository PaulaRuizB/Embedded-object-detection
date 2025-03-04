import torch
import torch.nn as nn
import argparse

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='weights path')  # from yolov5/models/
    parser.add_argument('--save_path', type=str, default='', help='path to save onnx model')

    args = parser.parse_args()
    weights = args.weights
    save_path = args.save_path

    batch_size = 1
    dummy_input = torch.randn(batch_size, 128)

    model_torch = model_torch()

    model_torch.load_state_dict(torch.load(weights))

    torch.onnx.export(model_torch, dummy_input, save_path,
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})