


# new_state = ["context_encoding.stages.0.2.negative_slope.weight", "context_encoding.stages.0.2.negative_slope.bias", "context_encoding.stages.0.2.negative_slope.running_mean", "context_encoding.stages.0.2.negative_slope.running_var", "context_encoding.stages.1.2.negative_slope.weight", "context_encoding.stages.1.2.negative_slope.bias", "context_encoding.stages.1.2.negative_slope.running_mean", "context_encoding.stages.1.2.negative_slope.running_var", "context_encoding.stages.2.2.negative_slope.weight", "context_encoding.stages.2.2.negative_slope.bias", "context_encoding.stages.2.2.negative_slope.running_mean", "context_encoding.stages.2.2.negative_slope.running_var", "context_encoding.stages.3.2.negative_slope.weight", "context_encoding.stages.3.2.negative_slope.bias", "context_encoding.stages.3.2.negative_slope.running_mean", "context_encoding.stages.3.2.negative_slope.running_var", "context_encoding.bottleneck.1.negative_slope.weight", "context_encoding.bottleneck.1.negative_slope.bias", "context_encoding.bottleneck.1.negative_slope.running_mean", "context_encoding.bottleneck.1.negative_slope.running_var", "edge.conv1.1.negative_slope.weight", "edge.conv1.1.negative_slope.bias", "edge.conv1.1.negative_slope.running_mean", "edge.conv1.1.negative_slope.running_var", "edge.conv2.1.negative_slope.weight", "edge.conv2.1.negative_slope.bias", "edge.conv2.1.negative_slope.running_mean", "edge.conv2.1.negative_slope.running_var", "edge.conv3.1.negative_slope.weight", "edge.conv3.1.negative_slope.bias", "edge.conv3.1.negative_slope.running_mean", "edge.conv3.1.negative_slope.running_var", "decoder.conv1.1.negative_slope.weight", "decoder.conv1.1.negative_slope.bias", "decoder.conv1.1.negative_slope.running_mean", "decoder.conv1.1.negative_slope.running_var", "decoder.conv2.1.negative_slope.weight", "decoder.conv2.1.negative_slope.bias", "decoder.conv2.1.negative_slope.running_mean", "decoder.conv2.1.negative_slope.running_var", "decoder.conv3.1.negative_slope.weight", "decoder.conv3.1.negative_slope.bias", "decoder.conv3.1.negative_slope.running_mean", "decoder.conv3.1.negative_slope.running_var", "decoder.conv3.3.negative_slope.weight", "decoder.conv3.3.negative_slope.bias", "decoder.conv3.3.negative_slope.running_mean", "decoder.conv3.3.negative_slope.running_var", "fushion.1.negative_slope.weight", "fushion.1.negative_slope.bias", "fushion.1.negative_slope.running_mean", "fushion.1.negative_slope.running_var"]
# old_state = ["context_encoding.stages.0.2.weight", "context_encoding.stages.0.2.bias", "context_encoding.stages.0.2.running_mean", "context_encoding.stages.0.2.running_var", "context_encoding.stages.1.2.weight", "context_encoding.stages.1.2.bias", "context_encoding.stages.1.2.running_mean", "context_encoding.stages.1.2.running_var", "context_encoding.stages.2.2.weight", "context_encoding.stages.2.2.bias", "context_encoding.stages.2.2.running_mean", "context_encoding.stages.2.2.running_var", "context_encoding.stages.3.2.weight", "context_encoding.stages.3.2.bias", "context_encoding.stages.3.2.running_mean", "context_encoding.stages.3.2.running_var", "context_encoding.bottleneck.1.weight", "context_encoding.bottleneck.1.bias", "context_encoding.bottleneck.1.running_mean", "context_encoding.bottleneck.1.running_var", "edge.conv1.1.weight", "edge.conv1.1.bias", "edge.conv1.1.running_mean", "edge.conv1.1.running_var", "edge.conv2.1.weight", "edge.conv2.1.bias", "edge.conv2.1.running_mean", "edge.conv2.1.running_var", "edge.conv3.1.weight", "edge.conv3.1.bias", "edge.conv3.1.running_mean", "edge.conv3.1.running_var", "decoder.conv1.1.weight", "decoder.conv1.1.bias", "decoder.conv1.1.running_mean", "decoder.conv1.1.running_var", "decoder.conv2.1.weight", "decoder.conv2.1.bias", "decoder.conv2.1.running_mean", "decoder.conv2.1.running_var", "decoder.conv3.1.weight", "decoder.conv3.1.bias", "decoder.conv3.1.running_mean", "decoder.conv3.1.running_var", "decoder.conv3.3.weight", "decoder.conv3.3.bias", "decoder.conv3.3.running_mean", "decoder.conv3.3.running_var", "fushion.1.weight", "fushion.1.bias", "fushion.1.running_mean", "fushion.1.running_var"]

# print(len(new_state))
# print(len(old_state))

# def check(what_2_check):
#     if what_2_check in old_state:
#         return 1
#     else:
#         return 0

# print(check ("context_encoding.stages.0.2.weight"))

"""import torch

import torch.nn as nn
import torch.nn.functional as F


class al_ABN(nn.Module):
    def __init__(self, size):
        super(al_ABN, self).__init__()
        self.bn = nn.BatchNorm2d(size)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.bn(x)
        out = self.act(out)
        return out

def conv(inp, out, s=1):
    return nn.Conv2d(inp, out, 5)

# model = nn.Sequential(
#           conv(1, 20),
#           al_ABN(20),
#           nn.ReLU(),
#           nn.Conv2d(20,64,5),
#           nn.ReLU()
#         )

# print(model)
features = 5
out_features = 10
sizes = [i for i in range(5)]
inputs = torch.randn(1,55,3,3)

bottleneck1 = nn.Sequential(
    nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, dilation=1,
                bias=False),
    nn.BatchNorm2d(out_features),
    nn.LeakyReLU(),
    nn.AvgPool2d(3))

bottleneck2 = nn.Sequential(
    nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, dilation=1,
                bias=False),
    al_ABN(out_features),
    nn.AvgPool2d(3))

print(bottleneck1(inputs) - bottleneck2(inputs))
# print(bottleneck)"""

"""from pytorch_model import preprocess_image, postprocess
import torch
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt


ONNX_FILE_PATH = "resnet50.onnx"
# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()


def build_engine(onnx_file_path):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # allow TensorRT to use up to 1GB of GPU memory for tactic selection
    builder.max_workspace_size = 1 << 30
    # we have only one image in batch
    builder.max_batch_size = 1
    # use FP16 mode if possible
    if builder.platform_has_fast_fp16:
        builder.fp16_mode = True

    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')

    # generate TensorRT engine optimized for the target platform
    print('Building an engine...')
    engine = builder.build_cuda_engine(network)
    context = engine.create_execution_context()
    print("Completed creating Engine")
    return engine, context


def main():
    # initialize TensorRT engine and parse ONNX model
    engine, context = build_engine(ONNX_FILE_PATH)
    # get sizes of input and output and allocate memory required for input data and for output data
    for binding in engine:
        if engine.binding_is_input(binding):  # we expect only one input
            input_shape = engine.get_binding_shape(binding)
            input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
            device_input = cuda.mem_alloc(input_size)
        else:  # and one output
            output_shape = engine.get_binding_shape(binding)
            # create page-locked memory buffers (i.e. won't be swapped to disk)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)

    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()


    # preprocess input data
    host_input = np.array(preprocess_image("turkish_coffee.jpg").numpy(), dtype=np.float32, order='C')
    cuda.memcpy_htod_async(device_input, host_input, stream)

    # run inference
    context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    stream.synchronize()

    # postprocess results
    output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[0])
    postprocess(output_data)


# if __name__ == '__main__':
#     main()"""
# import time
# import onnxsim
# onnx_file_path = "pretrain_model/local.onnx"
# with open(onnx_file_path, 'rb') as model:
#         print('Beginning ONNX file parsing')
#         start = time.time()

#         a = model.read()

# print(type(a))
# print(len(a))

# print("Ended in {:.4}".format(time.time()- start))

import onnx
import onnxsim

ONNX_FILE_PATH = "pretrain_model/local.onnx"
ONNX_SIM_FILE_PATH = "pretrain_model/sim_local.onnx"
print("start simplifying onnx")
onnx_model = onnx.load(ONNX_FILE_PATH)
target_version = 3
ir_version     = 5
# converted_model = version_converter.convert_version(onnx_model , target_version)
# onnx_model.ir_version = ir_version
sim_model, check = onnxsim.simplify(onnx_model)
onnx.save(sim_model, ONNX_SIM_FILE_PATH)