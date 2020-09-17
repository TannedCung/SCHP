# from pytorch_model import preprocess_image, postprocess
import torch
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import pickle

from datasets import postprocess_output
from datasets import simple_extractor_video as prepocess_input
import torchvision.transforms as transforms
import cv2
import time


ONNX_FILE_PATH = "pretrain_model/resnet50.onnx"
# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_file_path):
    # initialize TensorRT engine and parse ONNX model
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    # print("network: {}".format(network))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    # print("parser: {}".format(parser))

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
        reader = model.read()
        # print(len(reader))
        ret = parser.parse(reader)
        if not ret:
            print('ERROR: Failed to parse the ONNX file.')
            print ('-------------------------------------')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            print ('-------------------------------------')

    # generate TensorRT engine optimized for the target platform
    print('Building an engine...')
    network.mark_output(network.get_layer(network.num_layers - 1).get_output(0))
    engine = builder.build_cuda_engine(network)
    # print("engine: {}".format(engine))
    context = engine.create_execution_context()
    print("Completed creating Engine")
    return engine

def build_context(engine):
    context = engine.create_execution_context()
    return context

def main():
    # initialize TensorRT engine and parse ONNX model
    engine = build_engine(ONNX_FILE_PATH)
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
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])
    frame = cv2.imread('test/input/000001_0.jpg')
    pre = prepocess_input.SimpleVideo(transforms=transform)
    frame, meta = pre.get_item(frame)
    c = meta['center']
    s = meta['scale']
    w = meta['width']
    h = meta['height']
    host_input = np.array(np.array(frame), dtype=np.float32, order='C')
    cuda.memcpy_htod_async(device_input, host_input, stream)

    start = time.time()
    # run inference
    context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    stream.synchronize()

    # postprocess results
    post = postprocess_output.Poster(20)
    output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[0])
    ouput_image = postprocess_output(output_data)
    cv2.imwrite("output.jpg", output_image)
    end = time.time()
    print("Executed 1 image in {:.5}".format(start-end))

if __name__ == '__main__':
    main()
