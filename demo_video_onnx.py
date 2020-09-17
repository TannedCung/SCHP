import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import time
import onnx

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import networks
from utils.transforms import transform_logits
from utils import change_dict
from utils import gray2color
from datasets.simple_extractor_video import SimpleVideo
import onnxruntime

dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
    },
    'cihp': {
        'input_size' : [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper clothes', 'Dress', 'Coat', 'Socks', 'Pants','Scarf', 'Torsoskin',
                  'Skirt', 'Gloves', 'Face', 'Right arm', 'Left arm', 'Right leg', 'Left leg',' Right shoe', 'Left shoe']
    }
}

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    parser.add_argument("--dataset", type=str, default='cihp', choices=['lip', 'atr', 'pascal', 'cihp'])
    parser.add_argument("--model-restore", type=str, default='./checkpoints/exp_schp_multi_cihp_global.pth', help="restore pretrained model parameters.")
    parser.add_argument("--gpu", type=str, default='None', help="choose gpu device.")
    parser.add_argument("--input-dir", type=str, default='./test/input', help="path of input image folder.")
    parser.add_argument("--output-dir", type=str, default='./test/output_global', help="path of output image folder.")
    parser.add_argument("--logits", action='store_true', default=False, help="whether to save the logits.")

    return parser.parse_args()

def main():
    args = get_arguments()

    # gpus = [int(i) for i in args.gpu.split(',')]
    # assert len(gpus) == 1
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    num_classes = dataset_settings[args.dataset]['num_classes']
    input_size = dataset_settings[args.dataset]['input_size']
    label = dataset_settings[args.dataset]['label']
    # Model in torch init
    model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)

    state_dict = torch.load(args.model_restore)     #['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    secretary = change_dict.dictModify(new_dict=new_state_dict, old_dict=state_dict)
    new_state_dict = secretary.arange()
    model.load_state_dict(new_state_dict)

    model.eval()

    print("Evaluating total class number {} with {}".format(num_classes, label))

    sess = onnxruntime.InferenceSession("pretrain_model/sim_local.onnx")  


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])
    transformer = SimpleVideo(transforms=transform, input_size=[473,473])

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    color_man = gray2color.makeColor(num_classes)
    VIDEO_PATH = "input.mp4"
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 24, (frame_width*3,frame_height))
    with torch.no_grad():
        start = time.time()
        count = 0
        ret = True
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                # w, h, _ = frame.shape
                # c, s, w, h = cal_chwsr(0, 0, w, h)
                # frame = torch.tensor(frame)
                print("frame: {}".format(frame.shape))
                frame, meta = transformer.get_item(frame)
                c = meta['center']
                s = meta['scale']
                w = meta['width']
                h = meta['height']

                # out_put = model(frame)
                # input_name = sess.get_inputs()[0].name
                # print("input name:", input_name)
                # input_shape = sess.get_inputs()[0].shape
                # print("input shape:", input_shape)
                # input_type = sess.get_inputs()[0].type
                # print("input type:", input_type)
                input_name = sess.get_inputs()[0].name
                # output_name = sess.get_outputs()[0].name
                output_name = list(['output', '1250', '1245'])


                pre_output = sess.run(output_name, input_feed={sess.get_inputs()[0].name: np.array(frame)})
                output = [[pre_output[0], pre_output[1]], [pre_output[2]]]
                output_t = model(frame)
                # Post-process for output from onnx model
                upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
                upsample_output = upsample(torch.tensor(output[0][-1][0]).unsqueeze(0))
                upsample_output = upsample_output.squeeze()
                upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

                logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
                parsing_result = np.argmax(logits_result, axis=2)
                output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
                out_img_o = np.array(output_img)
                output_img_o = color_man.G2C(out_img_o)
                # Post-process for torch model
                upsample_output = upsample(torch.tensor(output_t[0][-1][0]).unsqueeze(0))
                upsample_output = upsample_output.squeeze()
                upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

                logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
                parsing_result = np.argmax(logits_result, axis=2)
                output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
                out_t_min = np.array(output_img)
                out_img_t = np.array(output_img)
                output_img_t = color_man.G2C(out_img_t)

                # final = cv2.hconcat(output_img_t, out_img_t-out_img_o)
                final = cv2.hconcat([output_img_t, cv2.cvtColor(out_img_t-out_img_o, cv2.COLOR_GRAY2RGB)*100, output_img_o])

                out.write(np.array(final))
                cv2.imshow("Tanned", np.array(final))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if args.logits:
                    logits_result_path = os.path.join(args.output_dir, img_name[:-4] + '.npy')
                    np.save(logits_result_path, logits_result)
                count += 1
            else:
                break
        end = time.time()
        cap.release()
        out.release()
        print("Processed {} images using {:.5} seconds, average each image took {:.5} seconds".format(count, end-start, (end-start)/(count+0.1)))
    return

if __name__ == '__main__':
    main()