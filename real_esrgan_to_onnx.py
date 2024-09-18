import sys
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import argparse

parser = argparse.ArgumentParser(description='Convert Real ESRGAN pytorch model to ONNX format.')
parser.add_argument('--input_file', type=str, default='./RealESRGAN_x4plus.pth', help='The input Real ESRGAN model file')
parser.add_argument('--output_file', type=str, required=True, help='The output ONNX file')
parser.add_argument('--batch_size', type=int, default=1, help='The inference batch size')
parser.add_argument('--input_height', type=int, required=True, help='The input video height')
parser.add_argument('--input_width', type=int, required=True, help='The input video width')
args = parser.parse_args()

if args.input_height % 2 == 1 or args.input_width % 2 == 1:
    print("Input height or width is not divisable by 2")
    exit(0)

# x2 model: https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.1
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
model_path = args.input_file
loadnet = torch.load(model_path, map_location='cpu', weights_only=True)
model.load_state_dict(loadnet['params_ema'], strict=True)
model = model.half() # use float16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Export to ONNX
# Input shape: (batch_size, number_of_channels_RGB, height, width)
# sys.argv[1] gives the batch size
dummy_input = torch.randn(args.batch_size, 3, args.input_height, args.input_width, dtype=torch.float16, device=device)
torch.onnx.export(model, dummy_input, args.output_file, export_params=True, opset_version=14)