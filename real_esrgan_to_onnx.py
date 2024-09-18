import sys
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

if len(sys.argv) != 3:
    # python vsr_real_esrgan_cuda.py [batch_size] real_esrgan.onnx
    print("python vsr_real_esrgan_cuda.py [tensorrt_engine_output_path]") 
    sys.exit(0)

# x2 model: https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.1
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
model_path = './RealESRGAN_x4plus.pth'
loadnet = torch.load(model_path, map_location='cpu', weights_only=True)
model.load_state_dict(loadnet['params_ema'], strict=True)
model = model.half() # use float16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Export to ONNX
# Input shape: (batch_size, number_of_channels_RGB, height, width)
# sys.argv[1] gives the batch size
dummy_input = torch.randn(int(sys.argv[1]), 3, 320, 180, dtype=torch.float16, device=device)
#dummy_input = dummy_input.to(device)
torch.onnx.export(model, dummy_input, sys.argv[2], export_params=True, opset_version=14)