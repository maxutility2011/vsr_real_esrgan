import sys
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

if len(sys.argv) != 2:
    # python vsr_real_esrgan_cuda.py real_esrgan.onnx
    print("python vsr_real_esrgan_cuda.py [tensorrt_engine_output_path]") 
    sys.exit(0)

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
model_path = './RealESRGAN_x4plus.pth'
loadnet = torch.load(model_path, map_location='cpu', weights_only=True)
model.load_state_dict(loadnet['params_ema'], strict=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Export to ONNX
# Input shape: (batch_size, number_of_channels_RGB, height, width)
dummy_input = torch.randn(1, 3, 128, 128, dtype=torch.float32, device=device)
#dummy_input = dummy_input.to(device)
torch.onnx.export(model, dummy_input, sys.argv[1], export_params=True, opset_version=11)