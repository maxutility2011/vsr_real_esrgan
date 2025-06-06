import sys
import torch
import time
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

if len(sys.argv) != 3:
    print("python vsr_real_esrgan_cuda.py [input_image_path] [output_image_path]")
    sys.exit(0)

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
model_path = './RealESRGAN_x4plus.pth'
loadnet = torch.load(model_path, map_location='cpu', weights_only=True)
model.load_state_dict(loadnet['params_ema'], strict=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).eval()
upsampler = RealESRGANer(scale=2,
                            model_path=model_path, 
                            model=model, 
                            half=False, 
                            gpu_id=0)

import cv2
print("Input image: ", sys.argv[1])
image = cv2.imread(sys.argv[1])

inference_start_time_ms = int(time.time() * 1000)
output_image, _ = upsampler.enhance(image)
inference_end_time_ms = int(time.time() * 1000)
print("Inference time taken: ", inference_end_time_ms - inference_start_time_ms, "ms")

cv2.imwrite(sys.argv[2], output_image)