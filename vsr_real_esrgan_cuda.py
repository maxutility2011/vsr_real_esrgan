import sys
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
model_path = './RealESRGAN_x4plus.pth'
loadnet = torch.load(model_path, map_location='cpu')
model.load_state_dict(loadnet['params_ema'], strict=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).eval()
upsampler = RealESRGANer(scale=4, 
                            model_path=model_path, 
                            model=model, 
                            half=False, 
                            gpu_id=0)

import cv2
print("Input image: ", sys.argv[1])
image = cv2.imread(sys.argv[1])
output_image, _ = upsampler.enhance(image)
#cv2.imwrite('_' + sys.argv[1], output_image)
cv2.imwrite('./output.png', output_image)