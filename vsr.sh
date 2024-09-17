python3 vsr_real_esrgan_batched_tensorrt.py ./real_esrgan.engine ../samples/vertical/batch_1/ output/ 1
python3 vsr_real_esrgan_batched_tensorrt.py ./real_esrgan.engine ../samples/vertical/batch_2/ output/ 2
python3 vsr_real_esrgan_batched_tensorrt.py ./real_esrgan.engine ../samples/vertical/batch_3/ output/ 3
python3 vsr_real_esrgan_batched_tensorrt.py ./real_esrgan.engine ../samples/vertical/batch_4/ output/ 4
python3 vsr_real_esrgan_batched_tensorrt.py ./real_esrgan.engine ../samples/vertical/batch_5/ output/ 5

ffmpeg -r 15 -s 360x640 -i output/image_%4d.png -vcodec libx264 output.mp4