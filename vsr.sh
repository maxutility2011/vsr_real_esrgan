echo "Upscaling video with super resolution"
python3 vsr_real_esrgan_batched_tensorrt.py ./real_esrgan.engine 45 ../samples/vertical/batch_ output/

echo "FFmpeg generating output video"
ffmpeg -hide_banner -loglevel error -r 15 -s 360x640 -i output/image_%4d.png -vcodec libx264 -y output.mp4