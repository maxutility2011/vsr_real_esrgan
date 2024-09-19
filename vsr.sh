echo "Removing previous output files..."
rm -rf output/*

echo "Upscaling video with super resolution..."
python3 vsr_real_esrgan_batched_tensorrt.py --loglevel=DEBUG --trt_engine=./real_esrgan.engine --input_folder=./input_batches --output_folder=output/

echo "Generating output video..."
ffmpeg -hide_banner -loglevel error -r 15 -s 576x1024 -i output/image_%4d.png -vcodec libx264 -preset faster -crf 25 -y output.mp4