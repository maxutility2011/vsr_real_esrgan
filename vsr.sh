echo "Removing previous output files..."
rm -rf $2 2>/dev/null
mkdir $2 2>/dev/null

echo "Upscaling video with super resolution..."
python3 vsr_real_esrgan_batched_tensorrt.py --loglevel=DEBUG --trt_engine=./real_esrgan.engine --input_folder=$1 --output_folder=$2

echo "Generating output video..."
ffmpeg -hide_banner -loglevel error -r 15 -s 720x1280 -i $2/image_%4d.bmp -vcodec libx264 -preset ultrafast -crf 30 -y output.mp4