mkdir images 2>/dev/null
rm images/* 2>/dev/null 
echo "Convert input video to images. Look for images under ./images/"
ffmpeg -i $1 -vf fps=$2 images/image_%4d.png
echo "Creating image batches for VSR inference"
rm input_batches/* 2>/dev/null
python3 create_batches.py --input_folder=images/ --output_folder=input_batches --batch_size=$3