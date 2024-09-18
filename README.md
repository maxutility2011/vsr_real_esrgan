# vsr_real_esrgan
First, run 
```
pip install -r requirements.txt
```
to install the dependencies.

If you wish to run inference without TensorRT, use the following command. 
```
python vsr_real_esrgan_cuda.py [input_image_path] [output_image_path]
```

If you wish to run inference with TensorRT on GPU, follow the steps below,
1. Install TensorRT. Please refer to https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html and choose one of the installation methods (e.g., from .deb, .rpm, .tar, .zip file).

2. If you choose to install TensorRT from a TAR file, add the TensorRT bin/ folder to your path. For example, add the following line,
```
export PATH=/home/bozhang/TensorRT-10.3.0.26/bin/:$PATH
```
to bash profile.

3. Install pycuda, "*pip install pycuda*" (**pycuda** should have been included in [requirements.txt](requirements.txt), so installing pycuda may not be needed). If you run into the following error, "*fatal error: cuda.h: No such file or directory*", configure the paths to your cuda installation in bashrc then run bashrc again, e.g., 
```
export CPATH=/usr/local/cuda/include:$CPATH
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

4. The inference application (e.g., [vsr_real_esrgan_tensorrt.py](vsr_real_esrgan_tensorrt.py)) uses TensorRT to generate upscaled images, so we need to include tensorrt in [requirements.txt](requirements.txt). Also, in step 4, we use **trtexec** (an utility program included in the TensorRT TAR file) to convert the ONNX file to TensorRT engine file. The two tensorrt versions (the one given in [requirements.txt](requirements.txt) and the one included in the TAR file) must match, otherwise you will see runtime errors. For my setup, I use TensorRT version 10.1.0.27 in both requirements.txt and the TAR file.

5. Run the following command
```
python real_esrgan_to_onnx.py --output_file=./real_esrgan.onnx --batch_size=3 --input_height=360 --input_width=640
```
to convert the pre-trained real_esrgan model to the intermediate ONNX format (the enclosed [RealESRGAN_x4plus.pth](RealESRGAN_x4plus.pth)). You can also download the latest model file from https://github.com/xinntao/Real-ESRGAN/releases. [batch_size] provides the batch size for tensorrt batched inference. If you only want to generate 1 image output at a time, set this to 1, otherwise set this to a value that will not exhaust your GPU memory. You may need to run this command multiple times to experient and find out the highest value of *batch_size* that would allow the best VSR inference throughput.

6. Use ffmpeg to prepare the input video,
   - Transcode the video to match the input tensor shape, e.g.,
   ```
   ffmpeg -i 10409938-uhd_2160_4096_25fps.mp4 -map v:0 -s:0 144x256 -c:v libx264 -preset slower -r 15 go_fishing_144x256.mp4
   ```
   Run the following command to break the transcoded video into a sequence of images, then copy the images to a number of batch folders.
   - 
   ```
   ./prep.sh [input_video] [frame_rate] [batch_size]
   ```
   For example, "*./prep.sh ../samples/go_fishing_144x256.mp4 15 3*". *frame_rate* should match the frame rate in the ffmpeg command (e.g., "*-r 15*"), *batch_size* should match the value you found out in step 5.

7. Run the following command
```
trtexec --onnx=real_esrgan.onnx --saveEngine=real_esrgan.engine
``` 
to convert the ONNX file to TensorRT engine. Every time you rebuild the ONNX file, you need to rebuild the engine file too.

8. Run the following command
```   
python vsr_real_esrgan_tensorrt.py --trt_engine=./real_esrgan.engine --input_folder=./input --output_folder=./output
```
to upscale multiple input images in a batch. Batched processing could be a bit faster. Batch number starts from 1.