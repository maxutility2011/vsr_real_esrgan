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

6. Use ffmpeg to prepare the input image set for VSR inference,
   - Transcode the video to match the input tensor shape, e.g.,
   ```
   ffmpeg -i 10409938-uhd_2160_4096_25fps.mp4 -map v:0 -s:0 144x256 -c:v libx264 -preset slower -r 15 go_fishing_144x256.mp4
   ```

   Run the following script to break the transcoded video into a sequence of images, then copy the images to a number of batch folders. The batched images will be fed to Real_ESRGAN TensorRT engine for VSR inference.

   - 
   ```
   ./prep.sh [input_video] [frame_rate]
   ```
   For example, "*./prep.sh ../samples/go_fishing_144x256.mp4 15*". *frame_rate* should match the frame rate in the ffmpeg command (e.g., "*-r 15*"). The above command outputs a folder *images/* which contains *video_duration_sec x frame_rate_fps* number of images. In step 8, *vsr_real_esrgan_tensorrt.py* will read the input images from *images/* and run VSR inference on the images.

7. Run the following command
```
trtexec --onnx=real_esrgan.onnx --saveEngine=real_esrgan.engine
``` 
to convert the ONNX file to TensorRT engine. Every time you rebuild the ONNX file, you need to rebuild the engine file too.

8. Run the following script to perform VSR inference (image upscaling and video transcoding) to upscale multiple input images in a batch. Batched inference could be a bit faster than single image inference.
```
./vsr.sh
```
Specically, the script runs the following steps,
   - Load the Real_ESRGAN TensorRT engine file (as specified in *trt_engine*), and run inference to upscale    images under the *input_folder*, and save the upscaled images to the *output_folder*.
```   
python vsr_real_esrgan_tensorrt.py --trt_engine=./real_esrgan.engine --input_folder=./images --output_folder=./output
```
   - Re-encode the upscaled images into a video output file.
```
ffmpeg -hide_banner -loglevel error -r 15 -s 576x1024 -i output/image_%4d.png -vcodec libx264 -preset faster -crf 25 -y output.mp4
```
In the above command, the frame rate must match the values in step 6. The output video resolution must equals the output image resolution, i.e., x4 upscaled based on the input resolution. For example, if the input resolution is 256x144, the output resolution should be 576x1024. For other ffmpeg options, feel free to choose any value of your preference. 