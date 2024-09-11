# vsr_real_esrgan
First, pip install -r requirements.txt to install the dependencies.

If you wish to run inference without TensorRT, use this command "*python vsr_real_esrgan_cuda.py [input_image_path] [output_image_path]*".

If you wish to run inference with TensorRT on GPU, follow the steps below,
1. Install TensorRT. Please refer to https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html, choose one of the installation method (e.g., from .deb, .rpm, .tar, .zip file).
2. If you installed TensorRT from a TAR file, add the TensorRT bin folder to your path, e.g., add "*export PATH=/mnt/c/Users/pirat/Downloads/TensorRT-10.3.0.26/bin/:$PATH*" to bashrc and run bashrc again.
3. Run "*python real_esrgan_to_onnx.py [onnx_file_output_path]*" to convert the real_esrgan model to the intermediate ONNX format.
4. Run "*trtexec --onnx=real_esrgan.onnx --saveEngine=real_esrgan.engine*" to convert the ONNX file to TensorRT engine.
5. Install pycuda, "*pip install pycuda*" (**pycuda** should have been included in requirements.txt, so installing pycuda may not be needed). If you run into the following error, "fatal error: cuda.h: No such file or directory", configure the paths to your cuda installation in bashrc then run bashrc again, e.g., 
"
export CPATH=/usr/local/cuda/include:$CPATH
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
"

6. The inference application (*vsr_real_esrgan_tensorrt.py*) use tensorrt to run inference (i.e., scale-up images) that is why we include tensorrt in requirements.txt. Also, in step 4, we use **trtexec** to convert the ONNX file to TensorRT engine file which is an utility program included in the TensorRT TAR file. The two tensorrt versions (the one installed by **pip** and the one included in the TAR file) must match, otherwise you will see runtime errors. For my setup, I installed tensorrt version 10.1.0.27 in both requirements.txt and the TAR file.