# vsr_real_esrgan
1. pip install -r requirements.txt
2. To run inference without TensorRT, run "python vsr_real_esrgan_cuda.py [input_image_path] [output_image_path]".
3. To run inference with TensorRT on GPU, follow the steps below,
4. Install TensorRT. Please refer to https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html
5. If you installed TensorRT from TAR files, add the TensorRT bin folder to your path, e.g., add "export PATH=/mnt/c/Users/pirat/Downloads/TensorRT-10.3.0.26/bin/:$PATH" to bashrc and run bashrc again.
6. Run "python real_esrgan_to_onnx.py [onnx_file_output_path]" to convert real_esrgan model to ONNX format.
7. Run "trtexec --onnx=real_esrgan.onnx --saveEngine=real_esrgan.engine" to convert the ONNX file to TensorRT engine.

8. Install pycuda, "pip install pycuda". If you run into the following error, "fatal error: cuda.h: No such file or directory", configure the paths to your cuda installation in bashrc then run bashrc again, e.g., 
"
export CPATH=/usr/local/cuda/include:$CPATH
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
"