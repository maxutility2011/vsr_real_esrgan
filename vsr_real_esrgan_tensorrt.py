import sys
import time
import gc
import tensorrt as trt 
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

print(trt.__version__)
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
with open(sys.argv[1], "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

if engine == None: 
    print("Failed to load engine!\n")
    exit(0)

inputs = []
outputs = []
buffers = []
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    is_input = False
    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        is_input = True

    dtype = engine.get_tensor_dtype(name)
    shape = engine.get_tensor_shape(name)

    if is_input:
        batch_size = shape[0]

    # Calculate the total buffer size to hold the input image: sizeof_float32 * batch_size * RGB_color_channels * image_height * image_width
    size = np.dtype(trt.nptype(dtype)).itemsize # bytes of float32
    print("\nReal_ESRGAN input tensor shape: ")
    print("batch_size | color channels | image height | image width ")
    for d in shape:
        print(d, " | ", end=" ")
        size *= d

    print("\n")

    print("Tensor name: ", name)
    print("Is input? ", is_input)
    print("dtype: ", dtype)
    print("shape: ", shape)
    print("batch_size: ", batch_size)
    print("buffer size: ", size)
    print("\n")
    
    buffer = cuda.mem_alloc(size)
    binding = {
        "index": i,
        "name": name,
        "dtype": np.dtype(trt.nptype(dtype)),
        "shape": list(shape),
        "allocation": buffer,
        "size": size
    }

    buffers.append(buffer)
    if is_input:
        inputs.append(binding)
    else:
        outputs.append(binding)

assert batch_size > 0
assert len(inputs) > 0
assert len(outputs) > 0
assert len(buffers) > 0

print("Input image: ", sys.argv[2])
input_image = cv2.imread(sys.argv[2])
input_image = cv2.resize(input_image, (128, 128)) # If the input image resolution is not 128 x 128, resize it to match the expected resolution.
input_image = input_image.astype(np.float32) / 255.0 # Change data precision from uint8 to float32 for high-precision inference, and normalize the image array by dividing 255. Pixel data values in 8 bits images are between 0-255 and Real-ESRGAN model uses normalized image data for training.

input_image = np.transpose(input_image, (2, 0, 1)) # Transpose to CHW format (as used by Real-ESRGAN)
input_image = np.expand_dims(input_image, axis=0) # Add batch dimension

context = engine.create_execution_context()

# Copy input data to the device buffer
cuda.memcpy_htod(inputs[0]['allocation'], input_image.ravel())

inference_start_time_ms = int(time.time() * 1000)

# Execute inference
context.execute_v2(buffers)

inference_end_time_ms = int(time.time() * 1000)
print("Inference time taken: ", inference_end_time_ms - inference_start_time_ms, "ms")

# Copy predictions (output data) back to host from the device
host_output = np.zeros(3 * 512 * 512, outputs[0]['dtype'])
cuda.memcpy_dtoh(host_output, outputs[0]['allocation'])

# Post-process and save the output image
output_image = np.squeeze(host_output)  # Remove batch dimension
output_image = output_image.reshape((3, 512, 512)) # Format the raw tensorrt output in CHW format
output_image = np.transpose(output_image, (1, 2, 0))  # Convert to HWC format for OpenCV image saving
output_image = np.clip(output_image * 255.0, 0, 255).astype(np.uint8) # De-normalize and convert to uint8 precision

cv2.imwrite(sys.argv[3], output_image)
cv2.imshow("output image", output_image)
print("Press any key to close the image...")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Output image: ", sys.argv[3])

del host_output
inputs[0]['allocation'].free()
outputs[0]['allocation'].free()

del context
del engine