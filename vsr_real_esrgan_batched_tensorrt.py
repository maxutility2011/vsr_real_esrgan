import os
import sys
import time
import gc
import tensorrt as trt 
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

def read_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        img = cv2.imread(file_path)

        if img is not None:
            images.append(img)
        else:
            print(f"Failed to load image: {file_path}")
    
    return images

def preprocess_batch(images, target_shape):
    # Resize images and convert them to the required input format
    batch = []
    for img in images:
        img_resized = cv2.resize(img, target_shape)
        img_resized = img_resized.astype(np.float16) / 255.0  # Normalize input data to range [0, 1]
        img_resized = np.transpose(img_resized, (2, 0, 1))    # Change to NCHW format
        batch.append(img_resized)

    return np.stack(batch, axis=0)  # Return a batch (N, C, H, W)

print(trt.__version__)
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
with open(sys.argv[1], "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

if engine == None: 
    print("Failed to load engine!\n")
    exit(0)

upscale_factor = 4
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

    batch_size = shape[0]

    # Calculate the total buffer size to hold the input image: sizeof(float16) * batch_size * RGB_color_channels * image_height * image_width
    size = np.dtype(trt.nptype(dtype)).itemsize * batch_size 
    if is_input == True:
        print("\nReal_ESRGAN input tensor shape: ")
    else:
        print("\nReal_ESRGAN output tensor shape: ")

    print("batch_size | color channels | image height | image width")
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
    
    if is_input:
        input_height = shape[2]
        input_width = shape[3]
    else:
        output_height = shape[2]
        output_width = shape[3]

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

print("Input image folder: ", sys.argv[2])
#print("Input height ", input_height, "Input width ", input_width)
 
input_images = read_images_from_folder(sys.argv[2])
batch = preprocess_batch(input_images, (input_width, input_height)) # reorder height and width of target shape to match opencv2.resize()
context = engine.create_execution_context()

# Copy input data to the device buffer
print("overall shape: ", batch.shape)
cuda.memcpy_htod(inputs[0]['allocation'], batch.ravel())

inference_start_time_ms = int(time.time() * 1000)

# Execute inference
context.execute_v2(buffers)

inference_end_time_ms = int(time.time() * 1000)
print("Inference time taken: ", inference_end_time_ms - inference_start_time_ms, "ms")

# Copy predictions (output data) back to host from the device
host_output = np.zeros(batch_size * 3 * input_height * upscale_factor * input_width * upscale_factor, outputs[0]['dtype']) # Allocate output buffer
cuda.memcpy_dtoh(host_output, outputs[0]['allocation'])

# Post-process and save the output image
output_images = host_output.reshape((batch_size, 3, input_height * upscale_factor, input_width * upscale_factor))
output_images = np.transpose(output_images, (0, 2, 3, 1))  # Convert to HWC format for OpenCV image saving
output_images = np.clip(output_images * 255.0, 0, 255).astype(np.uint8) # De-normalize and convert to uint8 precision

output_folder = sys.argv[3]
os.makedirs(output_folder, exist_ok=True)

batch_number = int(sys.argv[4]) - 1
for idx, img in enumerate(output_images):
    image_number = idx + batch_number * batch_size
    print(image_number)
    url = output_folder + "/image_" + ("%04d" % image_number) + ".png"
    #cv2.imwrite(output_folder + "/output_" + str(idx) + ".png", img)
    cv2.imwrite(url, img)

del host_output
inputs[0]['allocation'].free()
outputs[0]['allocation'].free()

del context
del engine