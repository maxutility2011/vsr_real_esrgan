import os
import sys
import time
import gc
import tensorrt as trt 
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import logging
import argparse

def read_images_from_folder(folder_path, batch_size):
    images = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        img = cv2.imread(file_path)

        if img is not None:
            images.append(img)
        else:
            logging.error("Failed to load image: %s", file_path)
    
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

parser = argparse.ArgumentParser(description='Set log level dynamically from command-line.')
parser.add_argument('--loglevel', type=str, default='INFO', help='Set the log level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)')
parser.add_argument('--trt_engine', type=str, default='./real_esrgan.engine', help='Path to the vsr inference engine file')
parser.add_argument('--input_folder', type=str, help='The top level input folder that contains all the batch subfolders')
parser.add_argument('--output_folder', type=str, help='The output folder')

args = parser.parse_args()

logger = logging.getLogger()
logging.basicConfig(level=args.loglevel, format='%(message)s')

#print("TensorRT version", trt.__version__)
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
with open(args.trt_engine, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

if engine == None: 
    logger.error("Failed to load engine!\n")
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
        logger.debug("Input tensor shape:")
    else:
        logger.debug("Output tensor shape:")

    logger.debug("(batch_size, color channels, image height, image width): (%d %d %d %d)", shape[0], shape[1], shape[2], shape[3])
    logger.debug("\n")

    for d in shape:
        size *= d

    logger.debug("Tensor name: %s", name)
    logger.debug("Is input? %s", is_input)
    logger.debug("dtype: %s", dtype)
    logger.debug("batch_size: %d", batch_size)
    logger.debug("buffer size: %d", size)
    logger.debug("\n")
    
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

logger.debug("Input image folder: %s", args.input_folder)
logger.debug("Input height: %d, Input width: %d", input_height, input_width)
 
context = engine.create_execution_context()

host_output_allocated = False
batch_idx = 0
for dirpath, dirnames, filenames in os.walk(args.input_folder, onerror=lambda e: print(e)):
    if dirpath == args.input_folder:
        continue

    input_images = read_images_from_folder(dirpath, batch_size)
    if len(input_images) == 0:
        logger.error("Failed to read images from %s", dirpath)
        break

    if len(input_images) > batch_size:
        logging.error("Batch folder %s contains %d files, more than %d (batch size) images", dirpath, len(input_images), batch_size)
        break

    batch_idx += 1
    batch = preprocess_batch(input_images, (input_width, input_height)) # reorder height and width of target shape to match opencv2.resize()

    # Copy input data to the device buffer
    logger.debug("Overall input shape: (%d %d %d %d)", batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3])
    cuda.memcpy_htod(inputs[0]['allocation'], batch.ravel())

    inference_start_time_ms = int(time.time() * 1000)

    # Execute inference
    context.execute_v2(buffers)

    inference_end_time_ms = int(time.time() * 1000)
    print("Batch", batch_idx, "inference time taken: ", inference_end_time_ms - inference_start_time_ms, "ms")

    # Copy predictions (output data) back to host from the device
    host_output = np.zeros(batch_size * 3 * input_height * upscale_factor * input_width * upscale_factor, outputs[0]['dtype']) # Allocate output buffer
    host_output_allocated = True
    cuda.memcpy_dtoh(host_output, outputs[0]['allocation'])

    # Post-process and save the output image
    output_images = host_output.reshape((batch_size, 3, input_height * upscale_factor, input_width * upscale_factor))
    output_images = np.transpose(output_images, (0, 2, 3, 1))  # Convert to HWC format for OpenCV image saving
    output_images = np.clip(output_images * 255.0, 0, 255).astype(np.uint8) # De-normalize and convert to uint8 precision

    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    for img_idx, img in enumerate(output_images):
        image_number = img_idx - 1 + batch_idx * batch_size
        url = output_folder + "/image_" + ("%04d" % image_number) + ".png"
        cv2.imwrite(url, img)

# Deallocate host memory
if host_output_allocated == True:
    del host_output

# Deallocate device memory
inputs[0]['allocation'].free()
outputs[0]['allocation'].free()

del context
del engine