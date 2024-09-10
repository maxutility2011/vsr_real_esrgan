import sys
import gc
import tensorrt as trt 
import pycuda.driver as cuda
import pycuda.autoinit
#from cuda import cuda, cudart
import numpy as np
#from cuda import cudart

print(trt.__version__)
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
with open(sys.argv[1], "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())


#input_shape = (1, 3, 128, 128)
#output_shape = (1, 3, 512, 512)
#input_buffer = cuda.mem_alloc(np.prod(input_shape) * np.float32().itemsize)
#output_buffer = cuda.mem_alloc(np.prod(output_shape) * np.float32().itemsize)
#stream = cuda.Stream()

#if engine != None:
#    inputs, outputs, bindings, stream = allocate_buffers(engine)
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

    size = np.dtype(trt.nptype(dtype)).itemsize
    print("size: ", size)
    for s in shape:
        print("s: ", s)
        size *= s

    print("Tensor name: ", name)
    print("Is input? ", is_input)
    print("dtype: ", dtype)
    print("shape: ", shape)
    print("batch_size: ", batch_size)
    print("Buffer size: ", size)
    print("......................")
    
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

import cv2
print("Input image: ", sys.argv[2])
input_image = cv2.imread(sys.argv[2])
input_image = cv2.resize(input_image, (128, 128)) # BoZhang: If the input image resolution is not 128 x 128, resize it to match the expected resolution.
input_image = input_image.astype(np.float32) / 255.0 # BoZhang: without this line, the output image will be all black

input_image = np.transpose(input_image, (2, 0, 1))
input_image = np.expand_dims(input_image, axis=0)

context = engine.create_execution_context()

# Copy input data to the device buffer
#np.copyto(inputs[0]['allocation'], input_image.ravel())
#cuda.memcpy_htod(inputs[0]['device'], inputs[0]['allocation'])
cuda.memcpy_htod(inputs[0]['allocation'], input_image.ravel())

# Execute inference
context.execute_v2(buffers)

# Copy predictions (output data) back to host from the device
#host_output = np.empty(outputs[0]['shape'], dtype=np.float32)
host_output = np.zeros(3 * 512 * 512, outputs[0]['dtype'])
print("host_output size: ", outputs[0]['size'])

cuda.memcpy_dtoh(host_output, outputs[0]['allocation'])
#print(host_output.reshape((shape[2], shape[3], shape[1])))

# Post-process and save the output image
output_image = np.squeeze(host_output)  # Remove batch dimension
#output_image = np.transpose(output_image, (1, 2, 0))  # Convert to HWC format
#output_image = np.clip(output_image * 255.0, 0, 255).astype(np.uint8)

output_image = output_image.reshape((512, 512, 3))
print(output_image.shape)
#output_image = output_image.reshape((shape[2], shape[3], shape[1])) # image.channels comes last, e.g., 3 for RGB
output_image = np.clip(output_image * 255.0, 0, 255).astype(np.uint8) # BoZhang: without this line, the output image will be all black

cv2.imwrite(sys.argv[3], output_image)

del host_output
inputs[0]['allocation'].free()
outputs[0]['allocation'].free()

del context
del engine