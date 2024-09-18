import os
import shutil
import argparse

parser = argparse.ArgumentParser(description='Creating input batches.')
parser.add_argument('--input_folder', type=str, required=True, help='The input folder that contains all the images')
parser.add_argument('--output_folder', type=str, required=True, help='The output folder that contains all image batches')
parser.add_argument('--batch_size', type=int, required=True, help='The batch size')

args = parser.parse_args()

# Create the root output folder
os.makedirs(args.output_folder, exist_ok=True)
root, dirs, files = next(os.walk(args.input_folder))
num_batches = len(files) / args.batch_size 

# Create all the batches' folder under the root folder
for i in range(int(num_batches)):
    print(args.output_folder + "/batch_" + ("%04d" % (i+1)))
    os.makedirs(args.output_folder + "/batch_" + ("%04d" % (i+1)), exist_ok=True)

# Distribute (copy) images under [input_folder] to the right batch subfolders under [output_folder]
image_idx = 0
for f in files:
    batch_idx = int(image_idx / args.batch_size) + 1
    image_idx += 1
    batch_folder = args.output_folder + "/batch_" + ("%04d" % batch_idx) + "/"
    if batch_idx <= num_batches:
        shutil.copyfile(os.path.join(root, f), batch_folder + f)