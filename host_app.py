'''
Copyright 2020 Xilinx Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from ctypes import *
from typing import List
from glob import glob
import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
import sys
import argparse
import shutil


#IMPORT DATASET
test_images_path = "dataset/images/test"
test_masks_path = "dataset/masks/test"

divider = '------------------------------------'

def parse_image(img_path, input_scale):
    image_size = 256
    image = cv2.imread(img_path, 0)
    h, w = image.shape
    if (h == image_size) and (w == image_size):
        pass
    else:
        image = cv2.resize(image, (image_size, image_size))
    # Enlarge shape to (image_size, image_size, 1)
    image = np.expand_dims(image, -1)
    # Normalize pixels between [0.,1.]
    image = image.astype('float32') / 255.0
    # Remove background noise
    image = np.where(image > 0.05, image, 0.0)
    image = image * input_scale 
    image = image.astype(np.int8)
    return image


def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]
  

def runThreads(id, dpu, img, input_scale):
    """Single thread .xmodel inference process"""
    '''get tensor'''
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)
    outputSize = int(outputTensors[0].get_data_size() / input_ndim[0])
    
    print("Running thread : ", id)

    output_data = [np.empty(output_ndim, dtype=np.int8, order="C")]
    input_data = [np.empty(input_ndim, dtype=np.int8, order="C")]
    # Create a pointer to the input buffer
    image_ptr = input_data[0]

    for i in range(len(img)):       
        s_img = parse_image(img[i], input_scale)
        image_ptr[0,...] = s_img
        job_id = dpu.execute_async(input_data, output_data)
        dpu.wait(job_id)

def app(threads, model):
    """Runs the n-threads inference process"""

    test_images = glob(os.path.join(test_images_path, "*"))
    test_images.sort()
    runTotal = len(test_images)

    if threads == 1:
        batch_size = len(test_images)
    else:
        batch_size = len(test_images) // threads-1 

    #create batches to be runned by each thread
    img_sub_arrays = [test_images[i:i + batch_size] for i in range(0, len(test_images), batch_size)]

    global out_q
    out_q = [None] * runTotal
    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)
    all_dpu_runners = []
    for i in range(threads):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    # input scaling
    input_fixpos = all_dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2 ** input_fixpos

    threadAll = []

    for i in range(threads):

        t1 = threading.Thread(target=runThreads, args=(i, all_dpu_runners[i], img_sub_arrays[i], input_scale))
        threadAll.append(t1)

    time1 = time.time()
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    time2 = time.time()
    timetotal = time2 - time1

    fps = float(runTotal / timetotal)
    print(divider)
    print("Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds" % (fps, runTotal, timetotal))
    print(divider)

    return


# only used if script is run as 'main' from command line
def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-t', '--threads', type=int, default=3, help='Number of threads. Default is 3')
    ap.add_argument('-m', '--model', type=str, default='HippoScan_Ultra96-2.xmodel',
                    help='Path of xmodel. Default is HippoScan_Ultra96-2.xmodel')
    args = ap.parse_args()

    print('Command line options:')
    print(' --threads   : ', args.threads)
    print(' --model     : ', args.model)

    app(args.threads, args.model)


if __name__ == '__main__':
    main()
