import numpy as np
from sys import getsizeof
import time
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import nibabel as nib
from multiprocessing.sharedctypes import Value, Array, RawArray
from multiprocessing import Process, Lock
import multiprocessing as mp
import ctypes
import scipy as sp
from scipy import ndimage

# Force spawn method (required for Windows)
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

print(f"Multiprocessing start method: {mp.get_start_method()}")

# Helper functions
def to_numpyarray(mp_arr):
    return np.frombuffer(mp_arr.get_obj(), dtype=np.float32)

def init_sharedarray(shared_array, img_shape, padded_shared_array, padded_img_shape, kernel):
    global filtered_shared_space
    global filtered_shared_matrix

    filtered_shared_space = shared_array
    filtered_shared_matrix = to_numpyarray(filtered_shared_space).reshape(img_shape)

    global padded_shared_space
    global padded_shared_matrix
    padded_shared_space = padded_shared_array
    padded_shared_matrix = to_numpyarray(padded_shared_space).reshape(padded_img_shape)

    global kernel_matrix
    kernel_matrix = kernel
    return

def same_padding(pad, arr):
    (rows, cols, depths) = arr.shape

    new_rows = rows + 2*pad
    new_cols = cols + 2*pad
    new_depths = depths + 2*pad

    padded = np.zeros((new_rows, new_cols, new_depths), dtype=arr.dtype)
    
    padded[pad:pad+rows, pad:pad+cols, pad:pad+depths] = arr

    for i in range(pad):
        # Pad along rows
        padded[i, pad:pad+cols, pad:pad+depths] = arr[0, :, :]
        padded[new_rows - i - 1, pad:pad+cols, pad:pad+depths] = arr[rows - 1, :, :]

        # Pad along columns
        padded[:, i, pad:pad+depths] = padded[:, pad, pad:pad+depths]
        padded[:, new_cols - i - 1, pad:pad+depths] = padded[:, new_cols - pad - 1, pad:pad+depths]

        # Pad along depths
        padded[:, :, i] = padded[:, :, pad]
        padded[:, :, new_depths - i - 1] = padded[:, :, new_depths - pad - 1]

    return padded

def convolve(row):
    global filtered_shared_space
    global filtered_shared_matrix
    global padded_shared_space
    global padded_shared_matrix
    global kernel_matrix

    print(f"Processing row {row}")

    rows, cols, depths = filtered_shared_matrix.shape
    k = kernel_matrix.shape[0] // 2

    filtered_row = np.zeros((cols, depths), dtype=np.float32)
    
    for c in range(cols):
        for d in range(depths):
            padded_row = row + k
            padded_c = c + k
            padded_d = d + k
            
            roi = padded_shared_matrix[padded_row - k:padded_row + k + 1, 
                         padded_c - k:padded_c + k + 1, 
                         padded_d - k:padded_d + k + 1]
            
            filtered_row[c, d] = float(np.sum(roi * kernel_matrix))
    
    with filtered_shared_space.get_lock():
        filtered_shared_matrix[row, :, :] = filtered_row

    print(f"Finished row {row}")
    return

def run_parallel(output_shm, input_image, input_shm, padded_shape, filter_kernel, NUMCORES=20):
    image_shape = input_image.shape
    
    padded_image_arr = same_padding(filter_kernel.shape[0]//2, input_image)
    
    padded_image = to_numpyarray(input_shm).reshape(padded_shape)
    padded_image[:] = padded_image_arr[:]
    
    rows = range(image_shape[0])

    with mp.Pool(processes=NUMCORES,
                initializer=init_sharedarray,
                initargs=[output_shm, image_shape, input_shm, padded_shape, filter_kernel]) as p:
        p.map(convolve, rows)

    result_matrix = to_numpyarray(output_shm).reshape(image_shape)
    return result_matrix

if __name__ == '__main__':
    # Load image
    MYDRIVE = "./imgs"
    IMAGE = join(MYDRIVE, 'someones_epi.nii.gz')
    
    print(f"Image to load: {IMAGE}")
    
    epi_img = nib.load(IMAGE)
    epi_img_data = epi_img.get_fdata().astype(np.float32)
    epi_img_data_final = epi_img_data.T
    
    print(f"Image shape: {epi_img_data_final.shape}")
    
    # Define filters
    filter1 = np.zeros((3,3,3), dtype=np.float32)
    filter1[1,1,1] = 1.0
    
    filter2 = np.zeros((3,3,3), dtype=np.float32)
    filter2[0,0,1] = -0.25
    filter2[2,0,1] = -0.25
    filter2[0,2,1] = 0.25
    filter2[2,2,1] = 0.25
    
    # Setup experiment
    image = epi_img_data_final[:,:,:]
    filter_to_use = filter1
    
    padded_image_arr = same_padding(filter_to_use.shape[0]//2, image)
    
    image_shape = image.shape
    print(f"Image shape: {image_shape}")
    data_size = image_shape[0]*image_shape[1]*image_shape[2]
    
    filtered_image_shm = Array(ctypes.c_float, data_size, lock=True)
    
    padded_shape = padded_image_arr.shape
    print(f"Padded shape: {padded_shape}")
    padded_data_size = padded_shape[0]*padded_shape[1]*padded_shape[2]
    
    padded_image_shm = Array(ctypes.c_float, padded_data_size, lock=True)
    
    # Run parallel execution
    print("Starting parallel execution...")
    start_time = time.time()
    
    result = run_parallel(filtered_image_shm, image, padded_image_shm, padded_shape, filter_to_use, NUMCORES=4)
    
    end_time = time.time()
    
    print(f"Parallel execution complete in {end_time - start_time:.2f}s!")
    print(f"Result shape: {result.shape}")
    print(f"Matches original (filter1): {np.allclose(result, image, atol=1e-4)}")
    
    # Verify against scipy
    actual_output = ndimage.correlate(image, filter_to_use, mode='nearest')
    print(f"Matches scipy output: {np.allclose(result, actual_output, atol=1e-4)}")