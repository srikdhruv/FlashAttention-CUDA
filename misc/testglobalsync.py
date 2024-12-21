import time 
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

# cuda.init()
device = cuda.Device(0)
print(f"Compute capability: {device.compute_capability()}")




try:
    
    mod = SourceModule("""
    #include <cooperative_groups.h>
    #include <cuda/barrier>
    using namespace cooperative_groups;
                       
    extern "C" __global__ void myKernel() {
    
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        printf("Before sync: Thread %d\\n", tid);
        
        grid_group grid = this_grid();
                       
        printf("After get grid: Thread %d\\n", tid);
        grid.sync();  // Synchronize threads within the grid
        
        printf("After sync: Thread %d\\n", tid);
    }

    
    """, options=[], no_extern_c=True)
except Exception as e:
    print(f"Error compiling CUDA kernel: {e}")

# # Attempt to retrieve the kernel function
# try:
#     kernel = mod.get_function("myKernel")
#     threads_per_block = 4
#     blocks = 2
#     kernel(grid=(blocks, 1, 1), block=(threads_per_block, 1, 1))
#     error = cuda.Error()
#     if error != cuda.stderr:
#         print(f"CUDA Error: {error}")
# except Exception as e:
#     print(f"Error retrieving kernel function: {e}")



# Retrieve the kernel function
try:
    kernel = mod.get_function("myKernel")
except pycuda.driver.LogicError as e:
    print("Logic Error in Kernel Retrieval:", e)
    raise  # Re-raise the exception

# Set grid and block dimensions
threads_per_block = 4
blocks = 2


# Launch the kernel
try:
    # kernel(grid=(blocks, 1, 1), block=(threads_per_block, 1, 1))
    cuda.Context.get_current().launch_cooperative_kernel(
        kernel,
        (blocks,1,1),
        (threads_per_block,1,1)
    )
except pycuda.driver.LaunchError as e:
    print("Kernel Launch Error:", e)
    raise  # Re-raise the exception

# Launch the kernel
try:
    kernel(grid=(blocks, 1, 1), block=(threads_per_block, 1, 1))
except pycuda.driver.RuntimeError as e:
    print("CUDA Runtime Error:", e)
    raise  # Re-raise the exception

# time.sleep(10)
# print("All fresh!")

# cuda.Context.synchronize()
# Optionally, you can check for CUDA errors after kernel execution
# error = cuda.Context.get_last_cuda_error()
# if error:
#     print(f"CUDA Error: {error}")





