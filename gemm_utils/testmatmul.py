import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np

BLOCK = (4, 4, 1)  # Use a tuple of integers for BLOCK dimensions


def check_cuda_errors():
    pass
    # error = cuda.Context.synchronize()
    # if error != cuda.Context.synchronize():
    #     print(f"CUDA Error: {cuda.Error(error)}")
    # else:
    #     error = cuda.Context.get_last_cuda_error()
    #     if error:
    #         print(f"CUDA Error: {cuda.Error(error)}")


def GRID(M, N):
    return (int(np.ceil(M / BLOCK[0])), int(np.ceil(N / BLOCK[1])),
            1)  # Return a tuple of integers for GRID dimensions


def get_random_matrices(M, K, N):
    A = np.random.random((M, K)).astype(np.float32)
    B = np.random.random((K, N)).astype(np.float32)
    C = np.random.random((M, N)).astype(np.float32)
    return A, B, C


def get_ABC_gpu(A, B, C):
    A_gpu = gpuarray.to_gpu(A)
    B_gpu = gpuarray.to_gpu(B)
    C_gpu = gpuarray.to_gpu(C)
    return A_gpu, B_gpu, C_gpu


def load_kernel(kernel_path='./matmulKernels.cpp'):
    with open(kernel_path, 'r') as f:
        kernel = f.read()
    kernel = r'{}'.format(kernel)
    return SourceModule(kernel)


def matmul_gpu(A, B, C, M, K, N, fname='naive', shared=False):
    A_gpu, B_gpu, C_gpu = get_ABC_gpu(A, B, C)
    mod = load_kernel()
    func = mod.get_function(fname)
    if (not shared):
        func(C_gpu, A_gpu, B_gpu, M, K, N, block=BLOCK, grid=GRID(M, N))
    else:
        SHARED_MEM_SIZE = int(BLOCK[0] * BLOCK[1] * 3
                              * np.dtype(np.float32).itemsize)
        func(C_gpu, A_gpu, B_gpu, M, K, N, block=BLOCK, grid=GRID(M, N),
             shared=SHARED_MEM_SIZE)
    C = C_gpu.get()
    return C


def matmul_withtranspose_gpu(A, B, C, M, K, N, fname='naive', shared=False):
    A_gpu, B_gpu, C_gpu = get_ABC_gpu(A, B.T, C)
    mod = load_kernel()
    func = mod.get_function(fname)
    if (not shared):
        func(C_gpu, A_gpu, B_gpu, M, K, N, block=BLOCK, grid=GRID(M, N))
    else:
        SHARED_MEM_SIZE = int(BLOCK[0] * BLOCK[1] * 3
                              * np.dtype(np.float32).itemsize)
        func(C_gpu, A_gpu, B_gpu, M, K, N, block=BLOCK, grid=GRID(M, N),
             shared=SHARED_MEM_SIZE)
    C = C_gpu.get()
    return C


def matmul_cpu(A, B, C, M, K, N, transpose=False):
    Ashape = A.shape
    Bshape = B.shape
    if (not transpose):
        assert (Ashape[1] == Bshape[0])
        C = np.matmul(A, B)
    else:
        assert (Ashape[1] == Bshape[1])
        C = np.matmul(A, B.T)
    return C


def valid_check(M, K, N, fname='naive', verbose=False, shared=False,
                transpose=False):
    A, B, C = get_random_matrices(M, K, N)
    if (transpose):
        C_gpu_comp = matmul_withtranspose_gpu(A, B, C, M, K, N, fname=fname,
                                              shared=shared)
    else:
        C_gpu_comp = matmul_gpu(A, B, C, M, K, N, fname=fname, shared=shared)
    C_cpu_comp = matmul_cpu(A, B, C, M, K, N, transpose)
    try:
        assert (np.allclose(C_cpu_comp, C_gpu_comp))
        print("Valid computation!")
        return 1
    except Exception:
        if (verbose):
            return 0
        else:
            raise Exception("CPU and GPU results do not match!")


sizes = 2**np.arange(4, 12).astype(np.int32)
# sizes = np.array([1235, 257]).astype(np.int32)
num_exps = 1
print(f"Block size = {BLOCK}")
for i in range(num_exps):
    TRANSPOSE = False
    for size in sizes:
        M, K, N = size, size, size
        print("*"*50)
        print(f"Testing M = {M}, K = {K}, N = {N}, transpose={TRANSPOSE}")
        valid = valid_check(M, K, N, fname='sharedMemKernel', shared=True,
                            verbose=True, transpose=TRANSPOSE)
        check_cuda_errors()
        if (valid == 1):
            print("\033[32mTest passed!\033[0m")
            print("*"*50)
        elif (valid == 0):
            print("\033[31m.....FAILED!.....\033[0m")
            print("*"*50)


