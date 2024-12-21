import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

import numpy as np
from scipy.special import softmax as softmax_scipy 
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

getbytes = lambda x : x*np.dtype(np.float32).itemsize


def attn_gpu_naive(Q, K, V, BS = 32, kernels_path = '../naiveKernel/simple.cu'):
    
    with open(kernels_path, 'r') as f:
        kernels = r"{}".format(f.read())

    mod = SourceModule(kernels)
    matmul_t = mod.get_function('matmulT')
    find_max = mod.get_function('findMax')
    find_sum = mod.get_function('findSum')
    softmax = mod.get_function('Softmax')
    matmul = mod.get_function('matmul')

    N, d = Q.shape
    N = np.int32(N)
    d = np.int32(d)

    bsize = int(BS)
    gsize = int((N-1)/BS + 1)
    one = int(1)

    block = (bsize, bsize, one)
    grid2d = (gsize, gsize, one)
    grid1d = (one, gsize, one)

    

    S = np.zeros((N, N)).astype(np.float32)
    P = np.empty_like(S)
    m = 1e-10*np.ones((N, 1)).astype(np.float32)
    l = np.zeros((N, 1)).astype(np.float32)

    start_kernel = cuda.Event()
    end_kernel = cuda.Event()
    start_mem = cuda.Event()
    end_mem = cuda.Event()

    start_mem.record()

    Q_gpu = gpuarray.to_gpu(Q)
    K_gpu = gpuarray.to_gpu(K)
    V_gpu = gpuarray.to_gpu(V)
    O_gpu = gpuarray.empty_like(Q_gpu)
    S_gpu = gpuarray.to_gpu(S)
    P_gpu = gpuarray.to_gpu(P)
    m_gpu = gpuarray.to_gpu(m)
    l_gpu = gpuarray.to_gpu(l)

    

    start_kernel.record()

    matmul_t(S_gpu, Q_gpu, K_gpu, np.int32(N), np.int32(d), block=block, grid=grid2d)
    smem_max = np.dtype(np.float32).itemsize*(3*block[1] + block[1]*block[0])
    find_max(S_gpu, m_gpu, N, shared=smem_max, block=block, grid=grid1d)
    smem_sum = np.dtype(np.float32).itemsize*(2*block[1] + block[1]*block[0])
    find_sum(S_gpu, m_gpu, l_gpu, shared=smem_sum, block=block, grid=grid1d)
    softmax(P_gpu, S_gpu, m_gpu, l_gpu, N, block=block, grid=grid2d)
    matmul(O_gpu, P_gpu, V_gpu, N, d, block=block, grid=grid2d)

    end_kernel.record()

    O = O_gpu.get()

    end_mem.record()
    cuda.Context.synchronize()

    total_time = start_mem.time_till(end_mem)
    kernel_time = start_kernel.time_till(end_kernel)
    mem_time = total_time - kernel_time

    return O, total_time, kernel_time, mem_time



def attn_gpu_1d(Q, K, V, BS=32, kernels_path='../1DGrid/1dgrid.cu'):
    with open(kernels_path, 'r') as f:
        kernel = r"{}".format(f.read())

    mod = SourceModule(kernel, options=["-w", "-G", "-O0"], no_extern_c=True)
    func = mod.get_function("attn_forward")

    N, d = Q.shape
    N = np.int32(N)
    d = np.int32(d)

    bsize = int(BS)
    gsize = int((N-1)/BS + 1)
    one = int(1)

    block = (bsize, bsize, one)
    grid1d = (one, gsize, one)
    

    m = 1e-10*np.ones((N, 1)).astype(np.float32)
    l = np.zeros((N, 1)).astype(np.float32)


    Br = block[1]
    Bc = block[0]
    SHARED_MEM_SIZE = int(np.dtype(np.float32).itemsize*(2*Br*d+2*Bc*d+4*Br+2*Br*Bc+2*Br))
        
    # debug loggers
    view_y = np.int32(0)
    view_x = np.int32(0)
    verbose = np.int32(0)

    # events
    start_kernel = cuda.Event()
    end_kernel = cuda.Event()
    start_mem = cuda.Event()
    end_mem = cuda.Event()

    start_mem.record()

    Q_gpu = gpuarray.to_gpu(Q)
    K_gpu = gpuarray.to_gpu(K)
    V_gpu = gpuarray.to_gpu(V)
    O_gpu = gpuarray.empty_like(Q_gpu)
    m_gpu = gpuarray.to_gpu(m)
    l_gpu = gpuarray.to_gpu(l)

    start_kernel.record()
    func(O_gpu, Q_gpu, K_gpu, V_gpu, m_gpu, l_gpu, d, N, 
         view_y, view_x, verbose, shared=SHARED_MEM_SIZE, block=block, grid=grid1d)
    
    end_kernel.record()

    O = O_gpu.get()
    end_mem.record()

    cuda.Context.synchronize()

    kernel_time = start_kernel.time_till(end_kernel)
    total_time = start_mem.time_till(end_mem)
    mem_time = total_time - kernel_time

    return O, total_time, kernel_time, mem_time


def attn_gpu_2d(Q, K, V, BS=32, kernels_path = '../2DGrid/flashforward-new.cpp'):

    
    N_SIZE, D_SIZE = Q.shape
    with open(kernels_path, 'r') as f:
        kernels = r"{}".format(f.read())

    mod = SourceModule(kernels)


    sft_n = mod.get_function('softmax_num')
    sft_d = mod.get_function('softmax_denom')
    op = mod.get_function('storeTo_O')
    N = np.int32(N_SIZE)
    d = np.int32(D_SIZE)
    block = (int(BS), int(BS), int(1))
    grid_size = int( (N-1)/block[1] + 1)
    grid = (grid_size, grid_size, int(1))

    Tr = grid_size #1
    Tc = grid_size
    Br = block[1] #32
    Bc = block[1]

    SHARED_MEM_SIZE = int((2*Br*d + 2*Bc*d + 2*Br*Bc + 2*Br*d + 2*Br)*np.dtype(np.float32).itemsize) 

    sm_smn = getbytes(int(Br*d + Bc*d + Br*Bc + Br))
    sm_smd = getbytes(int(Br*d + Bc*d + 2*Br*Bc))


    S = np.zeros((Tr*Br, Tc*Bc)).astype(np.float32)
    val = np.zeros((N_SIZE, D_SIZE)).astype(np.float32)
    m = -1e10*np.ones((N_SIZE, 1)).astype(np.float32)
    l = np.zeros((N_SIZE, 1)).astype(np.float32)
    O = np.zeros((N_SIZE, D_SIZE)).astype(np.float32)
    
    start_kernel = cuda.Event()
    end_kernel = cuda.Event()
    start_mem = cuda.Event()
    end_mem = cuda.Event()
    
    start_mem.record()
    
    Q_gpu = gpuarray.to_gpu(Q)
    K_gpu = gpuarray.to_gpu(K)
    V_gpu = gpuarray.to_gpu(V)
    S_gpu = gpuarray.to_gpu(S)
    val_gpu = gpuarray.to_gpu(val)
    m_gpu = gpuarray.to_gpu(m)
    l_gpu = gpuarray.to_gpu(l)
    O_gpu = gpuarray.to_gpu(O)

    
    start_kernel.record()
    sft_n(S_gpu, Q_gpu, K_gpu, m_gpu, d, N, shared=sm_smn, block=block, grid=grid)
    sft_d(val_gpu, S_gpu, V_gpu, m_gpu, l_gpu, d, N, shared=sm_smd, block=block, grid=grid)
    op(O_gpu, val_gpu, l_gpu, d, N, block=block, grid=grid)
    end_kernel.record()
    
    O_gpu_comp = O_gpu.get()
    end_mem.record()
    
    cuda.Context.synchronize()
    
    kernel_time = start_kernel.time_till(end_kernel)
    total_time = start_mem.time_till(end_mem)
    mem_time = total_time - kernel_time
    
    return O_gpu_comp, total_time, kernel_time, mem_time
    
    # print(f"\nO close for N = {N_SIZE}, d = {D_SIZE}, b = {BS}:", np.allclose(O, O_gpu_comp))


def get_QKV(N, d):
    Q = np.random.random((N,d)).astype(np.float32)
    K = np.random.random((N,d)).astype(np.float32)
    V = np.random.random((N,d)).astype(np.float32)
    return Q, K, V


BS = 4
Narray = [16, 64, 128, 256, 512]

naive_data = {'total':[],'kernel':[],'mem':[]}
_1d_data = {'total':[],'kernel':[],'mem':[]}
_2d_data = {'total':[],'kernel':[],'mem':[]}

for d in Narray:
    N = np.int32(256)
    print(N, d)
    Q, K, V = get_QKV(N, d)


    o_naive, naive_total, naive_kernel, naive_mem  = attn_gpu_naive(Q, K, V, BS=BS)
    o_1d, _1d_total, _1d_kernel, _1d_mem = attn_gpu_1d(Q, K, V, BS=BS)
    o_2d, _2d_total, _2d_kernel, _2d_mem = attn_gpu_2d(Q, K, V, BS=BS)

    naive_data['total'].append(naive_total)
    naive_data['kernel'].append(naive_kernel)
    naive_data['mem'].append(naive_mem)


    _1d_data['total'].append(_1d_total) 
    _1d_data['kernel'].append(_1d_kernel)
    _1d_data['mem'].append(_1d_mem)
    
    _2d_data['total'].append(_2d_total) 
    _2d_data['kernel'].append(_2d_kernel)
    _2d_data['mem'].append(_2d_mem)



def plot_execution_times(naive_data, _1d_data, Narray, logtoggle, savepath=None):
    """
    Plots execution times for naive and 1D data.
    // reuse for 2d
    Parameters:
    naive_data (dict): Dictionary containing 'total', 'kernel', and 'mem' lists for naive data.
    _1d_data (dict): Dictionary containing 'total', 'kernel', and 'mem' lists for 1D data.
    Narray (list): List of input sizes (N).
    logtoggle (bool): If true, set x-axis to logarithmic scale and plot log of data.
    
    Returns:
    None
    """
    
    plt.figure(figsize=(10, 6))
    
    # Plotting naive_data
    plt.plot(Narray, naive_data['total'], marker='o', label='2d total')
    plt.plot(Narray, naive_data['kernel'], marker='o', label='2d kernel')
    plt.plot(Narray, naive_data['mem'], marker='o', label='2d mem')
    
    # Plotting _1d_data
    # plt.plot(Narray, _1d_data['total'], marker='s', label='_1d total')
    # plt.plot(Narray, _1d_data['kernel'], marker='s', label='_1d kernel')
    # plt.plot(Narray, _1d_data['mem'], marker='s', label='_1d mem')
    plt.plot(Narray, _1d_data['total'], marker='s', label='naive total')
    plt.plot(Narray, _1d_data['kernel'], marker='s', label='naive kernel')
    plt.plot(Narray, _1d_data['mem'], marker='s', label='naive mem')

    # Setting x-axis and y-axis scales
    if logtoggle:
        plt.xscale('log')
        plt.yscale('log')
        
    title = 'Execution Time vs Input Size for Naive and 2D CUDA kernels - same N'
    if(logtoggle):
        title = '(log scale) '+ title
    # Adding labels and title
    plt.xlabel('Input dimension (d)')
    plt.ylabel('Execution Time (ms)')
    plt.title(title)
    
    # Adding grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Adding legend
    plt.legend(loc='best', title="Legend")
    
    # Customizing ticks
    plt.xticks(Narray, labels=Narray)
    # plt.tight_layout() # Adjust layout to prevent label overlap
    if(savepath is not None):
        plt.savefig(savepath)



plot_execution_times(_2d_data, naive_data, Narray, logtoggle=False, savepath='./naive2Dcompare_fixedD.pdf')
plot_execution_times(_2d_data, naive_data, Narray, logtoggle=True, savepath='./naive2Dcompare_log_fixedD.pdf')