import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

import numpy as np
from scipy.special import softmax as softmax_scipy 

kernels_path = './flashforward-new.cpp'

with open(kernels_path, 'r') as f:
    kernels = r"{}".format(f.read())

mod = SourceModule(kernels)


sft_n = mod.get_function('softmax_num')
sft_d = mod.get_function('softmax_denom')
op = mod.get_function('storeTo_O')

getbytes = lambda x : x*np.dtype(np.float32).itemsize


def standard_attention(Q, K, V):
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    S = np.matmul(Q, K.T)s
    P = softmax(S)
    out = np.matmul(P, V)
    
    return out

            

    

    
# N_range = [16, 32, 64]
N_range = [64]
d_range = [16, 32, 64]
# b_range = [16, 32, 64]
d_range = [16]
b_range = [32]

for N_SIZE in N_range:
    for D_SIZE in d_range:
        for BS in b_range:
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

            sm_smn = getbytes( int(Br*d + Bc*d + Br*Bc + Br))
            sm_smd = getbytes(int(Br*d+Bc*d+2*Br*Bc))


            Q = np.arange(0, N_SIZE*D_SIZE).reshape(N_SIZE, D_SIZE).astype(np.float32)
            K = np.arange(0, N_SIZE*D_SIZE).reshape(N_SIZE, D_SIZE).astype(np.float32)
            V = np.arange(0, N_SIZE*D_SIZE).reshape(N_SIZE, D_SIZE).astype(np.float32)
            S = np.zeros((Tr*Br, Tc*Bc)).astype(np.float32)
            val = np.zeros((N_SIZE, D_SIZE)).astype(np.float32)
            m = -1e10*np.ones((N_SIZE, 1)).astype(np.float32)
            l = np.zeros((N_SIZE, 1)).astype(np.float32)
            O = np.zeros((N_SIZE, D_SIZE)).astype(np.float32)

            Q_gpu = gpuarray.to_gpu(Q)
            K_gpu = gpuarray.to_gpu(K)
            V_gpu = gpuarray.to_gpu(V)
            S_gpu = gpuarray.to_gpu(S)
            val_gpu = gpuarray.to_gpu(val)
            m_gpu = gpuarray.to_gpu(m)
            l_gpu = gpuarray.to_gpu(l)
            O_gpu = gpuarray.to_gpu(O)

            O = standard_attention(Q, K, V)

            sft_n(S_gpu, Q_gpu, K_gpu, m_gpu, d, N, shared=sm_smn, block=block, grid=grid)
            sft_d(val_gpu, S_gpu, V_gpu, m_gpu, l_gpu, d, N, shared=sm_smd, block=block, grid=grid)
            op(O_gpu, val_gpu, l_gpu, d, N, block=block, grid=grid)

            O_gpu_comp = O_gpu.get()

            print(f"\nO close for N = {N_SIZE}, d = {D_SIZE}, b = {BS}:", np.allclose(O, O_gpu_comp))