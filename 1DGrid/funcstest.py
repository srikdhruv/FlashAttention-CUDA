"""
Author: Pranav Kumar Kota
"""
import numpy as np
from scipy.special import softmax
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import nvtx

def blockwise_softmax(S, Br, Bc):
    Pblk = S.copy()
    shape = S.shape
    print(f"S shape = {S.shape}, Br={Br}, Bc={Bc}")
    Tr = int(shape[0]/Br)
    Tc = int(shape[1]/Bc)
    print(f"Tr = {Tr}, Tc = {Tc}")
    for tr in range(Tr):
        for tc in range(Tc):
            rowrange = [tr*Br+i for i in range(Br)]
            colrange = [tc*Bc+i for i in range(Bc)]
            block = S[np.ix_(rowrange, colrange)]
            block = softmax(block, axis=1)
            Pblk[np.ix_(rowrange, colrange)] = block
            # print(f"Block ({tr}, {tc}):")
            # print(block)
    return Pblk

def printblock(S, by, bx, Br, Bc):
    rowrange = [by*Br+i for i in range(Br)]
    colrange = [bx*Bc+i for i in range(Bc)]
    block = S[np.ix_(rowrange, colrange)]
    print(block)

def gen_test_mat(N, d, B, T):
    vals = []
    base = np.ones((B, d)).astype(np.float32)
    vals.append(base)
    m0 = base.copy()
    for i in range(1, T):
        m0 = np.vstack([m0, (i+1)*base])
        vals.append((i+1)*base)
    return m0, vals

def randData(N, d):
    Q = np.random.random((N, d)).astype(np.float32)
    K = np.random.random((N, d)).astype(np.float32)
    V = np.random.random((N, d)).astype(np.float32)
    return Q, K, V

def get_gpu_vals(Q, K, V, m, l):
    Q_gpu = gpuarray.to_gpu(Q)
    O_gpu = gpuarray.empty_like(Q_gpu)
    K_gpu = gpuarray.to_gpu(K)
    V_gpu = gpuarray.to_gpu(V)
    m_gpu = gpuarray.to_gpu(m)
    l_gpu = gpuarray.to_gpu(l)
    return O_gpu, Q_gpu, K_gpu, V_gpu, m_gpu, l_gpu

def get_o_vals(qvals, kvals, vvals):
    ovals = []
    for i, qi in enumerate(qvals):
        oi = np.zeros(qi.shape)
        for j, (kj, vj) in enumerate(zip(kvals, vvals)):
            oi += softmax(np.matmul(qi, kj.T), axis=1)@vj
        ovals.append(oi)
    return ovals

def S_blocks(qvals, kvals):
    sblocks = []
    for i,qi in enumerate(qvals):
        rowblocks = []
        for j,kj in enumerate(kvals):
            rowblocks.append(qi@kj.T)
        sblocks.append(rowblocks)
    return sblocks

SIZE = 4
Ns = [16]
ds = [SIZE]
for N in Ns:
# N = 16
    for d in ds:
    # d = 2
        Br = 4
        # Bc = min(SIZE, d)
        Bc = 4

        Tr = int((N-1)/Br + 1)
        Tc = int((N-1)/Bc + 1)
        
        Q, qvals = gen_test_mat(N, d, Br, Tr)
        K, kvals = gen_test_mat(N, d, Bc, Tc)
        V, vvals = gen_test_mat(N, d, Bc, Tc)
        # ovals = get_o_vals(qvals, kvals, vvals)
        # sblocks = S_blocks(qvals, kvals)

        
        # Step 1: Compute S = Q @ K.T
        S = np.dot(Q, K.T)

        # Step 2: Find the row max of S (optional)
        m = np.max(S, axis=1, keepdims=True)


        # Step 4: Apply softmax to get P
        P = softmax(S-m)

        l = np.sum(P, axis=1, keepdims=True)

        # Step 5: Compute O = P @ V
        O_cpu = np.dot(P, V)

        with open("1dgrid.cu", "r") as f:
            kernel = f.read()

        mod = SourceModule(kernel, options=["-w", "-G", "-O0"], no_extern_c=True)
        func = mod.get_function("attn_forward")



        O_gpu, Q_gpu, K_gpu, V_gpu, m_gpu, l_gpu = get_gpu_vals(Q, K, V, m=-1e10*np.ones((N,1)).astype(np.float32), l=np.zeros((N,1)).astype(np.float32))

        block = (int(Bc), int(Br), int(1))
        grid = (int(1), int(Tr), int(1))
        N = np.int32(N)
        d = np.int32(d)

        view_y = np.int32(0)
        view_x = np.int32(2) 
        verbose = np.int32(1)

        SHARED_MEM_SIZE = int(np.dtype(np.float32).itemsize*(2*Br*d+2*Bc*d+4*Br+2*Br*Bc+2*Br))
        print(f"N = {N}, d = {d}, Br = {Br}, Bc = {Bc}")
        print(f"Tr = {Tr}, Tc = {Tc}")
       

        func(O_gpu, Q_gpu, K_gpu, V_gpu, m_gpu, l_gpu, d, N, view_y, view_x, verbose, shared=SHARED_MEM_SIZE, block=block, grid=grid)
        cuda.Context.synchronize()
 
        O = O_gpu.get()

        print("All m's close: ",np.allclose(m,m_gpu.get()))
        print("All l's close: ",np.allclose(l,l_gpu.get()))
        
        valid = np.allclose(O, O_cpu, atol=1e-5)
        if(not valid):
            print(f"Failed !! Average diff = {np.mean(np.abs(O-O_cpu))}")
        else:
            print("Passed !!")
