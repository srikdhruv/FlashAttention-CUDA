# import modules
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuad.compiler import SourceModule


# get random Q, K, V to perform tests


def get_random_QKV(N, d):
    Q = np.random.random(size=(N, d)).astype(np.float32)
    K = np.random.random(size=(N, d)).astype(np.float32)
    V = np.random.random(size=(N, d)).astype(np.float32)
    return Q, K, V

# send Q, K, V to GPUs


def get_QKV_gpu(Q, K, V):
    Q_gpu = gpuarray.to_gpu(Q)
    K_gpu = gpuarray.to_gpu(K)
    V_gpu = gpuarray.to_gpu(V)
    return Q_gpu, K_gpu, V_gpu

# compile kernel code and return source module


def load_source_module(kernel_path="./kernels.cpp"):
    with open(kernel_path, 'r') as f:
        kernel = f.read()
    return SourceModule(kernel)

# define attention class - init params, perform forward and valid_check


class Attention():
    def __init__(self, N, d, Q, K, V, kernel_path="./kernels.cpp"):
        try:
            Q_shape = np.shape(Q)
            K_shape = np.shape(K)
            V_shape = np.shape(V)
            assert (Q_shape == (N, d)
                    and K_shape == (N, d)
                    and V_shape == (N, d))
        except Exception:
            raise Exception(f"Expected input shapes->({N}, {d})\
            - got Q->({Q_shape}), K->({K_shape}), V->({V_shape})")
        self.Q, self.K, self.V = Q, K, V
        self.Q_gpu, self.K_gpu, self.V_gpu = get_QKV_gpu(Q, K, V)
        self.mod = load_source_module(kernel_path)
        self.set_grid_args(32, 32)  # standard block sizes

    def reinit(self, N, d, Q, K, V):
        try:
            Q_shape = np.shape(Q)
            K_shape = np.shape(K)
            V_shape = np.shape(V)
            assert (Q_shape == (N, d)
                    and K_shape == (N, d)
                    and V_shape == (N, d))
        except Exception:
            raise Exception(f"Expected input shapes->({N}, {d})\
            - got Q->({Q_shape}), K->({K_shape}), V->({V_shape})")
        self.Q, self.K, self.V = Q, K, V
        self.Q_gpu, self.K_gpu, self.V_gpu = get_QKV_gpu(Q, K, V)

    def set_grid_args(self, Br, Bc):
        try:
            assert (Br*Bc <= 1024)
        except Exception:
            raise Exception(f"Max threads per grid = 1024!\
            Br*Bc ({Br}*{Bc}) = {Br*Bc}")
        self.Br = Br
        self.Bc = Bc
        self.Tr = np.ceil(self.N/Br)
        self.Tc = np.ceil(self.N/Bc)

    def std_attn(self):
        def softmax(x):
            """
            Compute softmax values for each row of the matrix x.
            """
            e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e_x / np.sum(e_x, axis=1, keepdims=True)
        
        S = np.matmul(self.Q, self.K.T)
        P = softmax(S)
        out = np.matmul(P, self.V)
        return out

    def forward(self, funcname='attn_kernel'):
        func = self.mod.get_function(funcname)
        block = (self.Br, self.Bc)
        grid = (self.Tr, self.Tc)
        out_gpu = func(self.Q_gpu, self.K_gpu, self.V_gpu, block=block, grid=grid)
        out = out_gpu.get()
        return out

    def valid_check(self, funcname='attn_kernel', verbose=True):
        O_gpu_comp = self.forward(funcname)
        O_cpu_comp = self.std_attn()
        try:
            assert (np.allclose(O_cpu_comp, O_gpu_comp))
            print("Valid computation!")
            return 1
        except Exception:
            if (verbose):
                return 0
            else:
                raise Exception("CPU and GPU results do not match!")


if _name__ == "__main__":

    print("Testing attention kernel")
    sizes = 2**np.arange(4, 12)
    d = 16
    attn = Attention(N=1, d=d, Q=None, K=None, V=None)
    for N in sizes:
        attn.N = N
        Q, K, V = get_random_QKV(N, d)
        # reinitalize attn module with updates N and d
        attn.reinit(N, d, Q, K, V)
        print("*"*30)
        print(f"Testing N = {N}, d = {d}")
        # perform test on new params
        valid = attn.valid_check()
        if (valid == 1):
            print("Test passed!")
            print("*"*30)
        elif (valid == 0):
            print(".....FAILED!.....")
            print("*"*30)