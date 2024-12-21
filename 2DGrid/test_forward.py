import numpy as np
import pycuda.autoinit
import pycuda
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

# global tile sizes

BR = 4
BC = 4

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

def ml_args(N):
    m = np.ones((N,1)).astype(np.float32)
    l = np.ones((N,1)).astype(np.float32)
    return m,l

def ml_args_gpu(m, l):
    m_gpu = gpuarray.to_gpu(m)
    l_gpu = gpuarray.to_gpu(l)
    return m_gpu, l_gpu


# compile kernel code and return source module


def load_source_module(kernel_path="./flashforward.cpp"):
    with open(kernel_path, 'r') as f:
        kernel = f.read()
    return SourceModule(kernel, options=['--ptxas-options=-v'])

# define attention class - init params, perform forward and valid_check


class Attention():
    def __init__(self, N, d, Q, K, V, m, l, kernel_path="./flashforward.cpp"):
        if(Q is not None):
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
            self.O = np.empty_like(Q)
            self.O_gpu = gpuarray.to_gpu(O)
            self.m, self.l = m, l
            self.m_gpu, self.l_gpu = ml_args_gpu(m, l)
            self.N = N
            self.d = d
            try:
                print("Module Initialized")
                self.mod = load_source_module(kernel_path)
            except:
                self.mod = None
            
            self.set_grid_args(np.int32(BC), np.int32(BR))  # standard block sizes
        

    def reinit(self, N, d, Q, K, V, m, l, kernel_path="./flashforward.cpp"):
        try:
            Q_shape = np.shape(Q)
            K_shape = np.shape(K)
            V_shape = np.shape(V)
            m_shape = np.shape(m)
            l_shape = np.shape(l)
            print("*"*10,"\n",Q_shape, m_shape)

            assert (Q_shape == (N, d)
                    and K_shape == (N, d)
                    and V_shape == (N, d)
                    and m_shape == (N, 1)
                    and l_shape == (N, 1))
        except Exception:
            raise Exception(f"Expected input shapes->({N}, {d})\
            - got Q->({Q_shape}), K->({K_shape}), V->({V_shape})")
        self.Q, self.K, self.V = Q, K, V
        self.Q_gpu, self.K_gpu, self.V_gpu = get_QKV_gpu(Q, K, V)
        self.O = np.empty_like(Q)
        self.O_gpu = gpuarray.to_gpu(self.O)
        self.m, self.l = m, l
        self.m_gpu, self.l_gpu = ml_args_gpu(m, l)
        self.N = np.int32(N)
        self.d = np.int32(d)
        self.mod = self.mod = load_source_module(kernel_path)
        self.set_grid_args(np.int32(BC), np.int32(BR))  # standard block sizes

    def set_grid_args(self, Br, Bc):
        try:
            assert (Br*Bc <= 1024)
        except Exception:
            raise Exception(f"Max threads per grid = 1024!\
            Br*Bc ({Br}*{Bc}) = {Br*Bc}")
        self.Br = int(Br)
        self.Bc = int(Bc)
        self.Tr = int(np.ceil(self.N/Br))
        self.Tc = int(np.ceil(self.N/Bc))

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
        block = (self.Br, self.Bc, int(1))
        grid = (self.Tr, self.Tc, int(1))
        Br = self.Br
        Bc = self.Bc
        d = self.d
        N = self.N 
        SHAREDM_MEM_SIZE = int((2*Br*d + 2*Bc*d + 2*Br*Bc + 2*Br*d + 2*Br)*np.dtype(np.float32).itemsize)
        print("theoretical sm size: ",SHAREDM_MEM_SIZE)
        SHAREDM_MEM_SIZE = 1

        # func(self.O_gpu, self.Q_gpu, self.K_gpu, self.V_gpu, self.m, self.l, self.d, self.N, shared=SHAREDM_MEM_SIZE, block=block, grid=grid)

        try:
            func(self.O_gpu, self.Q_gpu, self.K_gpu, self.V_gpu, self.m, self.l, self.d, self.N, 
                shared=SHAREDM_MEM_SIZE, block=block, grid=grid)
            cuda.Context.synchronize()
        except Exception as e:
            print(f"Kernel launch failed with error: {e}")
            import traceback
            traceback.print_exc()
        cuda.Context.synchronize()
        self.O = self.O_gpu.get()
 
        

    def valid_check(self, funcname='flashattention', verbose=True):
        self.forward(funcname)
        O_cpu_comp = self.std_attn()
        try:
            assert (np.allclose(O_cpu_comp, self.O))
            print("Valid computation!")
            return 1
        except Exception:
            if (verbose):
                return 0
            else:
                raise Exception("CPU and GPU results do not match!")


if __name__ == "__main__":

    device = cuda.Device(0)
    device_name = device.name()
    print(f"*"*100)
    print(f"Using device: {device_name}")

    # Query resources available on the device
    max_threads_per_block = device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)
    max_shared_memory_per_block = device.get_attribute(cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)
    max_registers_per_block = device.get_attribute(cuda.device_attribute.MAX_REGISTERS_PER_BLOCK)

    print(f"Max threads per block: {max_threads_per_block}")
    print(f"Max shared memory per block (bytes): {max_shared_memory_per_block}")
    print(f"Max registers per block: {max_registers_per_block}")

    # Query multiprocessor information
    multiprocessor_count = device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
    print(f"Multiprocessors: {multiprocessor_count}")
    print(f"*"*100)


    print("Testing attention kernel")
    sizes = 2**np.arange(4, 12)
    d = 16
    attn = Attention(N=1, d=d, Q=None, K=None, V=None, m=None, l=None)
    for N in sizes:
        # context = pycuda.tools.make_default_context()
        attn.N = N
        Q, K, V = get_random_QKV(N, d)
        m, l = ml_args(N)
        # reinitalize attn module with updates N and d
        attn.reinit(N, d, Q, K, V, m, l)

        # print(attn.mod.get_module().get_ptx())
        
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

        