import numpy as np
from scipy.special import softmax as sfmax

d = 1024
n_samples = 10
dataset = np.random.random((d, n_samples))


class attentionModule():
        
    def __init__(self, d, ffn_expansion=4):
        self.d = d # latent size of input
        self.ffn_expansion = ffn_expansion # expansion factor - hidden layer of MLP
        self.scalefactor = np.sqrt(d)
        self.initParams()

    def initParams(self):
        ## attention head parameters
        self.Wk = np.random.random((d,d))
        self.Wq = np.random.random((d,d))
        self.Wv = np.random.random((d,d))
        ## MLP parameters
        self.Wfc1 = np.random.random((self.d*self.ffn_expansion, self.d))
        self.Wfc2 = np.random.random((self.d, self.d*self.ffn_expansion))
        

    def loadWeights(self,W,type='k'):
        assert(np.shape(W)==(d,d))
        if(type=='k'):
            self.Wk = W
        elif(type=='v'):
            self.Wv = W
        elif(type=='q'):
            self.Wq = W
        elif(type=='fc1')
            assert(W.shape[1]==self.d*self.ffn_expansion and W.shape[0]==self.d)
            self.Wfc1 = W
        elif(type=='fc2'):
            assert(W.shape[0]==self.d*self.ffn_expansion and W.shape[1]=self.d)
            self.Wfc2 = W
        else:
            raise Exception(f"Type has to be {k,v,q} - entered values is {type}")
    
    def softmax(self, x):
        return sfmax(x, axis=0)
    
    def relu(self, x):
        x =  np.maximum(x, 0)
        return x


    def attentionPass(self, x):
        assert(x.shape[0] == d)
        K = self.Wk@x # d*N
        V = self.Wv@x # d*N
        Q = self.Wq@x # d*N

        # compute softmax
        S = self.softmax(K.T@Q) # N*N
        S = S/self.scalefactor
        # compute output
        O = V@S # d*N
        return O
    def mlpPass(self, x):
        x = self.Wfc1@x
        x = self.relu(x)
        x = self.Wfc2@x
        return x

    def forward(self, x):
        x = self.attentionPass(x)
        x = self.mlpPass(x)
        return x

class flashAttentionModule():
    def __init__(self):
        """
        Attributes for instance of deviceAdd module
        Includes kernel code and input variables.
        """
        # Compile the kernel code when an instance
        # of this class is made. This way it only
        # needs to be done once for the functions
        # you will call from this class.
        self.mod = self.getSourceModule()

    def getSourceModule(self):
        """
        Compiles Kernel in Source Module to be used by functions across the class.
        """
        # Define the tiled matrix multiplication kernel below
        kernelwrapper = """
        #define TILE_SIZE 32

        __global__ void MatrixMultiplyTiled(float *A, float *B, float *C, int M, int K, int N)
        {
            // Shared memory for tiles of A and B
            __shared__ float tileA[TILE_SIZE][TILE_SIZE];
            __shared__ float tileB[TILE_SIZE][TILE_SIZE];
        
            // Calculate row and column of C to work on
            int row = blockIdx.y * TILE_SIZE + threadIdx.y;
            int col = blockIdx.x * TILE_SIZE + threadIdx.x;
        
            float value = 0.0;
        
            // Loop over tiles of A and B
            for (int tileIdx = 0; tileIdx < (K + TILE_SIZE - 1) / TILE_SIZE; ++tileIdx)
            {
                // Load elements of A into shared memory
                if (row < M && tileIdx * TILE_SIZE + threadIdx.x < K)
                {
                    tileA[threadIdx.y][threadIdx.x] = A[row * K + tileIdx * TILE_SIZE + threadIdx.x];
                }
                else
                {
                    tileA[threadIdx.y][threadIdx.x] = 0.0;
                }
        
                // Load elements of B into shared memory
                if (col < N && tileIdx * TILE_SIZE + threadIdx.y < K)
                {
                    tileB[threadIdx.y][threadIdx.x] = B[(tileIdx * TILE_SIZE + threadIdx.y) * N + col];
                }
                else
                {
                    tileB[threadIdx.y][threadIdx.x] = 0.0;
                }
        
                __syncthreads();
        
                // Perform multiplication for this tile
                for (int k = 0; k < TILE_SIZE; ++k)
                {
                    value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
                }
        
                __syncthreads(); // Synchronize threads before loading the next tile
            }
        
            // Write the computed value to the output matrix
            if (row < M && col < N)
            {
                C[row * N + col] = value;
            }
        }
        """
        return SourceModule(kernelwrapper)


## simple output generation
x = dataset
attmod = attentionModule(d=d)
o = attmod.forward(x)
print(f'input shape = {x.shape}')
print(f'latent size = {attmod.d}')
print(f'output shape = {o.shape}')
        