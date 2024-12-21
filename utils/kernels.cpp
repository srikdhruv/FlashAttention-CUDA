/*
Init pseudo random number generator state - store to HBM
Set block sizes Br, Bc (host code)
Initialize O=0, l=0, m=0 in HBM (host code)

Divide Q into Tr blocks, K,V into Tc blocks (grid shape inovked by host code)
Divide O into Tr blocks, l,m into Tr blocks 

Now, the algorithm begins computing the result for each (Qi,Oi), (Kj,Vj) pair - each is a block in CUDA

Things to load into SRAM (shared memory):
- Kj, Vj, Qi, Oi, li, mi

Things to compute:
- Sij = tau*Qi*Kj.T
- mij_tilde = rowmax(Sij)
- Pij_tilde = exp(Sij - mij_tilde) 
- lij_tilde = rowsum(Pij_tilde)

- mi_new = max(mi, mij_tilde)
- li_new = exp(mi-mi_new)*li + exp(mij_tilde-mi_new)*lij_tilde

Now, we computed all required results. We now update values Oi, li, mi.
*/

__global__ void flashattention(float *X, float *Q, float *K, float *V, float *O, int d,int N,
 float *mi, float *li){
    
    // define block and grid dimensions
    int Br = blockDim.y;
    int Bc = blockDim.x;
    int Tr = gridDim.y;
    int Tc = gridDim.x;

    //Note, we only define one block of shared memory here
    //This block of shared memory is used by all below shared memory variables
    extern __shared__ float shared_mem[];

    float *Qi = shared_mem;
    float *Oi = Qi + Br * d;
    float *Kj = Oi + Br * d;
    float *Vj = Kj + Bc * d;
    float *mi_new = Vj + Bc * d;
    float *li_new = mi_new + Br;
    float *mij_buffer = li_new + Br;

    // define register variables
    float *Sij;
    float *Pij_tilde;

    float *lij_tilde;
    float *mij_tilde;

    


    // load K, V into sm
    // load Q, O into sm
    loadMatIntoSharedMemory(Kj, K, N, d, 0);
    loadMatIntoSharedMemory(Vj, V, N, d, 0);
    loadMatIntoSharedMemory(Qi, Q, N, d, 1);
    loadMatIntoSharedMemory(Oi, O, N, d, 1);
    // load m, l into sm
    loadVecintoSharedMemory(mi, mi, N, 1);
    loadVecintoSharedMemory(li, li, N, 1);
    // synchronize to ensure all data is loaded
    __syncthreads();

    // compute Sij
    Sij = matmul(Qi, Kj, d);

    // update mij = rowmax

    // Use atomic opearations and update rowmax of each Sij row into corresponding index of mij

    __syncthreads(); // redundant check

    
    Pij_tilde = exp(Sij - mij)

    // update lij = rowmax using atomic operations
    
    // update mi_new, li_new

    // Compute and load Oi

    // update mi<-mi_new, li<-li_new

}

__device__ void findMaxSij(float *Sij, float *mij_tilde, float * mij_buffer, int Br){
    /*
    Finds the maximum value of an array using reduction techniques
    */
   int tx = threadIdx.x;
   int ty = threadIdx.y;
   int stride = 0;
   for(stride; stride<Br; stride*=2){
       if((tx%stride)==0){
           mij_buffer[ty] = max(Sij[ty*Br]); 
       }    
   }
   
   
   
}
    

__device__ float matmul(float *Qi, float *Kj, int d){
    // This computes stuff within a block
    // naive matrix multiplication
    // each output of Sij would loop through row of Qi, col of Kj^T

    int row = threadIdx.y;
    int col = threadIdx.x;

    int Br = blockDim.y;
    float val = 0.0;

    for(int i=0; i<d; i++){
        val = Qi[row*d + i]*Kj[i*d + col];    
    } 
    
    return val;
}


__device__ printSMvalues(float *Xi, int xrows, int xcols,int d, int N, int bidx, int bidy){
    /*
    Debug function to check if SM values are correctly loaded

    Choose a specific block (bidx, bidy) and threads (tx, ty)==(0,0) 
    to print required matrix in order
    */

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    if(bx==bidx && by==bidy){
        if(tx==0 && ty==0){
            for(int i=0; i<xrows; i++){
                for(int j=0; j<xcols; j++){
                    printf("X[%d][%d]: %f\n", i, j, Xi[i*xcols + j]);
                }
            }
        }
    }


}


__device__ loadVecintoSharedMemory(float *xi, float x, int N, int d){
    /*
    
    This device function loads xi into shared memory of corresponding block
    The shape of xi is assumed to be (Br x 1)

    mi, li \in Br x 1

    Two ways of loading 
    1. Use tx to load into shared memory -> more complex control, but coalesced memory access
    2. Use ty to load into shared memory -> less complex control, but less coalesced memory access

    Using 1. here
   
    */

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int Br = blockDim.y;
    int Bc = blockDim.x;

    int globalindex = by*Br + ty;
    int localindex = tx;
    if(localindex<Br && ty==0){
        // Ensure only Br threads load into shared memory
        // Also we only need 1D array of threads to load, so we use ty==0 threads

        // Reduced control divergence because threads along x are used to load into shared memory
        if(globalindex<N){
            xi[localindex] = x[globalindex];
        }
    }

}

__device__ loadMatIntoSharedMemory(float *Xi, float *X, int N, int d, int loadDirection){
    /*
    
    This device function loads Xi into shared memory of corresponding block
    The shape of X | (Bc x d) if loadDirection is 0 | (Br x d) if loadDirection is 1
    If loadDirection is 0, it loads Xi in each block -> (blockIdx.x, i) (for Kj, Vj)
    If loadDirection is 1, it loads Xi in each block -> (i, blockIdx.y) (for Oi, Qi)

    Kj, Vj \in Bc x d
    Oi, Qi \in Bc x d
    mi, li \in Br x 1


    */
   int tx = threadIdx.x;
   int ty = threadIdx.y;
   int bx = blockIdx.x;
   int by = blockIdx.y;

   int Br = blockDim.y;
   int Bc = blockDim.x;
   
   //blocksize is Bc x Br. So we need ceil(d/Br) phases to load all elements into shared memory 
   // To load K, V we also need to consider whether Bc<Br. In that case we need phases along the
   // y direction to load all elements of K, V into shared memory.
   
    int phases_x = int((d-1)/Br) + 1;

   if(loadDirection == 0){
        // load elements into shared memory in phases
        int phases_y = int((d-1)/Bc) + 1;

        for(int py=0; py< phases_y; py++){
            for(int px=0; px<phases_x; px++){

                int localindex = (ty + py*Bc)*d + (tx + px*Br);
                int globalindex = (bx*Br + ty + py*Bc)*d + (tx + px*Br);

                // Ensure local indices are within the dimensions of Xi
                // Ensure global indices are within the dimensions of X
                // (bx*Br + ty + py*Bc) < (bx+1)Br < Tr*Br <= N + 1;

                if((ty + py*Bc)<Br && (tx + px*Br)<d ){
                    if((bx*Br + ty + py*Bc) < N){
                        Xi[localindex] = X[globalindex];
                    }
                    else{
                        Xi[localindex] = 0.0;
                    }
                }
            }
        }

   }

   else if (loadDirection == 1){
        // load memort into shared memory in phases
        // No phases_y in this case - size of sm.y = Br = blockDim.y
        for(int px=0; px<phases_x; px++){
            
            int localindex = ty*d + (tx+px*Br);
            int globalindex = (by*Br + ty)*d + (tx+px*Br);

            // Ensure local indices are within the dimensions of Xi
            // Ensure global indices are within the dimensions of X
            if(tx+px*Br<d){
                if((by*Br + ty) < N){
                    Xi[localindex] = X[globalindex];
                }
                else{
                    Xi[localindex] = 0.0;
                }
            }
        }   
   }
}

