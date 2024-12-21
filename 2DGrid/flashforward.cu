// Forward Pass

__global__ void flashattention(float *X, float *Q, float *K, float *V, float *O, int d, int N, float *mi, float *li)
{
    
    // define block and grid dimensions
    int Br = blockDim.y;
    int Bc = blockDim.x;
    int Tr = gridDim.y;
    int Tc = gridDim.x;

    int idx = threadIdx.x + blockIdx.x * Br;
    int idy = threadIdx.y + blockIdx.y * Bc;
    i = blockIdx.x;
    j = blockIdx.y;

    //Note, we only define one block of shared memory here
    extern __shared__ float shared_mem[];

    //This block of shared memory is used by all below shared memory variables
    float *Qi = shared_mem;
    float *Oi = shared_mem + Br * d;
    float *Kj = Oi + Br * d;
    float *Vj = Kj + Bc * d;

    float *mij = Vj + Bc * d;
    float *lij = mij + Br;

    float *mi_new = lij + Br;
    float *li_new = mi_new + Br;

    // define register variables
    float *Sij_el;
    float *Pij_el;

    float *l_ij_el;
    float *m_ij_el;
    float *li_el;
    float *mi_el;
    
    // load K, V into sm | ASSUME Br <= d
    // Bc x d; j*Bc x d -> (j + 1)*Bc - 1 x d
    for(int local_row = threadIdx.x, local_col = threadIdx.y; local_col < d; local_col += Br)
    {
        if(j*Bc + local_row < N)
        {
            Kj[local_row*d + local_col] = K[(j*Bc + local_row) * d + local_col];
            Vj[local_row*d + local_col] = V[(j*Bc + local_row) * d + local_col];
        }
        else
        {
            Kj[local_row*d + local_col] = 0;
            Vj[local_row*d + local_col] = 0;
        }   
    }
    
    // load Q into sm | ASSUME Bc <= d
    for(int local_row = threadIdx.y, local_col = threadIdx.x; local_col < d; local_col += Bc)
    {
        if(i*Br + local_row < N)
        {
            Qi[local_row*d + local_col] = Q[(i*Br + local_row) * d + local_col];
            Oi[local_row*d + local_col] = 0;
        }
        else
        {
            Qi[local_row*d + local_col] = 0;
            Oi[local_row*d + local_col] = 0;
        }   
    }
    __syncthreads();

    //Sij_el = matmul(Qi, Kj, d);

    // update mij = rowmax
    // Use atomic opearations and update rowmax of each Sij row into corresponding index of mij

    //__syncthreads(); // redundant check

    // Step 1: Compute Sij = Qi * Kj^T
    for (int row = threadIdx.y; row < Br; row += blockDim.y)
    {
        for (int col = threadIdx.x; col < Bc; col += blockDim.x)
        {
            Sij = 0.0;
            for (int k = 0; k < d; ++k)
            {
                Sij += Qi[row * d + k] * Kj[col * d + k];
            }

            // Update mij (rowmax) and compute exp(Sij - mij)
            atomicMax((int*)&mij[row], __float_as_int(Sij)); // Atomic rowmax
            __syncthreads();

            exp_term = expf(Sij - mij[row]);
            atomicAdd(&l_ij, exp_term);
            atomicAdd(&Oi[row * d], exp_term * Vj[col * d]); // Weighted update
        }
    }
    __syncthreads();
    
    //Pij_el = exp(S-j_el - mij)

    // update lij = rowmax using atomic operations
    
    // update mi_new, li_new

    // Compute and load Oi

    // update mi<-mi_new, li<-li_new

    // Step 2: Update mi_new and normalize
    for (int row = threadIdx.y; row < Br; row += blockDim.y)
    {
        mi_new[row] = fmaxf(mij[row], m_ij);
        li_new[row] = expf(mij[row] - mi_new[row]) * l_ij;

        Oi[row * d] = (Oi[row * d] / li_new[row]); // Normalize Oi
    }

    // Write back to global memory
    for (int row = threadIdx.y; row < Br; row += blockDim.y)
    {
        for (int col = threadIdx.x; col < d; col += blockDim.x)
        {
            O[(i * Br + row) * d + col] = Oi[row * d + col];
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

    for(int i=0; i<d; i++)
    {
        val = Qi[row*d + i]*Kj[i*d + col];    
    } 
    
    return val;
}

