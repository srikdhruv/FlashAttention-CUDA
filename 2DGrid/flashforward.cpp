// Forward Pass
#include <float.h>

__device__ void t_matmul(float *Sij, float *Qi, float *Kj, int Br, int d, int Bc)
{
    // each output of Sij would loop through row of Qi, col of Kj^T
    // Boundary / block overflow conditions not required since dimensions of Sij are Br x Bc
    
    int row = threadIdx.y;
    int col = threadIdx.x;
    float val = 0.0;

    for(int i=0; i<d; i++)
    {
        val = Qi[row*d + i] * Kj[col*d + i];    
    } 
    
    Sij[row * Bc + col] = val;
}

__device__ void matmul(float *C, float *A, float *B, int X, int Y, int Z)
{
    // Matrix multiplication with boundary conditions
    // Used when matrices could be bigger than block
    // C = A*B  | A: X x Y | B: Y x Z | C: X x Z
    int Br = blockDim.y;
    int Bc = blockDim.x;

    int row = threadIdx.y;
    int col = threadIdx.x;
    
    for(int local_row = row; local_row < X; local_row += Br)
    {
        for(int local_col = col; local_col < Z; local_col += Bc)
        {
            float val = 0.0;
            
            for(int i=0; i<Y; i++)
            {
                val = A[local_row * Y + i] * B[i * Z + local_col];    
            } 
            
            C[local_row * Z + local_col] = val;
        }
    }
}

/*
    FlashAttention Forward Pass Implementation
    Heterogeneous Computing Fall '24
    Columbia Engineering
*/

__global__ void flashattention(float *O, float *Q, float *K, float *V, float *mi, float *li, int d, int N)
{
    // define block and grid dimensions
    int Br = blockDim.y;
    int Bc = blockDim.x;
    int Tr = gridDim.y;
    int Tc = gridDim.x;

    // record indices - global and block
    int idx = threadIdx.x + blockIdx.x * Br;
    int idy = threadIdx.y + blockIdx.y * Bc;
    int i = blockIdx.y;
    int j = blockIdx.x;

    //Note, we only define one block of shared memory here
    extern __shared__ float shared_mem[];

    //This block of shared memory is used by all below shared memory variables
    float *Qi = shared_mem;
    float *Oi = shared_mem + Br * d;
    float *Kj = Oi + Br * d;
    float *Vj = Kj + Bc * d;
    float *Sij = Vj + Bc * d;
    float *Pij = Sij + Br*Bc;
    float *val_ij = Pij + Br*Bc;
    float *mij = val_ij + Br * d;
    
    // float *lij = mij + Br; lij not necessary since mi global calculated
    // define register variables
    // float *Sij_el;
    // float *Pij_el;
    // float *l_ij_el;
    // float *m_ij_el;
    // float *li_el;
    // float *mi_el;
    
    // load K, V into sm | ASSUME Br <= d
    // Bc x d; j*Bc x d -> {(j + 1)*Bc - 1} x d
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

    // mij is softmax local max variable
    if(threadIdx.x == 0)
    {
        mij[threadIdx.y] = -FLT_MAX;
    }
    __syncthreads();

    // thread index in block
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Compute Sij = Qi * Kj^T
    t_matmul(Sij, Qi, Kj, Br, d, Bc);
    __syncthreads();
    
    // Rowmax using atomicMax -> mij
    atomicMax((int*)&mij[row], __float_as_int(Sij[row * Bc + col]));
    __syncthreads();

    // Global rowmax using atomicMax: mij -> mi
    if(col == 0 && i*Br + row < N)
        atomicMax((int*)&mi[i*Br + row], __float_as_int(mij[row]));
    // __sync_grid();

    // Pij calculation | Pij: Br x Bc
    Pij[row * Bc + col] = expf(Sij[row * Bc + col] - mi[row]);
    __syncthreads();

    // Calcuclating Pij*Vj for Oi calculation, adding all Pij globally -> li
    matmul(val_ij, Pij, Vj, Br, Bc, d);
    if(i*Br + row < N)
        atomicAdd(&li[i*Br + row], Pij[row * Bc + col]);
    // __sync_grid();

    // Save calculated value to Oi after scaling by softmax denominator li
    // Boundary condition since Oi: Br x d (bigger than block)
    for(int local_col = col; local_col < d; local_col += Bc)
    {
        Oi[row * d + local_col] += val_ij[row * d + local_col]/li[row];
    }

    // Write back to global memory - atomicAdd since multiple blocks' Oi try to load to same O
    // Boundary condition since Oi: Br x d (bigger than block)
    for (int local_col = col; local_col < d; local_col += Bc)
    {
        if(i*Br + row < N)
            atomicAdd(&O[(i*Br + row) * d + local_col], Oi[row * d + local_col]);
    }
}

