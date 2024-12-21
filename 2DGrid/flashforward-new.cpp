// Forward Pass
#include <float.h>

/*
    FlashAttention Forward Pass Implementation
    Heterogeneous Computing Fall '24
    Columbia Engineering
*/

__device__ void t_matmul(float *Sij, float *Qi, float *Kj, int Br, int d, int Bc)
{
    // each output of Sij would loop through row of Qi, col of Kj^T
    // Boundary / block overflow conditions not required since dimensions of Sij are Br x Bc
    
    int row = threadIdx.y;
    int col = threadIdx.x;
    float val = 0.0;

    for(int i=0; i<d; i++)
    {
        val += Qi[row*d + i] * Kj[col*d + i];    
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
            float v = 0.0;
            
            for(int k=0; k<Y; k++)
            {
                v += A[local_row * Y + k] * B[k * Z + local_col];    
            } 
            
            C[local_row * Z + local_col] = v;
        }
    }
}

/*
    SOFTMAX NUMERATOR
    
    Input: Q, K, (N, d)
    Operations:
    
        SM in: Qi, Kj; mij, Sij
        
        Sij <- Qi*Kj^T
        mij <- rowmax(Sij)
        mi <- global rowmax(mij)
        
        SM out: Sij
        
    Output: Sij_glob, mi
*/

__global__ void softmax_num(float *Sij_glob, float *Q, float *K, float *mi, int d, int N)
{
    // define block and grid dimensions
    int Br = blockDim.y;
    int Bc = blockDim.x;
    int Tr = gridDim.y;
    int Tc = gridDim.x;

    // record indices - global and block
    int idx = threadIdx.x + blockIdx.x * Bc;
    int idy = threadIdx.y + blockIdx.y * Br;
    int i = blockIdx.y;
    int j = blockIdx.x;

    // thread index in block
    int row = threadIdx.y;
    int col = threadIdx.x;

    //Note, we only define one block of shared memory here
    extern __shared__ float shared_mem[];

    //This block of shared memory is used by all below shared memory variables
    float *Qi = shared_mem;
    float *Kj = Qi + Br * d;
    float *Sij = Kj + Bc * d;
    float *mij = Sij + Br * Bc;
    
    // load K, V into sm | ASSUME Br <= d
    // Bc x d; j*Bc x d -> {(j + 1)*Bc - 1} x d
    for(int local_row = threadIdx.x, local_col = threadIdx.y; local_col < d; local_col += Br)
    {
        if(j*Bc + local_row < N)
        {
            Kj[local_row*d + local_col] = K[(j*Bc + local_row) * d + local_col];
            // printf("%d : %f\n", (j*Bc + local_row) * d + local_col, K[(j*Bc + local_row) * d + local_col]);
        }
        else
        {
            Kj[local_row*d + local_col] = 0;
        }   
    }
    
    // load Q into sm | ASSUME Bc <= d
    for(int local_row = threadIdx.y, local_col = threadIdx.x; local_col < d; local_col += Bc)
    {
        if(i*Br + local_row < N)
        {
            Qi[local_row*d + local_col] = Q[(i*Br + local_row) * d + local_col];
        }
        else
        {
            Qi[local_row*d + local_col] = 0;
        }   
    }

    // mij is softmax local max variable
    if(threadIdx.x == 0)
    {
        mij[threadIdx.y] = -FLT_MAX;
    }
    __syncthreads();

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
    
    // Move Sij to global
    Sij_glob[idy * (Tc*Bc) + idx] = Sij[row * Bc + col];
}




/*
    SOFTMAX DENOMINATOR
    
    Input: Sij_glob, V, mi, (N, d)
    Operations:
    
        SM in: Sij, Vj; Pij, val_ij
        
        Pij <- e^(Sij - mi)
        val_ij <- Pij*Vj
        li <- global rowsum(Pij)
        
        SM out: val_ij
        
    Output: val_ij_glob, li
*/

__global__ void softmax_denom(float *val_ij_glob, float *Sij_glob, float *V, float *mi, float *li, int d, int N)
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

     // thread index in block
     int row = threadIdx.y;
     int col = threadIdx.x;

    //Note, we only define one block of shared memory here
    extern __shared__ float shared_mem[];

    //This block of shared memory is used by all below shared memory variables
    // float *Qi = shared_mem;
    // float *Oi = shared_mem + Br * d;
    // float *Kj = Oi + Br * d;
    float *Vj = shared_mem;
    float *Sij = Vj + Bc * d;
    float *Pij = Sij + Br * Bc;
    float *val_ij = Pij + Br * Bc;
    
     // load K, V into sm | ASSUME Br <= d
     // Bc x d; j*Bc x d -> {(j + 1)*Bc - 1} x d
     for(int local_row = threadIdx.x, local_col = threadIdx.y; local_col < d; local_col += Br)
     {
         if(j*Bc + local_row < N)
         {
             Vj[local_row*d + local_col] = V[(j*Bc + local_row) * d + local_col];
         }
         else
         {
             Vj[local_row*d + local_col] = 0;
         }   
     }
    
    // Load Sij to shared memory
    Sij[row * Bc + col] = Sij_glob[idy * (Tc*Bc) + idx];
    __syncthreads();
    
    // if(row == 0 && col == 0)
    //     printf("%d : %f\n", row * d + col, V[row * d + col]);

    // Pij calculation | Pij: Br x Bc
    Pij[row * Bc + col] = expf(Sij[row * Bc + col] - mi[row]);

    // Calcuclating Pij*Vj for Oi calculation, adding all Pij globally -> li
    matmul(val_ij, Pij, Vj, Br, Bc, d);
    
    if(i*Br + row < N)
        atomicAdd(&li[i*Br + row], Pij[row * Bc + col]);
    // __sync_grid();
    
    // Save val_ij to global
    for(int local_row = row, local_col = col; local_col < d; local_col += Bc)
    {
        if(i*Br + local_row < N)
        {
            val_ij_glob[(i*Br + local_row) * d + local_col] = val_ij[local_row*d + local_col];
        }
    }
}

/*
    ATTENTION OUTPUT
    
    Input: val_ij_glob, li, (N, d)
    Operations:
        O <- val_ij/li
    Output: O
*/

__global__ void storeTo_O(float *O, float *val_ij_glob, float *li, int d, int N)
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

    // thread index in block
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    for(int local_row = threadIdx.y, local_col = threadIdx.x; local_col < d; local_col += Bc)
    {
        if(i*Br + local_row < N)
        {
            O[(i*Br + local_row) * d + local_col] = val_ij_glob[(i*Br + local_row) * d + local_col]/li[(i*Br + local_row)];
        }
    }
}

