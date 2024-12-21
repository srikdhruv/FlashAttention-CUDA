/*
Simple kernels for computing Attention 

(1) Compute S = Q@K^T
(2) Compute P = Softmax(S)
(3) Compute O = P@V

*/

__device__ void displaySharedMem(float *X, int ydim, int xdim, int by){
    if(blockIdx.y==by && threadIdx.x==0 && threadIdx.y==0){
        // Use one thread to print shared mem contents (or any data) og block (bx, by)
        printf("Printing shape (%d, %d)\n",ydim,xdim);
        for(int i=0; i<ydim; i++){
            for(int j=0; j<xdim; j++){
                printf("@(%d, %d) -> (%f)\n",i,j,X[i*xdim + j]);
            }
        }
    }
}

__device__ void loadVecToShared(float *m, float *mi, int N){
    int row = threadIdx.y;
    int globalRow = blockDim.x*blockIdx.y + row;
    if(threadIdx.x == 0){
        if(globalRow<N){
            mi[row] = m[globalRow];
        }
        else{
            mi[row] = -INFINITY;
        }
    }
}

__device__ void loadMatToShared(float *S, float *Sij, int N){
    int row = threadIdx.y;
    int col = threadIdx.x;
    int globalRow = blockDim.y*blockIdx.y + row;
    int globalCol = blockDim.x*blockIdx.x + col;

    if(globalRow<N && globalCol<N){
        Sij[row*blockDim.x + col] = S[globalRow*N + globalCol];
    }   
    else{
        Sij[row*blockDim.x + col] = 0.0f;
    }
}

__global__ void matmulT(float *S, float *Q, float *K, int N, int d){
    int row = threadIdx.y;
    int col = threadIdx.x;
    int globalRow = blockDim.y*blockIdx.y + row;
    int globalCol = blockDim.x*blockIdx.x + col;

    float result = 0.0f;
    if(globalRow<N && globalCol<N){
        for(int i=0; i<d; i++){
            result += Q[globalRow*d + i]*K[globalCol*d + i];
        }
        // write to HBM
        S[globalRow*N + globalCol] = result;
    }
}


__global__ void findMax(float *S, float *m, int N){
    /*
    Process with 1D grid - no need to sync grid
    Process tiles across columns in a loop
    Parallelize tiles across along blockDim.y in 1D grid - (1, Tr, 1)
    */
    
    int row = threadIdx.y;
    // int col = threadIdx.x;
    int Bc = blockDim.x;
    int Br = blockDim.y;

    extern __shared__ float sm[];

    float *mi = sm;
    // float *li = sm + Br;
    float *mij_tilde = mi + Br;
    // float *lij_tilde = mij_tilde + Br;
    float *mi_new = mij_tilde + Br;
    // float *li_new = mi_new + Br;
    float *Sij = mi_new + Br;

    
    loadVecToShared(m, mi, N);
    // loadVecToShared(l, li, N);
    __syncthreads();

    // displaySharedMem(li, Br, 1, 0);
    int Tc = (N-1)/blockDim.x + 1;

    for(int i=0; i<Tc; i++){
            
        loadMatToShared(S, Sij, N);
        __syncthreads();

        if(threadIdx.x==0){
            // Use first col threads to compute row-max
            float maxval = -INFINITY;
            for(int j=0; j<blockDim.x; j++){
                maxval = Sij[row*Bc + j];
                maxval = (maxval > mij_tilde[row]) ? maxval : mij_tilde[row];
            }
            mij_tilde[row] = maxval;
        }
        __syncthreads();
        // if(threadIdx.x == 0){
        //     // Use first col threads to compute row-sum
        //     float rowsum = 0.0f;
        //     for(int k=0; k<blockDim.x; k++){
        //         rowsum += expf(Sij[row*Bc + k] - mij_tilde[row]);
        //     }
        //     lij_tilde[row] = rowsum;
        // }
        __syncthreads();
        mi_new[row] = max(mi[row], mij_tilde[row]);

        mi[row] = mi_new[row];
 
    
    }
    __syncthreads();
    
    if(threadIdx.x == 0 && blockDim.y*blockIdx.y + row<N){
 
        m[blockDim.y*blockIdx.y + row] = mi[row];
 
    }

}

__global__ void findSum(float *S, float *m, float *l, int N){

    int row = threadIdx.y;
    int Bc = blockDim.x;
    int Br = blockDim.y;

    extern __shared__ float sm[];
    float *li = sm;
    float *mi = li + Br;
    float *Sij = mi + Br;

    loadVecToShared(m, mi, N);
    loadVecToShared(l, li, N);
    __syncthreads();
    
    int Tc = (N-1)/blockDim.x + 1;
    for(int i=0; i<Tc; i++){
        __syncthreads();

        loadMatToShared(S, Sij, N);
        __syncthreads();

        if(threadIdx.x==0){
            // Use first col threads to compute row-max
            float sum = 0;
            for(int j=0; j<blockDim.x; j++){
                sum += expf(Sij[row*Bc + j] -mi[row] );
            }
            li[row] += sum;
        }
        __syncthreads();
    }

    if(threadIdx.x == 0 && blockDim.y*blockIdx.y + row<N){
    
        l[blockDim.y*blockIdx.y + row] = li[row];
 
    }

}

__global__ void Softmax(float*P, float *S, float *m, float *l, int N){
    
    int row = threadIdx.y;
    int col = threadIdx.x;
    int globalRow = blockDim.y*blockIdx.y + row;
    int globalCol = blockDim.x*blockIdx.x + col;


    float smax = 0.0f;
    if(globalRow<N && globalCol<N){
        smax = expf(S[globalRow*N + globalCol] - m[globalRow])/l[globalRow];
        P[globalRow*N + globalCol] = smax;
    }
}

__global__ void matmul(float *O, float *P, float *V, int N, int d){
    int row = threadIdx.y;
    int col = threadIdx.x;
    int globalRow = blockDim.y*blockIdx.y + row;
    int globalCol = blockDim.x*blockIdx.x + col;

    float result = 0.0f;
    if(globalRow<N && globalCol<d){
        for(int i=0; i<N; i++){
            result += P[globalRow*N + i]*V[i*d + globalCol];
        }
        // Write to HBM
        O[globalRow*d + globalCol] = result;   
    }

}



