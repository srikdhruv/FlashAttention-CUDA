/*
Variants of Matrix Multiplication Kernels
*/
#include <float.h>
__constant__ float MAX_VALUE = 3.4028235e+38;

__global__ void sharedMemKernel(float *C, float *A, float *B, int M, int K, int N);

__global__ void naive(float *C, float *A, float *B, int M, int K, int N){
    /*
    Naive matmul beween A(M*K) and B(K*N) to give C (M*N)
    */

    // Tiles are of size blockDim.y * blockDim.x
    // Each thread computes one element of C
    
    // local row, col - block(output tile)
    int row = threadIdx.y;
    int col = threadIdx.x;

    // global row, col - output matrix
    int globalRow = blockIdx.y*blockDim.y + row;
    int globalCol = blockIdx.x*blockDim.x + col;

    float result = 0.0f;
    if(globalRow<M && globalCol<N){
        for(int i=0; i<K; i++){
            // reads from A are coalesced
            // reads from B are not coalesced
            
            result += A[globalRow*K + i]*B[i*N + globalCol];
        }
        // writes to C are coalesced
        C[globalRow*N + globalCol] = result;
    }
}


__device__ void loadIntoSharedMemoryColPhase(float *Ashared, float *A, int colTileIdx, int ArowSize, int AcolSize){
    /*
    Load data from input A into shared memory Ashared - of size (blockDim.y, blockDim.x)
    Assume that thread block size is same as Ashared size
    Phases move across columns of input matrix A
    */
    const unsigned int RowIdx =  blockIdx.y*blockDim.y + threadIdx.y;
    const unsigned int ColIdx =  colTileIdx*blockDim.x + threadIdx.x;
    
    if(RowIdx<ArowSize && ColIdx<AcolSize) {
        Ashared[threadIdx.y*blockDim.x + threadIdx.x] = A[RowIdx*AcolSize + ColIdx];
    }
    else{
        Ashared[threadIdx.y*blockDim.x + threadIdx.x] = 0.0f;
    }
    // Debug outptus
    // if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0){
    //     printf("loaded value(%f) from A[%d][%d]\n",A[RowIdx*AcolSize + ColIdx],RowIdx,ColIdx);
    // }
}

__device__ void loadIntoSharedMemoryRowPhase(float *Ashared, float *A, int rowTileIdx, int ArowSize, int AcolSize){
    /*
    Load data from input A into shared memory Ashared - of size (blockDim.y, blockDim.x)
    Assume that thread block size is same as Ashared size
    Phases move across rows of input matrix A
    */
    const unsigned int RowIdx =  rowTileIdx*blockDim.y + threadIdx.y;
    const unsigned int ColIdx =  blockIdx.x*blockDim.x + threadIdx.x;
    
    if(RowIdx<ArowSize && ColIdx<AcolSize) {
        Ashared[threadIdx.y*blockDim.x + threadIdx.x] = A[RowIdx*AcolSize + ColIdx];
    }
    else{
        Ashared[threadIdx.y*blockDim.x + threadIdx.x] = 0.0f;
    }
    // Debug outptus
    // if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0){
    //         printf("loaded value(%f) from B[%d][%d]\n",A[RowIdx*AcolSize + ColIdx],RowIdx,ColIdx);
    //     }
}

__global__ void sharedMemKernel(float *C, float *A, float *B, int M, int K, int N){
    /*
    Matmul between A(M*K) and B(K*N) to give C (M*N)
    Used shared memory optimization
    Each blocks loads required input tiles into shared memory
    computations are performed by reading from shared memory
    Final writes from threads are into global memory
    Assume square tiles of size blockDim.x = blockDim.y
    */
    extern __shared__ float sharedMem[];
    
    int Bsize = blockDim.x;
    
    // load inputs A,B in shared memory in phases
    // Assume that block is square
    
    // int phases = (int)ceil((float)K / Bsize);
    int phases = (K + Bsize - 1)/Bsize;

    float *CTile = sharedMem;
    float *ATile = CTile + Bsize*Bsize;
    float *BTile = ATile + Bsize*Bsize;

    // Ensure that shared memory allocated is enough to store all matrices
    size_t sharedMemSize = 3 * Bsize * Bsize * sizeof(float); 
    assert((char*)(BTile + Bsize * Bsize) - (char*)sharedMem <= sharedMemSize);


    // ensure CTile is all zeros
    CTile[threadIdx.y * Bsize + threadIdx.x] = 0.0f;
    __syncthreads();
    
    
    for(int p=0; p<phases; p++){
        //load A, B into shared memory
        
        loadIntoSharedMemoryColPhase(ATile, A, p, M, K);
        loadIntoSharedMemoryRowPhase(BTile, B, p, K, N);

        __syncthreads();

        float result = 0.0f;
        for(int i=0; i<Bsize; i++){
            //compute phase-wise matrix multiplication
            result += ATile[(threadIdx.y)*Bsize + i]*BTile[i*Bsize + threadIdx.x];
        }
        // Write result to output
        if (result > MAX_VALUE) {
            printf("Max value!!\n");
            result = MAX_VALUE;
        }
        CTile[threadIdx.y*Bsize + threadIdx.x] += result;
    } 
    if( (blockIdx.y*Bsize + threadIdx.y)<M && blockIdx.x*Bsize+threadIdx.x<N){
        C[(blockIdx.y*Bsize + threadIdx.y)*N + blockIdx.x*Bsize+threadIdx.x] = CTile[threadIdx.y*Bsize + threadIdx.x];
    }

//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         printf("CUDA error: %s\n", cudaGetErrorString(err));
// }

}


__global__ void sharedMemTransposeKernel(float *C, float *A, float *B, int M, int K, int N){
    /*
    Matmul between A(M*K) and B.T(K*N) to give C (M*N) [B(N*K)]
    Used shared memory optimization
    Each blocks loads required input tiles into shared memory
    computations are performed by reading from shared memory
    Final writes from threads are into global memory
    Assume square tiles of size blockDim.x = blockDim.y
    */
    
    extern __shared__ float sharedMem[];
    
    int Bsize = blockDim.x;
    
    // load inputs A,B in shared memory in phases
    // Assume that block is square
    
    int phases = (int)ceil((float)K / Bsize);

    float *CTile = sharedMem;
    float *ATile = CTile + Bsize*Bsize;
    float *BTile = ATile + Bsize*Bsize;

    // Ensure that shared memory allocated is enough to store all matrices
    size_t sharedMemSize = 3 * Bsize * Bsize * sizeof(float); 
    assert((char*)(BTile + Bsize * Bsize) - (char*)sharedMem <= sharedMemSize);


    // ensure CTile is all zeros
    CTile[threadIdx.y * Bsize + threadIdx.x] = 0.0f;
    __syncthreads();
    
    if( (blockIdx.y*Bsize + threadIdx.y)<M && blockIdx.x*Bsize+threadIdx.x<N){
        for(int p=0; p<phases; p++){
            //load A, B into shared memory
            
            loadIntoSharedMemoryColPhase(ATile, A, p, M, K);
            loadIntoSharedMemoryColPhase(BTile, B, p, K, N);

            __syncthreads();
    
            float result = 0.0f;
            for(int i=0; i<Bsize; i++){
                //compute phase-wise matrix multiplication
                result += ATile[(threadIdx.y)*Bsize + i]*BTile[(threadIdx.y)*Bsize + i];
            }
            // Write result to output
            if (result > MAX_VALUE) {
                printf("Max value!!\n");
                result = MAX_VALUE;
            }
            CTile[threadIdx.y*Bsize + threadIdx.x] += result;
        } 
    
        C[(blockIdx.y*Bsize + threadIdx.y)*N + blockIdx.x*Bsize+threadIdx.x] = CTile[threadIdx.y*Bsize + threadIdx.x];
    }
}


// __global__ void tileMatmulTransposed(float *C, float *A, float *B, )