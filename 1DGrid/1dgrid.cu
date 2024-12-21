#include<stdio.h>


__device__ void loadVecintoSharedMemory(float *xi, float* x, int N, int d, int mode){
    /*
    
    This device function loads xi into shared memory of corresponding block
    The shape of xi is assumed to be (Br x 1)

    mi, li \in Br x 1

    Two ways of loading 
    1. Use tx to load into shared memory -> more complex control, but coalesced memory access
    2. Use ty to load into shared memory -> less complex control, but uncoalesced memory access

    Using 1. here
   
    */

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int Br = blockDim.y;
    int Bc = blockDim.x;

    int globalindex = by*Br + ty;
    if(mode == 0){
        int phasesX = (Br-1)/Bc + 1;
        for(int px=0; px<phasesX; px++){

            int localindex = tx + px*Br;
            if(localindex<Br && ty==0){
                // Assume d>Br
                // Ensure only Br threads load into shared memory
                // Also we only need 1D array of threads to load, so we use ty==0 threads

                // Reduced control divergence because threads along x are used to load into shared memory
                if(globalindex<N){
                    xi[localindex] = x[globalindex];
                }
                else{
                    xi[localindex] = 0.0f;
                }
            }
        }
    }

    else if(mode == 1){
        int localindex = ty;
        if(tx==0){
            // Also we only need 1D array of threads to load, so we use tx==0 threads
            // use ty to read data from global to shared mem - uncoalesced read/write
            if(globalindex<N){
                xi[localindex] = x[globalindex];
            }
            else{
                xi[localindex] = 0.0f;
            }
        }
    }
}


__device__ void loadMatIntoSharedMemory(float *Xi, float *X, int N, int d, int groupIdx, int loadDirection){
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
   int by = blockIdx.y;

   int Br = blockDim.y;
   int Bc = blockDim.x;

   int Tr = gridDim.y;
   
   //blocksize is Br x Bc. So we need ceil(d/Bc) phases to load all elements into shared memory 
   // To load K, V we also need to consider whether Br<Bc. In that case we need phases along the
   // y direction to load all elements of K, V into shared memory.
   
    int phases_x = int((d-1)/Bc) + 1;

   if(loadDirection == 0){
        // load elements into shared memory in phases
        int phases_y = int((Bc-1)/Br) + 1;

        for(int py=0; py< phases_y; py++){
            for(int px=0; px<phases_x; px++){

                int localindex = (ty + py*Br)*d + (tx + px*Bc);
                int globalindex = (groupIdx*Bc + ty + py*Br)*d + (tx + px*Bc);

                // Ensure local indices are within the dimensions of Xi
                // Ensure global indices are within the dimensions of X

                if((ty + py*Br)<Bc && (tx + px*Bc)<d ){
                    if((groupIdx*Bc + ty + py*Br) < N){
                        // printf("Processed thread (%d, %d) -> glob row = (%d) -> value = (%f)\n", threadIdx.x, threadIdx.y, groupIdx*Bc + ty + py*Br, X[globalindex]);
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
        // load memory into shared memory in phases
        // No phases_y in this case - size of sm.y = Br = blockDim.y
        for(int px=0; px<phases_x; px++){
            
            int localindex = ty*d + (tx+px*Bc);
            int globalindex = (by*Br + ty)*d + (tx+px*Bc);

            // Ensure local indices are within the dimensions of Xi
            // Ensure global indices are within the dimensions of X
            if(tx+px*Bc<d){
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

__device__ void matmul(float *S, float *Q, float *K, int d){
    int row = threadIdx.y;
    int col = threadIdx.x;
    float res = 0.0f;
    for(int i=0; i<d; i++){
        res += Q[(threadIdx.y)*d + i]*K[(threadIdx.x)*d + i];
    }
    S[threadIdx.y*blockDim.x + threadIdx.x] = res;
}


__device__ void writeOToGlobalMem(float* O, float *Oi, int N, int d){
    int Br = blockDim.y;
    int Bc = blockDim.x;
    int row = threadIdx.y;
    int globalRow = blockIdx.y*Br + row;

    int phasesX = (d-1)/Bc + 1;
    for(int px=0; px<phasesX; px++){
        __syncthreads();// ensure coalesced writes
        int col = threadIdx.x + px*Bc;

        if(col<d && globalRow<N){
            O[globalRow*d + col] = Oi[row*Bc + col];
        }
    
    }
}

__device__ void writeMLToGlobalMem(float *m, float *mi, float *l, float *li, int N){
    int row = threadIdx.y;
    int globalRow = blockIdx.y*blockDim.y + row;
    if(globalRow<N){
        if(threadIdx.x==0){
            // update m
            m[globalRow] = mi[row];
        }
        if(threadIdx.x==1){
            // update l
            l[globalRow] = li[row];
        }
    }

}

__device__ void displaySharedMem(float *X, int ydim, int xdim, int by){
    if(blockIdx.y==by && threadIdx.x==0 && threadIdx.y==0){
        // Use one thread to print shared mem contents (or any data) og block (bx, by)
        printf("Printing shape (%d, %d)\n",ydim,xdim);
        for(int i=0; i<ydim; i++){
            for(int j=0; j<xdim; j++){
                // std::cout << s << " ";
                printf("@(%d, %d) -> (%f)\n",i,j,X[i*xdim + j]);
            }
        }
    }
}

__device__ void updateOi(float *Oi, float *Pij_tilde, float *Vj, 
    float *mi, float *li, float *mi_new, float * li_new,
                    float *mij_tilde, int N, int d){
    int Bc = blockDim.x;
    int phasesX = (d-1)/Bc + 1;
    int row = threadIdx.y;
    int globalRow = blockIdx.y*blockDim.y + row;
    if(globalRow<N){
        for(int px=0; px<phasesX; px++){
            int col = threadIdx.x + px*Bc;
            if(col<d){
                float pvResult = 0.0f;
                float oResult = 0.0f;
                for(int j=0; j<Bc; j++){
                    pvResult += Pij_tilde[row*Bc+j]*Vj[j*d + col];
                }
                oResult = li[row]*expf(mi[row] - mi_new[row])*Oi[row*d + col];
                oResult += expf(mij_tilde[row]-mi_new[row])*pvResult;
                oResult /= li_new[row];
                Oi[row*d + col] = oResult;       
            }
        }
    }
}

extern "C" __global__ void attn_forward(float *O, float *Q, float *K, float *V, float *m, float *l, int d, int N, int viewY, int viewX, int verbose){

    // 1D grid - compute Flash Attention
    // every row "element" of the grid corresponds to a (Qi, Oi) pair
    // we loop over (Kj, Vj) for block in grid
    // Then we continue to iterate and compute for j=0,Bc-1
    // block size = (Br, Bc)

    // define block and grid dimensions
    int Br = blockDim.y;
    int Bc = blockDim.x;
    int Tr = gridDim.y;
    int Tc = (N-1)/Bc+1;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //Note, we only define one block of shared memory here
    //This block of shared memory is used by all below shared memory variables
    extern __shared__ float shared_mem[];

    // Oi, Qi (load from HBM)
    float *Qi = shared_mem;
    float *Oi = Qi + Br * d;
    // Kj, Vj (load from HBM)
    float *Kj = Oi + Br * d;
    float *Vj = Kj + Bc * d;
    // mi, li (load from HBM)
    float *mi = Vj + Bc*d;
    float *li = mi + Br;
    // mi_new, li_new (computed per block)
    float *mi_new = li + Br;
    float *li_new = mi_new + Br;
    // Sij, Pij_tilde (computed per block)
    float *Sij = li_new + Br;
    float *Pij_tilde = Sij + Br*Bc;
    // shared mem mij_tilde, lij_tilde
    float* mij_tilde = Pij_tilde + Br*Bc;
    float* lij_tilde = mij_tilde + Br;

    // Begin FLASH ATTENTION!
    int row = threadIdx.y;
    int col = threadIdx.x;
    int idx = blockDim.y*blockIdx.y + ty; // grid (1, Tr, 1)
    // LOAD Qi, Oi (bx=0)
    loadMatIntoSharedMemory(Qi, Q, N, d, 0, 1);

    loadVecintoSharedMemory(mi, m, N, d, 1);
    loadVecintoSharedMemory(li, l, N, d, 1);
    __syncthreads();



    for(int j=0; j<Tc; j++){
        __syncthreads();

        loadMatIntoSharedMemory(Kj, K, N, d, j, 0);
        loadMatIntoSharedMemory(Vj, V, N, d, j, 0);
        __syncthreads();



        // find Sij
        matmul(Sij, Qi, Kj, d);
        __syncthreads();

        
        //find row-max mij
        // each rowmax computed by one thread
        float rowmax = -INFINITY;

        if(threadIdx.x==0){

            for(int i=0; i<Bc; i++){
                float currentVal = Sij[row*Bc + i];
                if( currentVal > rowmax ){
                    rowmax = currentVal;
                }
            }
            
            mij_tilde[row] = rowmax;

        }
        __syncthreads();
        

        Pij_tilde[row*blockDim.x + col] = expf( Sij[row*blockDim.x + col] - mij_tilde[row] );
        __syncthreads();
        // if(j==viewX && verbose)displaySharedMem(Pij_tilde, Br, Bc, viewY); 



        if(threadIdx.x==0){
            float rowsum = 0.0f;
            for(int i=0; i<Bc; i++){
                rowsum += Pij_tilde[row*Bc + i];
            }
            lij_tilde[row] = rowsum;


            mi_new[row] = max(mi[row], mij_tilde[row]);
            
        
            li_new[row] = expf(mi[row]-mi_new[row])*li[row] + 
                        expf(mij_tilde[row]-mi_new[row])*lij_tilde[row];

        }

            
        
        
        __syncthreads();

        // Update Output now
        updateOi(Oi, Pij_tilde, Vj, mi, li, mi_new, li_new, mij_tilde, N, d);

        __syncthreads();
        if(threadIdx.x==0){
            // Update li, mi
            li[row] = li_new[row];
            mi[row] = mi_new[row];
            
        }
        __syncthreads();

        writeOToGlobalMem(O, Oi, N, d);
        writeMLToGlobalMem(m, mi, l, li, N);

    }
    
    __syncthreads();

}