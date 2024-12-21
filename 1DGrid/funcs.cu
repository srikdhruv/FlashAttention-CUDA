/*

Test blockwise matrix multiplication
OPERATION => S = Q*K^t

Block size = (Br, Bc)
Inputs => Q, K of size (Br, d), (Bc, d)
Outputs => S of size (Br, Bc)

Description: Input a matrix with each block initialized with some values.
             Compute block-wise products with kernel
             Test with CPU code
*/
__global__ void matmul(float *S, float *Q, float *K, int d){
    int row = threadIdx.y;
    int col = threadIdx.x;
    float res = 0.0f;
    for(int i=0; i<d; i++){
        res += Q[(threadIdx.y)*d + i]*K[(threadIdx.x)*d + i];
    }
    S[threadIdx.y*blockDim.x + threadIdx.x] = res;
}


/*

Test blockwise matrix multiplication
OPERATION => PV = P*V

Block size = (Br, Bc)
Inputs => P, V of sizes (Br, Bc) & (Bc, d)
Outputs => PV of size (Br, d)

Description: Input a matrix with each block initialized with some values.
             Compute block-wise products with kernel
             Test with CPU code
*/

__global__ void PVmatmul(float* PV, float *P, float *V, int Bc, int d){

    int phases = ((d-1)/Bc);
    for (int p=0; p<phases; p++){
        // go through each phase
        // col range: p*Bc:(p+1)Bc
        int row = threadIdx.y;
        int col = threadIdx.x + p*Bc;
        float result = 0.0f;
        if(col < d){
            for(int i=0; i< Bc; i++){
                // go through all Bc elements 
                result += P[row*Bc + i]*V[i*d + col];
            }
        }
        PV[row*d + col] = result;
    }
}


/*
Test blockwise row-max
OPERATION => rowmax = row-max(Sij)

Block-size = (Br, Bc)
Inputs => rowmax, Sij of sizes (1) and (Br, Bc)
Outputs => row-max per row in Sij stored in row max of threadIdx.x = 0
*/
__global__ float rowMaximum(float *Sij){
    float rowmax=-INFINITY;
    return rowmax
}


/*
Test block-wise softmax
OPERATION => P = Softmax(S) / m is row-wise max of S

Block-size = (Br, Bc)
Inputs => S, m of sizes (Br, Bc), (Br, 1)
Outputs => P of size (Br, Bc)
*/

__global__ blockSoftmax(float *m, float *S){

}

/*
Test blockwise row-sum
OPERATION => l = rowsum(P)

Block-size = (Br, Bc)
Inputs => P of size (Br, Bc)
Outputs => l of size (Br, 1)
*/

__global__ rowSum(float *l, float *S){

}

/*
Tets blockwise new-max
OPERATION => mi_new = max(mi, mij_tilde) - elementwise

Block-size = (Br, Bc)
Inputs => mi, mij_tilde of sizes (Br, 1) each
Outputs => mi_new of size (Br, 1)
*/

__global__ void m_newMax(float *mi_new, float *mi, float *mij_tilde){

}


/*
Tets blockwise new-max l
OPERATION => li_new = max(li, li_tilde) - elementwise

Block-size = (Br, Bc)
Inputs => li, li_tilde of sizes (Br, 1) each
Outputs => li_new of size (Br, 1)
*/

__global__ void l_newMax(float *li_new, float *li, float *lij_tilde){

}

/*
Tests update of Output
OPERATION => Oi <- Oi ... expression update

Block-size = (Br, Bc)
Inputs => li_new, li, mi_new, mi, mij_tilde, Oi, PV
*/

__global__ void updateO(){

}


/*
Tests update to global memory
OPERATION => li <- li_new, mi <- mi_new

Block-size = (Br, Bc)
Inputs => li, mi, li_new, mi_new
Outputs => updated values @ li, mi
*/

