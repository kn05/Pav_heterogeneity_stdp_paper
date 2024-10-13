#include "definitionsInternal.h"
#include <iostream>
#include <random>
#include <cstdint>

struct MergedNeuronInitGroup0
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    float* sT;
    float* prevST;
    curandState* rng;
    scalar* V;
    scalar* U;
    float* inSynInSyn0;
    float* inSynInSyn1;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronInitGroup1
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    unsigned int* spkCntEvnt;
    unsigned int* spkEvnt;
    float* sT;
    float* seT;
    float* prevST;
    float* prevSET;
    curandState* rng;
    scalar* V;
    scalar* U;
    scalar* D;
    float* inSynInSyn0;
    float* inSynInSyn1;
    unsigned int numNeurons;
    
}
;
struct MergedSynapseConnectivityInitGroup0
 {
    unsigned int* rowLength;
    uint16_t* ind;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    unsigned int rowStride;
    
}
;
struct MergedSynapseConnectivityInitGroup1
 {
    unsigned int* rowLength;
    uint8_t* ind;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    unsigned int rowStride;
    
}
;
struct MergedSynapseConnectivityInitGroup2
 {
    unsigned int* rowLength;
    uint8_t* ind;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    unsigned int rowStride;
    
}
;
struct MergedSynapseConnectivityInitGroup3
 {
    unsigned int* rowLength;
    uint16_t* ind;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    unsigned int rowStride;
    
}
;
struct MergedSynapseSparseInitGroup0
 {
    scalar* tauMinus;
    scalar* c;
    scalar* g;
    scalar* wMax;
    scalar* wMin;
    scalar* aMinus;
    scalar* aPlus;
    scalar* tauD;
    scalar* tauC;
    scalar* tauPlus;
    unsigned int* remap;
    unsigned int* colLength;
    uint8_t* ind;
    unsigned int* rowLength;
    unsigned int numSrcNeurons;
    unsigned int colStride;
    unsigned int rowStride;
    unsigned int numTrgNeurons;
    
}
;
struct MergedSynapseSparseInitGroup1
 {
    scalar* tauMinus;
    scalar* c;
    scalar* g;
    scalar* wMax;
    scalar* wMin;
    scalar* aMinus;
    scalar* aPlus;
    scalar* tauD;
    scalar* tauC;
    scalar* tauPlus;
    unsigned int* remap;
    unsigned int* colLength;
    uint16_t* ind;
    unsigned int* rowLength;
    unsigned int numSrcNeurons;
    unsigned int colStride;
    unsigned int rowStride;
    unsigned int numTrgNeurons;
    
}
;
__device__ __constant__ MergedNeuronInitGroup0 d_mergedNeuronInitGroup0[1];
void pushMergedNeuronInitGroup0ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, float* sT, float* prevST, curandState* rng, scalar* V, scalar* U, float* inSynInSyn0, float* inSynInSyn1, unsigned int numNeurons) {
    MergedNeuronInitGroup0 group = {spkCnt, spk, sT, prevST, rng, V, U, inSynInSyn0, inSynInSyn1, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronInitGroup0, &group, sizeof(MergedNeuronInitGroup0), idx * sizeof(MergedNeuronInitGroup0)));
}
__device__ __constant__ MergedNeuronInitGroup1 d_mergedNeuronInitGroup1[1];
void pushMergedNeuronInitGroup1ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, unsigned int* spkCntEvnt, unsigned int* spkEvnt, float* sT, float* seT, float* prevST, float* prevSET, curandState* rng, scalar* V, scalar* U, scalar* D, float* inSynInSyn0, float* inSynInSyn1, unsigned int numNeurons) {
    MergedNeuronInitGroup1 group = {spkCnt, spk, spkCntEvnt, spkEvnt, sT, seT, prevST, prevSET, rng, V, U, D, inSynInSyn0, inSynInSyn1, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronInitGroup1, &group, sizeof(MergedNeuronInitGroup1), idx * sizeof(MergedNeuronInitGroup1)));
}
__device__ __constant__ MergedSynapseConnectivityInitGroup0 d_mergedSynapseConnectivityInitGroup0[1];
void pushMergedSynapseConnectivityInitGroup0ToDevice(unsigned int idx, unsigned int* rowLength, uint16_t* ind, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int rowStride) {
    MergedSynapseConnectivityInitGroup0 group = {rowLength, ind, numSrcNeurons, numTrgNeurons, rowStride, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedSynapseConnectivityInitGroup0, &group, sizeof(MergedSynapseConnectivityInitGroup0), idx * sizeof(MergedSynapseConnectivityInitGroup0)));
}
__device__ __constant__ MergedSynapseConnectivityInitGroup1 d_mergedSynapseConnectivityInitGroup1[1];
void pushMergedSynapseConnectivityInitGroup1ToDevice(unsigned int idx, unsigned int* rowLength, uint8_t* ind, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int rowStride) {
    MergedSynapseConnectivityInitGroup1 group = {rowLength, ind, numSrcNeurons, numTrgNeurons, rowStride, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedSynapseConnectivityInitGroup1, &group, sizeof(MergedSynapseConnectivityInitGroup1), idx * sizeof(MergedSynapseConnectivityInitGroup1)));
}
__device__ __constant__ MergedSynapseConnectivityInitGroup2 d_mergedSynapseConnectivityInitGroup2[1];
void pushMergedSynapseConnectivityInitGroup2ToDevice(unsigned int idx, unsigned int* rowLength, uint8_t* ind, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int rowStride) {
    MergedSynapseConnectivityInitGroup2 group = {rowLength, ind, numSrcNeurons, numTrgNeurons, rowStride, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedSynapseConnectivityInitGroup2, &group, sizeof(MergedSynapseConnectivityInitGroup2), idx * sizeof(MergedSynapseConnectivityInitGroup2)));
}
__device__ __constant__ MergedSynapseConnectivityInitGroup3 d_mergedSynapseConnectivityInitGroup3[1];
void pushMergedSynapseConnectivityInitGroup3ToDevice(unsigned int idx, unsigned int* rowLength, uint16_t* ind, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int rowStride) {
    MergedSynapseConnectivityInitGroup3 group = {rowLength, ind, numSrcNeurons, numTrgNeurons, rowStride, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedSynapseConnectivityInitGroup3, &group, sizeof(MergedSynapseConnectivityInitGroup3), idx * sizeof(MergedSynapseConnectivityInitGroup3)));
}
__device__ __constant__ MergedSynapseSparseInitGroup0 d_mergedSynapseSparseInitGroup0[1];
void pushMergedSynapseSparseInitGroup0ToDevice(unsigned int idx, scalar* tauMinus, scalar* c, scalar* g, scalar* wMax, scalar* wMin, scalar* aMinus, scalar* aPlus, scalar* tauD, scalar* tauC, scalar* tauPlus, unsigned int* remap, unsigned int* colLength, uint8_t* ind, unsigned int* rowLength, unsigned int numSrcNeurons, unsigned int colStride, unsigned int rowStride, unsigned int numTrgNeurons) {
    MergedSynapseSparseInitGroup0 group = {tauMinus, c, g, wMax, wMin, aMinus, aPlus, tauD, tauC, tauPlus, remap, colLength, ind, rowLength, numSrcNeurons, colStride, rowStride, numTrgNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedSynapseSparseInitGroup0, &group, sizeof(MergedSynapseSparseInitGroup0), idx * sizeof(MergedSynapseSparseInitGroup0)));
}
__device__ __constant__ MergedSynapseSparseInitGroup1 d_mergedSynapseSparseInitGroup1[1];
void pushMergedSynapseSparseInitGroup1ToDevice(unsigned int idx, scalar* tauMinus, scalar* c, scalar* g, scalar* wMax, scalar* wMin, scalar* aMinus, scalar* aPlus, scalar* tauD, scalar* tauC, scalar* tauPlus, unsigned int* remap, unsigned int* colLength, uint16_t* ind, unsigned int* rowLength, unsigned int numSrcNeurons, unsigned int colStride, unsigned int rowStride, unsigned int numTrgNeurons) {
    MergedSynapseSparseInitGroup1 group = {tauMinus, c, g, wMax, wMin, aMinus, aPlus, tauD, tauC, tauPlus, remap, colLength, ind, rowLength, numSrcNeurons, colStride, rowStride, numTrgNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedSynapseSparseInitGroup1, &group, sizeof(MergedSynapseSparseInitGroup1), idx * sizeof(MergedSynapseSparseInitGroup1)));
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
__device__ unsigned int d_mergedNeuronInitGroupStartID0[] = {0, };
__device__ unsigned int d_mergedNeuronInitGroupStartID1[] = {224, };
__device__ unsigned int d_mergedSynapseConnectivityInitGroupStartID0[] = {1024, };
__device__ unsigned int d_mergedSynapseConnectivityInitGroupStartID1[] = {1248, };
__device__ unsigned int d_mergedSynapseConnectivityInitGroupStartID2[] = {1472, };
__device__ unsigned int d_mergedSynapseConnectivityInitGroupStartID3[] = {2272, };
__device__ unsigned int d_mergedSynapseSparseInitGroupStartID0[] = {0, };
__device__ unsigned int d_mergedSynapseSparseInitGroupStartID1[] = {64, };

extern "C" __global__ void initializeRNGKernel(unsigned long long deviceRNGSeed) {
    if(threadIdx.x == 0) {
        curand_init(deviceRNGSeed, 0, 0, &d_rng);
    }
}

extern "C" __global__ void initializeKernel(unsigned long long deviceRNGSeed) {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x;
    // ------------------------------------------------------------------------
    // Local neuron groups
    // merged0
    if(id < 224) {
        struct MergedNeuronInitGroup0 *group = &d_mergedNeuronInitGroup0[0]; 
        const unsigned int lid = id - 0;
        // only do this for existing neurons
        if(lid < group->numNeurons) {
            curand_init(deviceRNGSeed, id, 0, &group->rng[lid]);
            if(lid == 0) {
                group->spkCnt[0] = 0;
            }
            group->spk[lid] = 0;
            group->sT[lid] = -TIME_MAX;
            group->prevST[lid] = -TIME_MAX;
             {
                scalar initVal;
                initVal = (-6.50000000000000000e+01f);
                group->V[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (-1.30000000000000000e+01f);
                group->U[lid] = initVal;
            }
             {
                group->inSynInSyn0[lid] = 0.000000000e+00f;
            }
             {
                group->inSynInSyn1[lid] = 0.000000000e+00f;
            }
            // current source variables
        }
    }
    // merged1
    if(id >= 224 && id < 1024) {
        struct MergedNeuronInitGroup1 *group = &d_mergedNeuronInitGroup1[0]; 
        const unsigned int lid = id - 224;
        // only do this for existing neurons
        if(lid < group->numNeurons) {
            curand_init(deviceRNGSeed, id, 0, &group->rng[lid]);
            if(lid == 0) {
                group->spkCnt[0] = 0;
            }
            if(lid == 0) {
                group->spkCntEvnt[0] = 0;
            }
            group->spk[lid] = 0;
            group->spkEvnt[lid] = 0;
            group->sT[lid] = -TIME_MAX;
            group->prevST[lid] = -TIME_MAX;
            group->seT[lid] = -TIME_MAX;
            group->prevSET[lid] = -TIME_MAX;
             {
                scalar initVal;
                initVal = (-6.50000000000000000e+01f);
                group->V[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (-1.30000000000000000e+01f);
                group->U[lid] = initVal;
            }
             {
                scalar initVal;
                initVal = (0.00000000000000000e+00f);
                group->D[lid] = initVal;
            }
             {
                group->inSynInSyn0[lid] = 0.000000000e+00f;
            }
             {
                group->inSynInSyn1[lid] = 0.000000000e+00f;
            }
            // current source variables
        }
    }
    
    // ------------------------------------------------------------------------
    // Synapse groups
    
    // ------------------------------------------------------------------------
    // Custom update groups
    
    // ------------------------------------------------------------------------
    // Custom WU update groups
    
    // ------------------------------------------------------------------------
    // Synapse groups with sparse connectivity
    // merged0
    if(id >= 1024 && id < 1248) {
        struct MergedSynapseConnectivityInitGroup0 *group = &d_mergedSynapseConnectivityInitGroup0[0]; 
        const unsigned int lid = id - 1024;
        // only do this for existing presynaptic neurons
        if(lid < group->numSrcNeurons) {
            group->rowLength[lid] = 0;
            curandStatePhilox4_32_10_t localRNG = d_rng;
            skipahead_sequence((unsigned long long)id, &localRNG);
            // Build sparse connectivity
            int prevJ = -1.00000000000000000e+00f;
            while(true) {
                const scalar u = curand_uniform(&localRNG);
                prevJ += (1 + (int)(log(u) * (-9.49122158102990454e+00f)));
                if(prevJ < group->numTrgNeurons) {
                   do {
                    const unsigned int idx = (lid * group->rowStride) + group->rowLength[lid];
                    group->ind[idx] = prevJ+0;
                    group->rowLength[lid]++;
                }
                while(false);
                }
                else {
                   break;
                }
                
            }
        }
    }
    // merged1
    if(id >= 1248 && id < 1472) {
        struct MergedSynapseConnectivityInitGroup1 *group = &d_mergedSynapseConnectivityInitGroup1[0]; 
        const unsigned int lid = id - 1248;
        // only do this for existing presynaptic neurons
        if(lid < group->numSrcNeurons) {
            group->rowLength[lid] = 0;
            curandStatePhilox4_32_10_t localRNG = d_rng;
            skipahead_sequence((unsigned long long)id, &localRNG);
            // Build sparse connectivity
            int prevJ = -1.00000000000000000e+00f;
            while(true) {
                const scalar u = curand_uniform(&localRNG);
                prevJ += (1 + (int)(log(u) * (-9.49122158102990454e+00f)));
                if(prevJ < group->numTrgNeurons) {
                   do {
                    const unsigned int idx = (lid * group->rowStride) + group->rowLength[lid];
                    group->ind[idx] = prevJ+0;
                    group->rowLength[lid]++;
                }
                while(false);
                }
                else {
                   break;
                }
                
            }
        }
    }
    // merged2
    if(id >= 1472 && id < 2272) {
        struct MergedSynapseConnectivityInitGroup2 *group = &d_mergedSynapseConnectivityInitGroup2[0]; 
        const unsigned int lid = id - 1472;
        // only do this for existing presynaptic neurons
        if(lid < group->numSrcNeurons) {
            group->rowLength[lid] = 0;
            curandStatePhilox4_32_10_t localRNG = d_rng;
            skipahead_sequence((unsigned long long)id, &localRNG);
            // Build sparse connectivity
            int prevJ = -1.00000000000000000e+00f;
            while(true) {
                const scalar u = curand_uniform(&localRNG);
                prevJ += (1 + (int)(log(u) * (-9.49122158102990454e+00f)));
                if(prevJ < group->numTrgNeurons) {
                   do {
                    const unsigned int idx = (lid * group->rowStride) + group->rowLength[lid];
                    group->ind[idx] = prevJ+0;
                    group->rowLength[lid]++;
                }
                while(false);
                }
                else {
                   break;
                }
                
            }
        }
    }
    // merged3
    if(id >= 2272 && id < 3072) {
        struct MergedSynapseConnectivityInitGroup3 *group = &d_mergedSynapseConnectivityInitGroup3[0]; 
        const unsigned int lid = id - 2272;
        // only do this for existing presynaptic neurons
        if(lid < group->numSrcNeurons) {
            group->rowLength[lid] = 0;
            curandStatePhilox4_32_10_t localRNG = d_rng;
            skipahead_sequence((unsigned long long)id, &localRNG);
            // Build sparse connectivity
            int prevJ = -1.00000000000000000e+00f;
            while(true) {
                const scalar u = curand_uniform(&localRNG);
                prevJ += (1 + (int)(log(u) * (-9.49122158102990454e+00f)));
                if(prevJ < group->numTrgNeurons) {
                   do {
                    const unsigned int idx = (lid * group->rowStride) + group->rowLength[lid];
                    group->ind[idx] = prevJ+0;
                    group->rowLength[lid]++;
                }
                while(false);
                }
                else {
                   break;
                }
                
            }
        }
    }
    
}
extern "C" __global__ void initializeSparseKernel() {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x;
    __shared__ unsigned int shRowLength[32];
    // merged0
    if(id < 64) {
        struct MergedSynapseSparseInitGroup0 *group = &d_mergedSynapseSparseInitGroup0[0]; 
        const unsigned int lid = id - 0;
        curandStatePhilox4_32_10_t localRNG = d_rng;
        skipahead_sequence((unsigned long long)3072 + id, &localRNG);
        const unsigned int numBlocks = (group->numSrcNeurons + 32 - 1) / 32;
        unsigned int idx = lid;
        for(unsigned int r = 0; r < numBlocks; r++) {
            const unsigned numRowsInBlock = (r == (numBlocks - 1)) ? ((group->numSrcNeurons - 1) % 32) + 1 : 32;
            __syncthreads();
            if (threadIdx.x < numRowsInBlock) {
                shRowLength[threadIdx.x] = group->rowLength[(r * 32) + threadIdx.x];
            }
            __syncthreads();
            for(unsigned int i = 0; i < numRowsInBlock; i++) {
                if(lid < shRowLength[i]) {
                     {
                        scalar initVal;
                        const scalar scale = (3.50000000000000000e+01f) - (5.00000000000000000e+00f);
                        initVal = (5.00000000000000000e+00f) + (curand_uniform(&localRNG) * scale);
                        group->tauPlus[(((r * 32) + i) * group->rowStride) + lid] = initVal;
                    }
                     {
                        scalar initVal;
                        const scalar scale = (3.50000000000000000e+01f) - (5.00000000000000000e+00f);
                        initVal = (5.00000000000000000e+00f) + (curand_uniform(&localRNG) * scale);
                        group->tauMinus[(((r * 32) + i) * group->rowStride) + lid] = initVal;
                    }
                     {
                        scalar initVal;
                        initVal = (1.00000000000000000e+03f);
                        group->tauC[(((r * 32) + i) * group->rowStride) + lid] = initVal;
                    }
                     {
                        scalar initVal;
                        initVal = (2.00000000000000000e+02f);
                        group->tauD[(((r * 32) + i) * group->rowStride) + lid] = initVal;
                    }
                     {
                        scalar initVal;
                        initVal = (1.00000000000000006e-01f);
                        group->aPlus[(((r * 32) + i) * group->rowStride) + lid] = initVal;
                    }
                     {
                        scalar initVal;
                        initVal = (1.49999999999999994e-01f);
                        group->aMinus[(((r * 32) + i) * group->rowStride) + lid] = initVal;
                    }
                     {
                        scalar initVal;
                        initVal = (0.00000000000000000e+00f);
                        group->wMin[(((r * 32) + i) * group->rowStride) + lid] = initVal;
                    }
                     {
                        scalar initVal;
                        initVal = (4.00000000000000000e+00f);
                        group->wMax[(((r * 32) + i) * group->rowStride) + lid] = initVal;
                    }
                     {
                        scalar initVal;
                        initVal = (1.00000000000000000e+00f);
                        group->g[(((r * 32) + i) * group->rowStride) + lid] = initVal;
                    }
                     {
                        scalar initVal;
                        initVal = (0.00000000000000000e+00f);
                        group->c[(((r * 32) + i) * group->rowStride) + lid] = initVal;
                    }
                     {
                        const unsigned int postIndex = group->ind[idx];
                        const unsigned int colLocation = atomicAdd(&group->colLength[postIndex], 1);
                        const unsigned int colMajorIndex = (postIndex * group->colStride) + colLocation;
                        group->remap[colMajorIndex] = idx;
                    }
                }
                idx += group->rowStride;
            }
        }
    }
    // merged1
    if(id >= 64 && id < 192) {
        struct MergedSynapseSparseInitGroup1 *group = &d_mergedSynapseSparseInitGroup1[0]; 
        const unsigned int lid = id - 64;
        curandStatePhilox4_32_10_t localRNG = d_rng;
        skipahead_sequence((unsigned long long)3072 + id, &localRNG);
        const unsigned int numBlocks = (group->numSrcNeurons + 32 - 1) / 32;
        unsigned int idx = lid;
        for(unsigned int r = 0; r < numBlocks; r++) {
            const unsigned numRowsInBlock = (r == (numBlocks - 1)) ? ((group->numSrcNeurons - 1) % 32) + 1 : 32;
            __syncthreads();
            if (threadIdx.x < numRowsInBlock) {
                shRowLength[threadIdx.x] = group->rowLength[(r * 32) + threadIdx.x];
            }
            __syncthreads();
            for(unsigned int i = 0; i < numRowsInBlock; i++) {
                if(lid < shRowLength[i]) {
                     {
                        scalar initVal;
                        const scalar scale = (3.50000000000000000e+01f) - (5.00000000000000000e+00f);
                        initVal = (5.00000000000000000e+00f) + (curand_uniform(&localRNG) * scale);
                        group->tauPlus[(((r * 32) + i) * group->rowStride) + lid] = initVal;
                    }
                     {
                        scalar initVal;
                        const scalar scale = (3.50000000000000000e+01f) - (5.00000000000000000e+00f);
                        initVal = (5.00000000000000000e+00f) + (curand_uniform(&localRNG) * scale);
                        group->tauMinus[(((r * 32) + i) * group->rowStride) + lid] = initVal;
                    }
                     {
                        scalar initVal;
                        initVal = (1.00000000000000000e+03f);
                        group->tauC[(((r * 32) + i) * group->rowStride) + lid] = initVal;
                    }
                     {
                        scalar initVal;
                        initVal = (2.00000000000000000e+02f);
                        group->tauD[(((r * 32) + i) * group->rowStride) + lid] = initVal;
                    }
                     {
                        scalar initVal;
                        initVal = (1.00000000000000006e-01f);
                        group->aPlus[(((r * 32) + i) * group->rowStride) + lid] = initVal;
                    }
                     {
                        scalar initVal;
                        initVal = (1.49999999999999994e-01f);
                        group->aMinus[(((r * 32) + i) * group->rowStride) + lid] = initVal;
                    }
                     {
                        scalar initVal;
                        initVal = (0.00000000000000000e+00f);
                        group->wMin[(((r * 32) + i) * group->rowStride) + lid] = initVal;
                    }
                     {
                        scalar initVal;
                        initVal = (4.00000000000000000e+00f);
                        group->wMax[(((r * 32) + i) * group->rowStride) + lid] = initVal;
                    }
                     {
                        scalar initVal;
                        initVal = (1.00000000000000000e+00f);
                        group->g[(((r * 32) + i) * group->rowStride) + lid] = initVal;
                    }
                     {
                        scalar initVal;
                        initVal = (0.00000000000000000e+00f);
                        group->c[(((r * 32) + i) * group->rowStride) + lid] = initVal;
                    }
                     {
                        const unsigned int postIndex = group->ind[idx];
                        const unsigned int colLocation = atomicAdd(&group->colLength[postIndex], 1);
                        const unsigned int colMajorIndex = (postIndex * group->colStride) + colLocation;
                        group->remap[colMajorIndex] = idx;
                    }
                }
                idx += group->rowStride;
            }
        }
    }
}
void initialize() {
    unsigned long long deviceRNGSeed = 0;
    deviceRNGSeed = 1234;
    initializeRNGKernel<<<1, 1>>>(deviceRNGSeed);
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    CHECK_CUDA_ERRORS(cudaMemset(d_colLengthEE, 0, 800 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaMemset(d_colLengthEI, 0, 200 * sizeof(unsigned int)));
     {
        const dim3 threads(32, 1);
        const dim3 grid(96, 1);
        initializeKernel<<<grid, threads>>>(deviceRNGSeed);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}

void initializeSparse() {
    copyStateToDevice(true);
    copyConnectivityToDevice(true);
    
     {
        const dim3 threads(32, 1);
        const dim3 grid(6, 1);
        initializeSparseKernel<<<grid, threads>>>();
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}
