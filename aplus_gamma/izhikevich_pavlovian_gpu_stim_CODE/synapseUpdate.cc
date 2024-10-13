#include "definitionsInternal.h"
#include "supportCode.h"

struct MergedPresynapticUpdateGroup0
 {
    float* inSyn;
    unsigned int* srcSpkCnt;
    unsigned int* srcSpk;
    unsigned int* rowLength;
    uint8_t* ind;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    unsigned int rowStride;
    
}
;
struct MergedPresynapticUpdateGroup1
 {
    float* inSyn;
    unsigned int* srcSpkCnt;
    unsigned int* srcSpk;
    unsigned int* rowLength;
    uint16_t* ind;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    unsigned int rowStride;
    
}
;
struct MergedPresynapticUpdateGroup2
 {
    float* prevSTPost;
    scalar* c;
    scalar* g;
    scalar* wMax;
    scalar* wMin;
    scalar* aMinus;
    scalar* aPlus;
    scalar* tauD;
    scalar* tauC;
    scalar* tauMinus;
    scalar* tauPlus;
    uint8_t* ind;
    unsigned int* rowLength;
    float* prevSETPre;
    float* prevSTPre;
    float* seTPre;
    float* sTPost;
    float* sTPre;
    scalar* DPre;
    unsigned int* srcSpkEvnt;
    unsigned int* srcSpkCntEvnt;
    unsigned int* srcSpk;
    unsigned int* srcSpkCnt;
    float* inSyn;
    unsigned int numSrcNeurons;
    unsigned int rowStride;
    unsigned int numTrgNeurons;
    
}
;
struct MergedPresynapticUpdateGroup3
 {
    float* prevSTPost;
    scalar* c;
    scalar* g;
    scalar* wMax;
    scalar* wMin;
    scalar* aMinus;
    scalar* aPlus;
    scalar* tauD;
    scalar* tauC;
    scalar* tauMinus;
    scalar* tauPlus;
    uint16_t* ind;
    unsigned int* rowLength;
    float* prevSETPre;
    float* prevSTPre;
    float* seTPre;
    float* sTPost;
    float* sTPre;
    scalar* DPre;
    unsigned int* srcSpkEvnt;
    unsigned int* srcSpkCntEvnt;
    unsigned int* srcSpk;
    unsigned int* srcSpkCnt;
    float* inSyn;
    unsigned int numSrcNeurons;
    unsigned int rowStride;
    unsigned int numTrgNeurons;
    
}
;
struct MergedPostsynapticUpdateGroup0
 {
    unsigned int* rowLength;
    scalar* c;
    scalar* g;
    scalar* wMax;
    scalar* wMin;
    scalar* aMinus;
    scalar* aPlus;
    scalar* tauD;
    scalar* tauC;
    scalar* tauMinus;
    scalar* tauPlus;
    unsigned int* remap;
    unsigned int* colLength;
    uint8_t* ind;
    float* prevSETPre;
    float* prevSTPost;
    float* prevSTPre;
    float* seTPre;
    float* sTPost;
    float* sTPre;
    scalar* DPre;
    unsigned int* trgSpk;
    unsigned int* trgSpkCnt;
    unsigned int numSrcNeurons;
    unsigned int colStride;
    unsigned int rowStride;
    unsigned int numTrgNeurons;
    
}
;
struct MergedPostsynapticUpdateGroup1
 {
    unsigned int* rowLength;
    scalar* c;
    scalar* g;
    scalar* wMax;
    scalar* wMin;
    scalar* aMinus;
    scalar* aPlus;
    scalar* tauD;
    scalar* tauC;
    scalar* tauMinus;
    scalar* tauPlus;
    unsigned int* remap;
    unsigned int* colLength;
    uint16_t* ind;
    float* prevSETPre;
    float* prevSTPost;
    float* prevSTPre;
    float* seTPre;
    float* sTPost;
    float* sTPre;
    scalar* DPre;
    unsigned int* trgSpk;
    unsigned int* trgSpkCnt;
    unsigned int numSrcNeurons;
    unsigned int colStride;
    unsigned int rowStride;
    unsigned int numTrgNeurons;
    
}
;
__device__ __constant__ MergedPresynapticUpdateGroup0 d_mergedPresynapticUpdateGroup0[1];
void pushMergedPresynapticUpdateGroup0ToDevice(unsigned int idx, float* inSyn, unsigned int* srcSpkCnt, unsigned int* srcSpk, unsigned int* rowLength, uint8_t* ind, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int rowStride) {
    MergedPresynapticUpdateGroup0 group = {inSyn, srcSpkCnt, srcSpk, rowLength, ind, numSrcNeurons, numTrgNeurons, rowStride, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedPresynapticUpdateGroup0, &group, sizeof(MergedPresynapticUpdateGroup0), idx * sizeof(MergedPresynapticUpdateGroup0)));
}
__device__ __constant__ MergedPresynapticUpdateGroup1 d_mergedPresynapticUpdateGroup1[1];
void pushMergedPresynapticUpdateGroup1ToDevice(unsigned int idx, float* inSyn, unsigned int* srcSpkCnt, unsigned int* srcSpk, unsigned int* rowLength, uint16_t* ind, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int rowStride) {
    MergedPresynapticUpdateGroup1 group = {inSyn, srcSpkCnt, srcSpk, rowLength, ind, numSrcNeurons, numTrgNeurons, rowStride, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedPresynapticUpdateGroup1, &group, sizeof(MergedPresynapticUpdateGroup1), idx * sizeof(MergedPresynapticUpdateGroup1)));
}
__device__ __constant__ MergedPresynapticUpdateGroup2 d_mergedPresynapticUpdateGroup2[1];
void pushMergedPresynapticUpdateGroup2ToDevice(unsigned int idx, float* prevSTPost, scalar* c, scalar* g, scalar* wMax, scalar* wMin, scalar* aMinus, scalar* aPlus, scalar* tauD, scalar* tauC, scalar* tauMinus, scalar* tauPlus, uint8_t* ind, unsigned int* rowLength, float* prevSETPre, float* prevSTPre, float* seTPre, float* sTPost, float* sTPre, scalar* DPre, unsigned int* srcSpkEvnt, unsigned int* srcSpkCntEvnt, unsigned int* srcSpk, unsigned int* srcSpkCnt, float* inSyn, unsigned int numSrcNeurons, unsigned int rowStride, unsigned int numTrgNeurons) {
    MergedPresynapticUpdateGroup2 group = {prevSTPost, c, g, wMax, wMin, aMinus, aPlus, tauD, tauC, tauMinus, tauPlus, ind, rowLength, prevSETPre, prevSTPre, seTPre, sTPost, sTPre, DPre, srcSpkEvnt, srcSpkCntEvnt, srcSpk, srcSpkCnt, inSyn, numSrcNeurons, rowStride, numTrgNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedPresynapticUpdateGroup2, &group, sizeof(MergedPresynapticUpdateGroup2), idx * sizeof(MergedPresynapticUpdateGroup2)));
}
__device__ __constant__ MergedPresynapticUpdateGroup3 d_mergedPresynapticUpdateGroup3[1];
void pushMergedPresynapticUpdateGroup3ToDevice(unsigned int idx, float* prevSTPost, scalar* c, scalar* g, scalar* wMax, scalar* wMin, scalar* aMinus, scalar* aPlus, scalar* tauD, scalar* tauC, scalar* tauMinus, scalar* tauPlus, uint16_t* ind, unsigned int* rowLength, float* prevSETPre, float* prevSTPre, float* seTPre, float* sTPost, float* sTPre, scalar* DPre, unsigned int* srcSpkEvnt, unsigned int* srcSpkCntEvnt, unsigned int* srcSpk, unsigned int* srcSpkCnt, float* inSyn, unsigned int numSrcNeurons, unsigned int rowStride, unsigned int numTrgNeurons) {
    MergedPresynapticUpdateGroup3 group = {prevSTPost, c, g, wMax, wMin, aMinus, aPlus, tauD, tauC, tauMinus, tauPlus, ind, rowLength, prevSETPre, prevSTPre, seTPre, sTPost, sTPre, DPre, srcSpkEvnt, srcSpkCntEvnt, srcSpk, srcSpkCnt, inSyn, numSrcNeurons, rowStride, numTrgNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedPresynapticUpdateGroup3, &group, sizeof(MergedPresynapticUpdateGroup3), idx * sizeof(MergedPresynapticUpdateGroup3)));
}
__device__ __constant__ MergedPostsynapticUpdateGroup0 d_mergedPostsynapticUpdateGroup0[1];
void pushMergedPostsynapticUpdateGroup0ToDevice(unsigned int idx, unsigned int* rowLength, scalar* c, scalar* g, scalar* wMax, scalar* wMin, scalar* aMinus, scalar* aPlus, scalar* tauD, scalar* tauC, scalar* tauMinus, scalar* tauPlus, unsigned int* remap, unsigned int* colLength, uint8_t* ind, float* prevSETPre, float* prevSTPost, float* prevSTPre, float* seTPre, float* sTPost, float* sTPre, scalar* DPre, unsigned int* trgSpk, unsigned int* trgSpkCnt, unsigned int numSrcNeurons, unsigned int colStride, unsigned int rowStride, unsigned int numTrgNeurons) {
    MergedPostsynapticUpdateGroup0 group = {rowLength, c, g, wMax, wMin, aMinus, aPlus, tauD, tauC, tauMinus, tauPlus, remap, colLength, ind, prevSETPre, prevSTPost, prevSTPre, seTPre, sTPost, sTPre, DPre, trgSpk, trgSpkCnt, numSrcNeurons, colStride, rowStride, numTrgNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedPostsynapticUpdateGroup0, &group, sizeof(MergedPostsynapticUpdateGroup0), idx * sizeof(MergedPostsynapticUpdateGroup0)));
}
__device__ __constant__ MergedPostsynapticUpdateGroup1 d_mergedPostsynapticUpdateGroup1[1];
void pushMergedPostsynapticUpdateGroup1ToDevice(unsigned int idx, unsigned int* rowLength, scalar* c, scalar* g, scalar* wMax, scalar* wMin, scalar* aMinus, scalar* aPlus, scalar* tauD, scalar* tauC, scalar* tauMinus, scalar* tauPlus, unsigned int* remap, unsigned int* colLength, uint16_t* ind, float* prevSETPre, float* prevSTPost, float* prevSTPre, float* seTPre, float* sTPost, float* sTPre, scalar* DPre, unsigned int* trgSpk, unsigned int* trgSpkCnt, unsigned int numSrcNeurons, unsigned int colStride, unsigned int rowStride, unsigned int numTrgNeurons) {
    MergedPostsynapticUpdateGroup1 group = {rowLength, c, g, wMax, wMin, aMinus, aPlus, tauD, tauC, tauMinus, tauPlus, remap, colLength, ind, prevSETPre, prevSTPost, prevSTPre, seTPre, sTPost, sTPre, DPre, trgSpk, trgSpkCnt, numSrcNeurons, colStride, rowStride, numTrgNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedPostsynapticUpdateGroup1, &group, sizeof(MergedPostsynapticUpdateGroup1), idx * sizeof(MergedPostsynapticUpdateGroup1)));
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
__device__ __constant__ unsigned int d_mergedPresynapticUpdateGroupStartID0[] = {0, };
__device__ __constant__ unsigned int d_mergedPresynapticUpdateGroupStartID1[] = {64, };
__device__ __constant__ unsigned int d_mergedPresynapticUpdateGroupStartID2[] = {192, };
__device__ __constant__ unsigned int d_mergedPresynapticUpdateGroupStartID3[] = {256, };
__device__ __constant__ unsigned int d_mergedPostsynapticUpdateGroupStartID0[] = {0, };
__device__ __constant__ unsigned int d_mergedPostsynapticUpdateGroupStartID1[] = {128, };
extern "C" __global__ void updatePresynapticKernel(float t)
 {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x; 
    __shared__ unsigned int shRowLength[32];
    __shared__ unsigned int shSpk[32];
    __shared__ unsigned int shSpkEvnt[32];
    // merged0
    if(id < 64) {
        struct MergedPresynapticUpdateGroup0 *group = &d_mergedPresynapticUpdateGroup0[0]; 
        const unsigned int lid = id - 0;
         {
            const unsigned int numSpikes = group->srcSpkCnt[0];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                __syncthreads();
                if (threadIdx.x < numSpikesInBlock) {
                    const unsigned int spk = group->srcSpk[(r * 32) + threadIdx.x];
                    shSpk[threadIdx.x] = spk;
                    shRowLength[threadIdx.x] = group->rowLength[spk];
                }
                __syncthreads();
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < group->rowStride) {
                        const unsigned int synAddress = (shSpk[j] * group->rowStride) + lid;
                        const unsigned int npost = shRowLength[j];
                        if (lid < npost) {
                            const unsigned int ipost = group->ind[synAddress];
                            atomicAdd(&group->inSyn[ipost], (-1.00000000000000000e+00f));
                        }
                    }
                }
            }
        }
        
    }
    // merged1
    if(id >= 64 && id < 192) {
        struct MergedPresynapticUpdateGroup1 *group = &d_mergedPresynapticUpdateGroup1[0]; 
        const unsigned int lid = id - 64;
         {
            const unsigned int numSpikes = group->srcSpkCnt[0];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                __syncthreads();
                if (threadIdx.x < numSpikesInBlock) {
                    const unsigned int spk = group->srcSpk[(r * 32) + threadIdx.x];
                    shSpk[threadIdx.x] = spk;
                    shRowLength[threadIdx.x] = group->rowLength[spk];
                }
                __syncthreads();
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < group->rowStride) {
                        const unsigned int synAddress = (shSpk[j] * group->rowStride) + lid;
                        const unsigned int npost = shRowLength[j];
                        if (lid < npost) {
                            const unsigned int ipost = group->ind[synAddress];
                            atomicAdd(&group->inSyn[ipost], (-1.00000000000000000e+00f));
                        }
                    }
                }
            }
        }
        
    }
    // merged2
    if(id >= 192 && id < 256) {
        struct MergedPresynapticUpdateGroup2 *group = &d_mergedPresynapticUpdateGroup2[0]; 
        const unsigned int lid = id - 192;
         {
            const unsigned int numSpikes = group->srcSpkCntEvnt[0];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                __syncthreads();
                if (threadIdx.x < numSpikesInBlock) {
                    const unsigned int spk = group->srcSpkEvnt[(r * 32) + threadIdx.x];
                    shSpkEvnt[threadIdx.x] = spk;
                    shRowLength[threadIdx.x] = group->rowLength[spk];
                }
                __syncthreads();
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < group->rowStride) {
                        const unsigned int synAddress = (shSpkEvnt[j] * group->rowStride) + lid;
                        const unsigned int npost = shRowLength[j];
                        if (lid < npost) {
                            const unsigned int ipost = group->ind[synAddress];
                            
                                // Calculate time of last tag update
                                const scalar scale = 1.0f / -((1.0f / group->tauC[synAddress]) + (1.0f / group->tauD[synAddress]));
                                const scalar tc = fmax((1.00000000000000000e+00f + group->sTPre[shSpkEvnt[j]]), fmax((1.00000000000000000e+00f + group->prevSTPost[ipost]), (1.00000000000000000e+00f + group->prevSETPre[shSpkEvnt[j]])));
                            
                            // Calculate how much tag has decayed since last update
                            const scalar tagDT = t - tc;
                            const scalar tagDecay = exp(-tagDT / group->tauC[synAddress]);
                            // Calculate how much dopamine has decayed since last update
                            const scalar dopamineDT = t - (1.00000000000000000e+00f + group->seTPre[shSpkEvnt[j]]);
                            const scalar dopamineDecay = exp(-dopamineDT / group->tauD[synAddress]);
                            // Calculate offset to integrate over correct area
                            const scalar offset = (tc <= (1.00000000000000000e+00f + group->seTPre[shSpkEvnt[j]])) ? exp(-((1.00000000000000000e+00f + group->seTPre[shSpkEvnt[j]]) - tc) / group->tauC[synAddress]) : exp(-(tc - (1.00000000000000000e+00f + group->seTPre[shSpkEvnt[j]])) / group->tauD[synAddress]);
                            // Update weight and clamp
                            // group->g[synAddress] += (group->c[synAddress] * group->DPre[shSpkEvnt[j]] * $(scale)) * ((tagDecay * dopamineDecay) - offset);
                            group->g[synAddress] += (group->c[synAddress] * group->DPre[shSpkEvnt[j]] * scale) * ((tagDecay * dopamineDecay) - offset);
                            group->g[synAddress] = fmax(group->wMin[synAddress], fmin(group->wMax[synAddress], group->g[synAddress]));
                            
                                // Decay tag
                                group->c[synAddress] *= tagDecay;
                        }
                    }
                }
            }
        }
         {
            const unsigned int numSpikes = group->srcSpkCnt[0];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                __syncthreads();
                if (threadIdx.x < numSpikesInBlock) {
                    const unsigned int spk = group->srcSpk[(r * 32) + threadIdx.x];
                    shSpk[threadIdx.x] = spk;
                    shRowLength[threadIdx.x] = group->rowLength[spk];
                }
                __syncthreads();
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < group->rowStride) {
                        const unsigned int synAddress = (shSpk[j] * group->rowStride) + lid;
                        const unsigned int npost = shRowLength[j];
                        if (lid < npost) {
                            const unsigned int ipost = group->ind[synAddress];
                            
                                atomicAdd(&group->inSyn[ipost], group->g[synAddress]);
                                const scalar scale = 1.0f / -((1.0f / group->tauC[synAddress]) + (1.0f / group->tauD[synAddress]));
                                // Calculate time of last tag update
                                const scalar tc = fmax((1.00000000000000000e+00f + group->prevSTPre[shSpk[j]]), fmax((1.00000000000000000e+00f + group->prevSTPost[ipost]), (1.00000000000000000e+00f + group->prevSETPre[shSpk[j]])));
                            
                            // Calculate how much tag has decayed since last update
                            const scalar tagDT = t - tc;
                            const scalar tagDecay = exp(-tagDT / group->tauC[synAddress]);
                            // Calculate how much dopamine has decayed since last update
                            const scalar dopamineDT = t - (1.00000000000000000e+00f + group->seTPre[shSpk[j]]);
                            const scalar dopamineDecay = exp(-dopamineDT / group->tauD[synAddress]);
                            // Calculate offset to integrate over correct area
                            const scalar offset = (tc <= (1.00000000000000000e+00f + group->seTPre[shSpk[j]])) ? exp(-((1.00000000000000000e+00f + group->seTPre[shSpk[j]]) - tc) / group->tauC[synAddress]) : exp(-(tc - (1.00000000000000000e+00f + group->seTPre[shSpk[j]])) / group->tauD[synAddress]);
                            // Update weight and clamp
                            // group->g[synAddress] += (group->c[synAddress] * group->DPre[shSpk[j]] * $(scale)) * ((tagDecay * dopamineDecay) - offset);
                            group->g[synAddress] += (group->c[synAddress] * group->DPre[shSpk[j]] * scale) * ((tagDecay * dopamineDecay) - offset);
                            group->g[synAddress] = fmax(group->wMin[synAddress], fmin(group->wMax[synAddress], group->g[synAddress]));
                            
                                // Decay tag and apply STDP
                                scalar newTag = group->c[synAddress] * tagDecay;
                                const scalar dt = t - (1.00000000000000000e+00f + group->sTPost[ipost]);
                                if (dt > 0) {
                                    scalar timing = exp(-dt / group->tauMinus[synAddress]);
                                    newTag -= (group->aMinus[synAddress] * timing);
                                }
                                // Write back updated tag and update time
                                group->c[synAddress] = newTag;
                        }
                    }
                }
            }
        }
        
    }
    // merged3
    if(id >= 256 && id < 384) {
        struct MergedPresynapticUpdateGroup3 *group = &d_mergedPresynapticUpdateGroup3[0]; 
        const unsigned int lid = id - 256;
         {
            const unsigned int numSpikes = group->srcSpkCntEvnt[0];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                __syncthreads();
                if (threadIdx.x < numSpikesInBlock) {
                    const unsigned int spk = group->srcSpkEvnt[(r * 32) + threadIdx.x];
                    shSpkEvnt[threadIdx.x] = spk;
                    shRowLength[threadIdx.x] = group->rowLength[spk];
                }
                __syncthreads();
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < group->rowStride) {
                        const unsigned int synAddress = (shSpkEvnt[j] * group->rowStride) + lid;
                        const unsigned int npost = shRowLength[j];
                        if (lid < npost) {
                            const unsigned int ipost = group->ind[synAddress];
                            
                                // Calculate time of last tag update
                                const scalar scale = 1.0f / -((1.0f / group->tauC[synAddress]) + (1.0f / group->tauD[synAddress]));
                                const scalar tc = fmax((1.00000000000000000e+00f + group->sTPre[shSpkEvnt[j]]), fmax((1.00000000000000000e+00f + group->prevSTPost[ipost]), (1.00000000000000000e+00f + group->prevSETPre[shSpkEvnt[j]])));
                            
                            // Calculate how much tag has decayed since last update
                            const scalar tagDT = t - tc;
                            const scalar tagDecay = exp(-tagDT / group->tauC[synAddress]);
                            // Calculate how much dopamine has decayed since last update
                            const scalar dopamineDT = t - (1.00000000000000000e+00f + group->seTPre[shSpkEvnt[j]]);
                            const scalar dopamineDecay = exp(-dopamineDT / group->tauD[synAddress]);
                            // Calculate offset to integrate over correct area
                            const scalar offset = (tc <= (1.00000000000000000e+00f + group->seTPre[shSpkEvnt[j]])) ? exp(-((1.00000000000000000e+00f + group->seTPre[shSpkEvnt[j]]) - tc) / group->tauC[synAddress]) : exp(-(tc - (1.00000000000000000e+00f + group->seTPre[shSpkEvnt[j]])) / group->tauD[synAddress]);
                            // Update weight and clamp
                            // group->g[synAddress] += (group->c[synAddress] * group->DPre[shSpkEvnt[j]] * $(scale)) * ((tagDecay * dopamineDecay) - offset);
                            group->g[synAddress] += (group->c[synAddress] * group->DPre[shSpkEvnt[j]] * scale) * ((tagDecay * dopamineDecay) - offset);
                            group->g[synAddress] = fmax(group->wMin[synAddress], fmin(group->wMax[synAddress], group->g[synAddress]));
                            
                                // Decay tag
                                group->c[synAddress] *= tagDecay;
                        }
                    }
                }
            }
        }
         {
            const unsigned int numSpikes = group->srcSpkCnt[0];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                __syncthreads();
                if (threadIdx.x < numSpikesInBlock) {
                    const unsigned int spk = group->srcSpk[(r * 32) + threadIdx.x];
                    shSpk[threadIdx.x] = spk;
                    shRowLength[threadIdx.x] = group->rowLength[spk];
                }
                __syncthreads();
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < group->rowStride) {
                        const unsigned int synAddress = (shSpk[j] * group->rowStride) + lid;
                        const unsigned int npost = shRowLength[j];
                        if (lid < npost) {
                            const unsigned int ipost = group->ind[synAddress];
                            
                                atomicAdd(&group->inSyn[ipost], group->g[synAddress]);
                                const scalar scale = 1.0f / -((1.0f / group->tauC[synAddress]) + (1.0f / group->tauD[synAddress]));
                                // Calculate time of last tag update
                                const scalar tc = fmax((1.00000000000000000e+00f + group->prevSTPre[shSpk[j]]), fmax((1.00000000000000000e+00f + group->prevSTPost[ipost]), (1.00000000000000000e+00f + group->prevSETPre[shSpk[j]])));
                            
                            // Calculate how much tag has decayed since last update
                            const scalar tagDT = t - tc;
                            const scalar tagDecay = exp(-tagDT / group->tauC[synAddress]);
                            // Calculate how much dopamine has decayed since last update
                            const scalar dopamineDT = t - (1.00000000000000000e+00f + group->seTPre[shSpk[j]]);
                            const scalar dopamineDecay = exp(-dopamineDT / group->tauD[synAddress]);
                            // Calculate offset to integrate over correct area
                            const scalar offset = (tc <= (1.00000000000000000e+00f + group->seTPre[shSpk[j]])) ? exp(-((1.00000000000000000e+00f + group->seTPre[shSpk[j]]) - tc) / group->tauC[synAddress]) : exp(-(tc - (1.00000000000000000e+00f + group->seTPre[shSpk[j]])) / group->tauD[synAddress]);
                            // Update weight and clamp
                            // group->g[synAddress] += (group->c[synAddress] * group->DPre[shSpk[j]] * $(scale)) * ((tagDecay * dopamineDecay) - offset);
                            group->g[synAddress] += (group->c[synAddress] * group->DPre[shSpk[j]] * scale) * ((tagDecay * dopamineDecay) - offset);
                            group->g[synAddress] = fmax(group->wMin[synAddress], fmin(group->wMax[synAddress], group->g[synAddress]));
                            
                                // Decay tag and apply STDP
                                scalar newTag = group->c[synAddress] * tagDecay;
                                const scalar dt = t - (1.00000000000000000e+00f + group->sTPost[ipost]);
                                if (dt > 0) {
                                    scalar timing = exp(-dt / group->tauMinus[synAddress]);
                                    newTag -= (group->aMinus[synAddress] * timing);
                                }
                                // Write back updated tag and update time
                                group->c[synAddress] = newTag;
                        }
                    }
                }
            }
        }
        
    }
}
extern "C" __global__ void updatePostsynapticKernel(float t)
 {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x; 
    __shared__ unsigned int shSpk[32];
    __shared__ unsigned int shColLength[32];
    // merged0
    if(id < 128) {
        struct MergedPostsynapticUpdateGroup0 *group = &d_mergedPostsynapticUpdateGroup0[0]; 
        const unsigned int lid = id - 0;
        const unsigned int numSpikes = group->trgSpkCnt[0];
        const unsigned int numSpikeBlocks = (numSpikes + 31) / 32;
        for (unsigned int r = 0; r < numSpikeBlocks; r++) {
            const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
            if (threadIdx.x < numSpikesInBlock) {
                const unsigned int spk = group->trgSpk[(r * 32) + threadIdx.x];
                shSpk[threadIdx.x] = spk;
                shColLength[threadIdx.x] = group->colLength[spk];
            }
            __syncthreads();
            // only work on existing neurons
            if (lid < group->colStride) {
                // loop through all incoming spikes for learning
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    if (lid < shColLength[j]) {
                        const unsigned int synAddress = group->remap[(shSpk[j] * group->colStride) + lid];
                        const unsigned int ipre = synAddress / group->rowStride;
                        
                            // Calculate time of last tag update
                            const scalar scale = 1.0f / -((1.0f / group->tauC[synAddress]) + (1.0f / group->tauD[synAddress]));
                            const scalar tc = fmax((1.00000000000000000e+00f + group->sTPre[ipre]), fmax((1.00000000000000000e+00f + group->prevSTPost[shSpk[j]]), (1.00000000000000000e+00f + group->seTPre[ipre])));
                        
                        // Calculate how much tag has decayed since last update
                        const scalar tagDT = t - tc;
                        const scalar tagDecay = exp(-tagDT / group->tauC[synAddress]);
                        // Calculate how much dopamine has decayed since last update
                        const scalar dopamineDT = t - (1.00000000000000000e+00f + group->seTPre[ipre]);
                        const scalar dopamineDecay = exp(-dopamineDT / group->tauD[synAddress]);
                        // Calculate offset to integrate over correct area
                        const scalar offset = (tc <= (1.00000000000000000e+00f + group->seTPre[ipre])) ? exp(-((1.00000000000000000e+00f + group->seTPre[ipre]) - tc) / group->tauC[synAddress]) : exp(-(tc - (1.00000000000000000e+00f + group->seTPre[ipre])) / group->tauD[synAddress]);
                        // Update weight and clamp
                        // group->g[synAddress] += (group->c[synAddress] * group->DPre[ipre] * $(scale)) * ((tagDecay * dopamineDecay) - offset);
                        group->g[synAddress] += (group->c[synAddress] * group->DPre[ipre] * scale) * ((tagDecay * dopamineDecay) - offset);
                        group->g[synAddress] = fmax(group->wMin[synAddress], fmin(group->wMax[synAddress], group->g[synAddress]));
                        
                            // Decay tag and apply STDP
                            scalar newTag = group->c[synAddress] * tagDecay;
                            const scalar dt = t - (1.00000000000000000e+00f + group->sTPre[ipre]);
                            if (dt > 0) {
                                scalar timing = exp(-dt / group->tauPlus[synAddress]);
                                newTag += (group->aPlus[synAddress] * timing);
                            }
                            // Write back updated tag and update time
                            group->c[synAddress] = newTag;
                    }
                }
            }
        }
    }
    // merged1
    if(id >= 128 && id < 256) {
        struct MergedPostsynapticUpdateGroup1 *group = &d_mergedPostsynapticUpdateGroup1[0]; 
        const unsigned int lid = id - 128;
        const unsigned int numSpikes = group->trgSpkCnt[0];
        const unsigned int numSpikeBlocks = (numSpikes + 31) / 32;
        for (unsigned int r = 0; r < numSpikeBlocks; r++) {
            const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
            if (threadIdx.x < numSpikesInBlock) {
                const unsigned int spk = group->trgSpk[(r * 32) + threadIdx.x];
                shSpk[threadIdx.x] = spk;
                shColLength[threadIdx.x] = group->colLength[spk];
            }
            __syncthreads();
            // only work on existing neurons
            if (lid < group->colStride) {
                // loop through all incoming spikes for learning
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    if (lid < shColLength[j]) {
                        const unsigned int synAddress = group->remap[(shSpk[j] * group->colStride) + lid];
                        const unsigned int ipre = synAddress / group->rowStride;
                        
                            // Calculate time of last tag update
                            const scalar scale = 1.0f / -((1.0f / group->tauC[synAddress]) + (1.0f / group->tauD[synAddress]));
                            const scalar tc = fmax((1.00000000000000000e+00f + group->sTPre[ipre]), fmax((1.00000000000000000e+00f + group->prevSTPost[shSpk[j]]), (1.00000000000000000e+00f + group->seTPre[ipre])));
                        
                        // Calculate how much tag has decayed since last update
                        const scalar tagDT = t - tc;
                        const scalar tagDecay = exp(-tagDT / group->tauC[synAddress]);
                        // Calculate how much dopamine has decayed since last update
                        const scalar dopamineDT = t - (1.00000000000000000e+00f + group->seTPre[ipre]);
                        const scalar dopamineDecay = exp(-dopamineDT / group->tauD[synAddress]);
                        // Calculate offset to integrate over correct area
                        const scalar offset = (tc <= (1.00000000000000000e+00f + group->seTPre[ipre])) ? exp(-((1.00000000000000000e+00f + group->seTPre[ipre]) - tc) / group->tauC[synAddress]) : exp(-(tc - (1.00000000000000000e+00f + group->seTPre[ipre])) / group->tauD[synAddress]);
                        // Update weight and clamp
                        // group->g[synAddress] += (group->c[synAddress] * group->DPre[ipre] * $(scale)) * ((tagDecay * dopamineDecay) - offset);
                        group->g[synAddress] += (group->c[synAddress] * group->DPre[ipre] * scale) * ((tagDecay * dopamineDecay) - offset);
                        group->g[synAddress] = fmax(group->wMin[synAddress], fmin(group->wMax[synAddress], group->g[synAddress]));
                        
                            // Decay tag and apply STDP
                            scalar newTag = group->c[synAddress] * tagDecay;
                            const scalar dt = t - (1.00000000000000000e+00f + group->sTPre[ipre]);
                            if (dt > 0) {
                                scalar timing = exp(-dt / group->tauPlus[synAddress]);
                                newTag += (group->aPlus[synAddress] * timing);
                            }
                            // Write back updated tag and update time
                            group->c[synAddress] = newTag;
                    }
                }
            }
        }
    }
}
void updateSynapses(float t) {
     {
        const dim3 threads(32, 1);
        const dim3 grid(12, 1);
        updatePresynapticKernel<<<grid, threads>>>(t);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
     {
        const dim3 threads(32, 1);
        const dim3 grid(8, 1);
        updatePostsynapticKernel<<<grid, threads>>>(t);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}
