#include "definitionsInternal.h"
#include "supportCode.h"

struct MergedNeuronUpdateGroup0
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
    unsigned int* startStimCS0;
    unsigned int* endStimCS0;
    scalar* stimTimesCS0;
    uint32_t* recordSpk;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronUpdateGroup1
 {
    scalar* V;
    uint32_t* recordSpk;
    scalar* stimTimesCS0;
    unsigned int* endStimCS0;
    unsigned int* startStimCS0;
    float* inSynInSyn1;
    float* inSynInSyn0;
    uint32_t* rewardTimesteps;
    scalar* D;
    scalar* U;
    curandState* rng;
    float* prevSET;
    float* prevST;
    float* seT;
    float* sT;
    unsigned int* spkEvnt;
    unsigned int* spkCntEvnt;
    unsigned int* spk;
    unsigned int* spkCnt;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronSpikeQueueUpdateGroup0
 {
    unsigned int* spkCnt;
    
}
;
struct MergedNeuronSpikeQueueUpdateGroup1
 {
    unsigned int* spkCnt;
    unsigned int* spkCntEvnt;
    
}
;
struct MergedNeuronPrevSpikeTimeUpdateGroup0
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    float* prevST;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronPrevSpikeTimeUpdateGroup1
 {
    unsigned int* spkCnt;
    unsigned int* spkCntEvnt;
    unsigned int* spk;
    float* prevST;
    unsigned int* spkEvnt;
    float* prevSET;
    unsigned int numNeurons;
    
}
;
__device__ __constant__ MergedNeuronSpikeQueueUpdateGroup0 d_mergedNeuronSpikeQueueUpdateGroup0[1];
void pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(unsigned int idx, unsigned int* spkCnt) {
    MergedNeuronSpikeQueueUpdateGroup0 group = {spkCnt, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronSpikeQueueUpdateGroup0, &group, sizeof(MergedNeuronSpikeQueueUpdateGroup0), idx * sizeof(MergedNeuronSpikeQueueUpdateGroup0)));
}
__device__ __constant__ MergedNeuronSpikeQueueUpdateGroup1 d_mergedNeuronSpikeQueueUpdateGroup1[1];
void pushMergedNeuronSpikeQueueUpdateGroup1ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spkCntEvnt) {
    MergedNeuronSpikeQueueUpdateGroup1 group = {spkCnt, spkCntEvnt, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronSpikeQueueUpdateGroup1, &group, sizeof(MergedNeuronSpikeQueueUpdateGroup1), idx * sizeof(MergedNeuronSpikeQueueUpdateGroup1)));
}
__device__ __constant__ MergedNeuronPrevSpikeTimeUpdateGroup0 d_mergedNeuronPrevSpikeTimeUpdateGroup0[1];
void pushMergedNeuronPrevSpikeTimeUpdateGroup0ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, float* prevST, unsigned int numNeurons) {
    MergedNeuronPrevSpikeTimeUpdateGroup0 group = {spkCnt, spk, prevST, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronPrevSpikeTimeUpdateGroup0, &group, sizeof(MergedNeuronPrevSpikeTimeUpdateGroup0), idx * sizeof(MergedNeuronPrevSpikeTimeUpdateGroup0)));
}
__device__ __constant__ MergedNeuronPrevSpikeTimeUpdateGroup1 d_mergedNeuronPrevSpikeTimeUpdateGroup1[1];
void pushMergedNeuronPrevSpikeTimeUpdateGroup1ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spkCntEvnt, unsigned int* spk, float* prevST, unsigned int* spkEvnt, float* prevSET, unsigned int numNeurons) {
    MergedNeuronPrevSpikeTimeUpdateGroup1 group = {spkCnt, spkCntEvnt, spk, prevST, spkEvnt, prevSET, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronPrevSpikeTimeUpdateGroup1, &group, sizeof(MergedNeuronPrevSpikeTimeUpdateGroup1), idx * sizeof(MergedNeuronPrevSpikeTimeUpdateGroup1)));
}
__device__ __constant__ MergedNeuronUpdateGroup0 d_mergedNeuronUpdateGroup0[1];
void pushMergedNeuronUpdateGroup0ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, float* sT, float* prevST, curandState* rng, scalar* V, scalar* U, float* inSynInSyn0, float* inSynInSyn1, unsigned int* startStimCS0, unsigned int* endStimCS0, scalar* stimTimesCS0, uint32_t* recordSpk, unsigned int numNeurons) {
    MergedNeuronUpdateGroup0 group = {spkCnt, spk, sT, prevST, rng, V, U, inSynInSyn0, inSynInSyn1, startStimCS0, endStimCS0, stimTimesCS0, recordSpk, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup0, &group, sizeof(MergedNeuronUpdateGroup0), idx * sizeof(MergedNeuronUpdateGroup0)));
}
__device__ __constant__ MergedNeuronUpdateGroup1 d_mergedNeuronUpdateGroup1[1];
void pushMergedNeuronUpdateGroup1ToDevice(unsigned int idx, scalar* V, uint32_t* recordSpk, scalar* stimTimesCS0, unsigned int* endStimCS0, unsigned int* startStimCS0, float* inSynInSyn1, float* inSynInSyn0, uint32_t* rewardTimesteps, scalar* D, scalar* U, curandState* rng, float* prevSET, float* prevST, float* seT, float* sT, unsigned int* spkEvnt, unsigned int* spkCntEvnt, unsigned int* spk, unsigned int* spkCnt, unsigned int numNeurons) {
    MergedNeuronUpdateGroup1 group = {V, recordSpk, stimTimesCS0, endStimCS0, startStimCS0, inSynInSyn1, inSynInSyn0, rewardTimesteps, D, U, rng, prevSET, prevST, seT, sT, spkEvnt, spkCntEvnt, spk, spkCnt, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup1, &group, sizeof(MergedNeuronUpdateGroup1), idx * sizeof(MergedNeuronUpdateGroup1)));
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
void pushMergedNeuronUpdate0stimTimesCS0ToDevice(unsigned int idx, scalar* value) {
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup0, &value, sizeof(value), (sizeof(MergedNeuronUpdateGroup0) * (idx)) + offsetof(MergedNeuronUpdateGroup0, stimTimesCS0)));
}

void pushMergedNeuronUpdate0recordSpkToDevice(unsigned int idx, uint32_t* value) {
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup0, &value, sizeof(value), (sizeof(MergedNeuronUpdateGroup0) * (idx)) + offsetof(MergedNeuronUpdateGroup0, recordSpk)));
}

void pushMergedNeuronUpdate1stimTimesCS0ToDevice(unsigned int idx, scalar* value) {
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup1, &value, sizeof(value), (sizeof(MergedNeuronUpdateGroup1) * (idx)) + offsetof(MergedNeuronUpdateGroup1, stimTimesCS0)));
}

void pushMergedNeuronUpdate1recordSpkToDevice(unsigned int idx, uint32_t* value) {
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup1, &value, sizeof(value), (sizeof(MergedNeuronUpdateGroup1) * (idx)) + offsetof(MergedNeuronUpdateGroup1, recordSpk)));
}

void pushMergedNeuronUpdate1rewardTimestepsToDevice(unsigned int idx, uint32_t* value) {
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup1, &value, sizeof(value), (sizeof(MergedNeuronUpdateGroup1) * (idx)) + offsetof(MergedNeuronUpdateGroup1, rewardTimesteps)));
}

__device__ __constant__ unsigned int d_mergedNeuronUpdateGroupStartID0[] = {0, };
__device__ __constant__ unsigned int d_mergedNeuronUpdateGroupStartID1[] = {224, };
__device__ __constant__ unsigned int d_mergedNeuronPrevSpikeTimeUpdateGroupStartID0[] = {0, };
__device__ __constant__ unsigned int d_mergedNeuronPrevSpikeTimeUpdateGroupStartID1[] = {224, };

extern "C" __global__ void neuronPrevSpikeTimeUpdateKernel(float t) {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x;
    // merged0
    if(id < 224) {
        struct MergedNeuronPrevSpikeTimeUpdateGroup0 *group = &d_mergedNeuronPrevSpikeTimeUpdateGroup0[0]; 
        const unsigned int lid = id - 0;
         {
            if(lid < group->spkCnt[0]) {
                group->prevST[group->spk[lid]] = t - DT;
            }
            
        }
    }
    // merged1
    if(id >= 224 && id < 1024) {
        struct MergedNeuronPrevSpikeTimeUpdateGroup1 *group = &d_mergedNeuronPrevSpikeTimeUpdateGroup1[0]; 
        const unsigned int lid = id - 224;
         {
            if(lid < group->spkCnt[0]) {
                group->prevST[group->spk[lid]] = t - DT;
            }
            if(lid < group->spkCntEvnt[0]) {
                group->prevSET[group->spkEvnt[lid]] = t - DT;
            }
            
        }
    }
}

extern "C" __global__ void neuronSpikeQueueUpdateKernel() {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x;
    if(id < 1) {
        struct MergedNeuronSpikeQueueUpdateGroup0 *group = &d_mergedNeuronSpikeQueueUpdateGroup0[id - 0]; 
        group->spkCnt[0] = 0;
    }
    if(id >= 1 && id < 2) {
        struct MergedNeuronSpikeQueueUpdateGroup1 *group = &d_mergedNeuronSpikeQueueUpdateGroup1[id - 1]; 
        group->spkCntEvnt[0] = 0;
        group->spkCnt[0] = 0;
    }
}

extern "C" __global__ void updateNeuronsKernel(float t, unsigned int recordingTimestep)
 {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x; 
    __shared__ unsigned int shSpkEvnt[32];
    __shared__ unsigned int shPosSpkEvnt;
    __shared__ unsigned int shSpkEvntCount;
    
    if (threadIdx.x == 1) {
        shSpkEvntCount = 0;
    }
    
    __shared__ unsigned int shSpk[32];
    __shared__ unsigned int shPosSpk;
    __shared__ unsigned int shSpkCount;
    if (threadIdx.x == 0) {
        shSpkCount = 0;
    }
    
    __shared__ uint32_t shSpkRecord;
    if (threadIdx.x == 0) {
        shSpkRecord = 0;
    }
    __syncthreads();
    // merged0
    if(id < 224) {
        struct MergedNeuronUpdateGroup0 *group = &d_mergedNeuronUpdateGroup0[0]; 
        const unsigned int lid = id - 0;
        
        if(lid < group->numNeurons) {
            scalar lV = group->V[lid];
            scalar lU = group->U[lid];
            const float lsT = group->sT[lid];
            const float lprevST = group->prevST[lid];
            
            float Isyn = 0;
             {
                // pull inSyn values in a coalesced access
                float linSyn = group->inSynInSyn0[lid];
                Isyn += linSyn; linSyn = 0;
                
                group->inSynInSyn0[lid] = linSyn;
            }
             {
                // pull inSyn values in a coalesced access
                float linSyn = group->inSynInSyn1[lid];
                Isyn += linSyn; linSyn = 0;
                
                group->inSynInSyn1[lid] = linSyn;
            }
            // current source 0
             {
                unsigned int lcsstartStim = group->startStimCS0[lid];
                const unsigned int lcsendStim = group->endStimCS0[lid];
                
                scalar current = (curand_uniform(&group->rng[lid]) * (6.50000000000000000e+00f) * 2.0f) - (6.50000000000000000e+00f);
                if(lcsstartStim != lcsendStim && t >= group->stimTimesCS0[lcsstartStim]) {
                   current += (4.00000000000000000e+01f);
                   lcsstartStim++;
                }
                Isyn += current;
                
                group->startStimCS0[lid] = lcsstartStim;
            }
            // test whether spike condition was fulfilled previously
            // calculate membrane potential
            if (lV >= 30.0f){
               lV=(-6.50000000000000000e+01f);
               lU+=(2.00000000000000000e+00f);
            } 
            lV+=0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+Isyn)*DT; //at two times for numerical stability
            lV+=0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+Isyn)*DT;
            lU+=(1.00000000000000006e-01f)*((2.00000000000000011e-01f)*lV-lU)*DT;
            if (lV > 30.0f){   //keep this to not confuse users with unrealistiv voltage values 
              lV=30.0f; 
            }
            
            // test for and register a true spike
            if (lV >= 29.99f) {
                const unsigned int spkIdx = atomicAdd(&shSpkCount, 1);
                shSpk[spkIdx] = lid;
                atomicOr(&shSpkRecord, 1 << threadIdx.x);
            }
            group->V[lid] = lV;
            group->U[lid] = lU;
        }
        __syncthreads();
        if(threadIdx.x == 0) {
            if (shSpkCount > 0) {
                shPosSpk = atomicAdd(&group->spkCnt[0], shSpkCount);
            }
        }
        __syncthreads();
        if(threadIdx.x < shSpkCount) {
            const unsigned int n = shSpk[threadIdx.x];
            group->spk[shPosSpk + threadIdx.x] = n;
            group->sT[n] = t;
        }
        if(threadIdx.x < 1) {
            const unsigned int numRecordingWords = (group->numNeurons + 31) / 32;
            const unsigned int popWordIdx = (lid / 32) + threadIdx.x;
            if(popWordIdx < numRecordingWords) {
                group->recordSpk[(recordingTimestep * numRecordingWords * 1) + popWordIdx] = shSpkRecord;
            }
        }
    }
    // merged1
    if(id >= 224 && id < 1024) {
        struct MergedNeuronUpdateGroup1 *group = &d_mergedNeuronUpdateGroup1[0]; 
        const unsigned int lid = id - 224;
        
        if(lid < group->numNeurons) {
            scalar lV = group->V[lid];
            scalar lU = group->U[lid];
            scalar lD = group->D[lid];
            const float lsT = group->sT[lid];
            const float lprevST = group->prevST[lid];
            const float lseT = group->seT[lid];
            const float lprevSET = group->prevSET[lid];
            
            float Isyn = 0;
             {
                // pull inSyn values in a coalesced access
                float linSyn = group->inSynInSyn0[lid];
                Isyn += linSyn; linSyn = 0;
                
                group->inSynInSyn0[lid] = linSyn;
            }
             {
                // pull inSyn values in a coalesced access
                float linSyn = group->inSynInSyn1[lid];
                Isyn += linSyn; linSyn = 0;
                
                group->inSynInSyn1[lid] = linSyn;
            }
            // current source 0
             {
                unsigned int lcsstartStim = group->startStimCS0[lid];
                const unsigned int lcsendStim = group->endStimCS0[lid];
                
                scalar current = (curand_uniform(&group->rng[lid]) * (6.50000000000000000e+00f) * 2.0f) - (6.50000000000000000e+00f);
                if(lcsstartStim != lcsendStim && t >= group->stimTimesCS0[lcsstartStim]) {
                   current += (4.00000000000000000e+01f);
                   lcsstartStim++;
                }
                Isyn += current;
                
                group->startStimCS0[lid] = lcsstartStim;
            }
            // test whether spike condition was fulfilled previously
            const bool oldSpike = (lV >= 30.0f);
            // calculate membrane potential
            
            lV+=0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+Isyn)*DT; //at two times for numerical stability
            lV+=0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+Isyn)*DT;
            lU+=(2.00000000000000004e-02f)*((2.00000000000000011e-01f)*lV-lU)*DT;
            const unsigned int timestep = (unsigned int)(t / DT);
            const bool injectDopamine = ((group->rewardTimesteps[timestep / 32] & (1 << (timestep % 32))) != 0);
            if(injectDopamine) {
               const scalar dopamineDT = t - lprevSET;
               const scalar dopamineDecay = exp(-dopamineDT / (2.00000000000000000e+02f));
               lD = (lD * dopamineDecay) + (5.00000000000000000e-01f);
            }
            
            
            bool spikeLikeEvent = false;
             {
                spikeLikeEvent |= (injectDopamine);
            }
            // register a spike-like event
            if (spikeLikeEvent) {
                const unsigned int spkEvntIdx = atomicAdd(&shSpkEvntCount, 1);
                shSpkEvnt[spkEvntIdx] = lid;
            }
            // test for and register a true spike
            if ((lV >= 30.0f) && !(oldSpike)) {
                const unsigned int spkIdx = atomicAdd(&shSpkCount, 1);
                shSpk[spkIdx] = lid;
                atomicOr(&shSpkRecord, 1 << threadIdx.x);
                // spike reset code
                
                lV=(-6.50000000000000000e+01f);
                lU+=(8.00000000000000000e+00f);
                
            }
            group->V[lid] = lV;
            group->U[lid] = lU;
            group->D[lid] = lD;
        }
        __syncthreads();
        if (threadIdx.x == 1) {
            if (shSpkEvntCount > 0) {
                shPosSpkEvnt = atomicAdd(&group->spkCntEvnt[0], shSpkEvntCount);
            }
        }
        __syncthreads();
        if(threadIdx.x == 0) {
            if (shSpkCount > 0) {
                shPosSpk = atomicAdd(&group->spkCnt[0], shSpkCount);
            }
        }
        __syncthreads();
        if(threadIdx.x < shSpkEvntCount) {
            const unsigned int n = shSpkEvnt[threadIdx.x];
            group->spkEvnt[shPosSpkEvnt + threadIdx.x] = n;
            group->seT[n] = t;
        }
        if(threadIdx.x < shSpkCount) {
            const unsigned int n = shSpk[threadIdx.x];
            group->spk[shPosSpk + threadIdx.x] = n;
            group->sT[n] = t;
        }
        if(threadIdx.x < 1) {
            const unsigned int numRecordingWords = (group->numNeurons + 31) / 32;
            const unsigned int popWordIdx = (lid / 32) + threadIdx.x;
            if(popWordIdx < numRecordingWords) {
                group->recordSpk[(recordingTimestep * numRecordingWords * 1) + popWordIdx] = shSpkRecord;
            }
        }
    }
}
void updateNeurons(float t, unsigned int recordingTimestep) {
     {
        const dim3 threads(32, 1);
        const dim3 grid(32, 1);
        neuronPrevSpikeTimeUpdateKernel<<<grid, threads>>>(t);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
     {
        const dim3 threads(32, 1);
        const dim3 grid(1, 1);
        neuronSpikeQueueUpdateKernel<<<grid, threads>>>();
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
     {
        const dim3 threads(32, 1);
        const dim3 grid(32, 1);
        updateNeuronsKernel<<<grid, threads>>>(t, recordingTimestep);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}
