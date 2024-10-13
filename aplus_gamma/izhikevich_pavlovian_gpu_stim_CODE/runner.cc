#include "definitionsInternal.h"

extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
unsigned long long iT;
float t;
unsigned long long numRecordingTimesteps = 0;
__device__ curandStatePhilox4_32_10_t d_rng;

// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
double initTime = 0.0;
double initSparseTime = 0.0;
double neuronUpdateTime = 0.0;
double presynapticUpdateTime = 0.0;
double postsynapticUpdateTime = 0.0;
double synapseDynamicsTime = 0.0;
// ------------------------------------------------------------------------
// merged group arrays
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// local neuron groups
// ------------------------------------------------------------------------
unsigned int* glbSpkCntE;
unsigned int* d_glbSpkCntE;
unsigned int* glbSpkE;
unsigned int* d_glbSpkE;
uint32_t* recordSpkE;
uint32_t* d_recordSpkE;
unsigned int* glbSpkCntEvntE;
unsigned int* d_glbSpkCntEvntE;
unsigned int* glbSpkEvntE;
unsigned int* d_glbSpkEvntE;
float* sTE;
float* d_sTE;
float* prevSTE;
float* d_prevSTE;
float* seTE;
float* d_seTE;
float* prevSETE;
float* d_prevSETE;
curandState* d_rngE;
scalar* VE;
scalar* d_VE;
scalar* UE;
scalar* d_UE;
scalar* DE;
scalar* d_DE;
uint32_t* rewardTimestepsE;
uint32_t* d_rewardTimestepsE;
// current source variables
unsigned int* startStimECurr;
unsigned int* d_startStimECurr;
unsigned int* endStimECurr;
unsigned int* d_endStimECurr;
scalar* stimTimesECurr;
scalar* d_stimTimesECurr;
unsigned int* glbSpkCntI;
unsigned int* d_glbSpkCntI;
unsigned int* glbSpkI;
unsigned int* d_glbSpkI;
uint32_t* recordSpkI;
uint32_t* d_recordSpkI;
float* sTI;
float* d_sTI;
float* prevSTI;
float* d_prevSTI;
curandState* d_rngI;
scalar* VI;
scalar* d_VI;
scalar* UI;
scalar* d_UI;
// current source variables
unsigned int* startStimICurr;
unsigned int* d_startStimICurr;
unsigned int* endStimICurr;
unsigned int* d_endStimICurr;
scalar* stimTimesICurr;
scalar* d_stimTimesICurr;

// ------------------------------------------------------------------------
// custom update variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// pre and postsynaptic variables
// ------------------------------------------------------------------------
float* inSynIE;
float* d_inSynIE;
float* inSynEE;
float* d_inSynEE;
float* inSynII;
float* d_inSynII;
float* inSynEI;
float* d_inSynEI;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------
const unsigned int maxRowLengthEE = 127;
unsigned int* rowLengthEE;
unsigned int* d_rowLengthEE;
uint16_t* indEE;
uint16_t* d_indEE;
unsigned int* d_colLengthEE;
unsigned int* d_remapEE;
const unsigned int maxRowLengthEI = 45;
unsigned int* rowLengthEI;
unsigned int* d_rowLengthEI;
uint8_t* indEI;
uint8_t* d_indEI;
unsigned int* d_colLengthEI;
unsigned int* d_remapEI;
const unsigned int maxRowLengthIE = 124;
unsigned int* rowLengthIE;
unsigned int* d_rowLengthIE;
uint16_t* indIE;
uint16_t* d_indIE;
const unsigned int maxRowLengthII = 43;
unsigned int* rowLengthII;
unsigned int* d_rowLengthII;
uint8_t* indII;
uint8_t* d_indII;

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------
scalar* tauPlusEE;
scalar* d_tauPlusEE;
scalar* tauMinusEE;
scalar* d_tauMinusEE;
scalar* tauCEE;
scalar* d_tauCEE;
scalar* tauDEE;
scalar* d_tauDEE;
scalar* aPlusEE;
scalar* d_aPlusEE;
scalar* aMinusEE;
scalar* d_aMinusEE;
scalar* wMinEE;
scalar* d_wMinEE;
scalar* wMaxEE;
scalar* d_wMaxEE;
scalar* gEE;
scalar* d_gEE;
scalar* cEE;
scalar* d_cEE;
scalar* tauPlusEI;
scalar* d_tauPlusEI;
scalar* tauMinusEI;
scalar* d_tauMinusEI;
scalar* tauCEI;
scalar* d_tauCEI;
scalar* tauDEI;
scalar* d_tauDEI;
scalar* aPlusEI;
scalar* d_aPlusEI;
scalar* aMinusEI;
scalar* d_aMinusEI;
scalar* wMinEI;
scalar* d_wMinEI;
scalar* wMaxEI;
scalar* d_wMaxEI;
scalar* gEI;
scalar* d_gEI;
scalar* cEI;
scalar* d_cEI;

}  // extern "C"
// ------------------------------------------------------------------------
// extra global params
// ------------------------------------------------------------------------
void allocaterewardTimestepsE(unsigned int count) {
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rewardTimestepsE, count * sizeof(uint32_t), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rewardTimestepsE, count * sizeof(uint32_t)));
    pushMergedNeuronUpdate1rewardTimestepsToDevice(0, d_rewardTimestepsE);
}
void freerewardTimestepsE() {
    CHECK_CUDA_ERRORS(cudaFreeHost(rewardTimestepsE));
    CHECK_CUDA_ERRORS(cudaFree(d_rewardTimestepsE));
}
void pushrewardTimestepsEToDevice(unsigned int count) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_rewardTimestepsE, rewardTimestepsE, count * sizeof(uint32_t), cudaMemcpyHostToDevice));
}
void pullrewardTimestepsEFromDevice(unsigned int count) {
    CHECK_CUDA_ERRORS(cudaMemcpy(rewardTimestepsE, d_rewardTimestepsE, count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}
void allocatestimTimesECurr(unsigned int count) {
    CHECK_CUDA_ERRORS(cudaHostAlloc(&stimTimesECurr, count * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_stimTimesECurr, count * sizeof(scalar)));
    pushMergedNeuronUpdate1stimTimesCS0ToDevice(0, d_stimTimesECurr);
}
void freestimTimesECurr() {
    CHECK_CUDA_ERRORS(cudaFreeHost(stimTimesECurr));
    CHECK_CUDA_ERRORS(cudaFree(d_stimTimesECurr));
}
void pushstimTimesECurrToDevice(unsigned int count) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_stimTimesECurr, stimTimesECurr, count * sizeof(scalar), cudaMemcpyHostToDevice));
}
void pullstimTimesECurrFromDevice(unsigned int count) {
    CHECK_CUDA_ERRORS(cudaMemcpy(stimTimesECurr, d_stimTimesECurr, count * sizeof(scalar), cudaMemcpyDeviceToHost));
}
void allocatestimTimesICurr(unsigned int count) {
    CHECK_CUDA_ERRORS(cudaHostAlloc(&stimTimesICurr, count * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_stimTimesICurr, count * sizeof(scalar)));
    pushMergedNeuronUpdate0stimTimesCS0ToDevice(0, d_stimTimesICurr);
}
void freestimTimesICurr() {
    CHECK_CUDA_ERRORS(cudaFreeHost(stimTimesICurr));
    CHECK_CUDA_ERRORS(cudaFree(d_stimTimesICurr));
}
void pushstimTimesICurrToDevice(unsigned int count) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_stimTimesICurr, stimTimesICurr, count * sizeof(scalar), cudaMemcpyHostToDevice));
}
void pullstimTimesICurrFromDevice(unsigned int count) {
    CHECK_CUDA_ERRORS(cudaMemcpy(stimTimesICurr, d_stimTimesICurr, count * sizeof(scalar), cudaMemcpyDeviceToHost));
}

// ------------------------------------------------------------------------
// copying things to device
// ------------------------------------------------------------------------
void pushESpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntE, glbSpkCntE, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkE, glbSpkE, 800 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushECurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntE, glbSpkCntE, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkE, glbSpkE, glbSpkCntE[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushESpikeEventsToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntEvntE, glbSpkCntEvntE, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkEvntE, glbSpkEvntE, 800 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushECurrentSpikeEventsToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntEvntE, glbSpkCntEvntE, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkEvntE, glbSpkEvntE, glbSpkCntEvntE[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushESpikeTimesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_sTE, sTE, 800 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushEPreviousSpikeTimesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_prevSTE, prevSTE, 800 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushESpikeEventTimesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_seTE, seTE, 800 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushEPreviousSpikeEventTimesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_prevSETE, prevSETE, 800 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushVEToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_VE, VE, 800 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentVEToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_VE, VE, 800 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushUEToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_UE, UE, 800 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentUEToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_UE, UE, 800 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushDEToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_DE, DE, 800 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentDEToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_DE, DE, 800 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushEStateToDevice(bool uninitialisedOnly) {
    pushVEToDevice(uninitialisedOnly);
    pushUEToDevice(uninitialisedOnly);
    pushDEToDevice(uninitialisedOnly);
}

void pushstartStimECurrToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_startStimECurr, startStimECurr, 800 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushendStimECurrToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_endStimECurr, endStimECurr, 800 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushECurrStateToDevice(bool uninitialisedOnly) {
    pushstartStimECurrToDevice(uninitialisedOnly);
    pushendStimECurrToDevice(uninitialisedOnly);
}

void pushISpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntI, glbSpkCntI, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkI, glbSpkI, 200 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushICurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntI, glbSpkCntI, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkI, glbSpkI, glbSpkCntI[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushISpikeTimesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_sTI, sTI, 200 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushIPreviousSpikeTimesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_prevSTI, prevSTI, 200 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushVIToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_VI, VI, 200 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentVIToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_VI, VI, 200 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushUIToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_UI, UI, 200 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentUIToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_UI, UI, 200 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushIStateToDevice(bool uninitialisedOnly) {
    pushVIToDevice(uninitialisedOnly);
    pushUIToDevice(uninitialisedOnly);
}

void pushstartStimICurrToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_startStimICurr, startStimICurr, 200 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushendStimICurrToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_endStimICurr, endStimICurr, 200 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushICurrStateToDevice(bool uninitialisedOnly) {
    pushstartStimICurrToDevice(uninitialisedOnly);
    pushendStimICurrToDevice(uninitialisedOnly);
}

void pushEEConnectivityToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_rowLengthEE, rowLengthEE, 800 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_indEE, indEE, 101600 * sizeof(uint16_t), cudaMemcpyHostToDevice));
    }
}

void pushEIConnectivityToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_rowLengthEI, rowLengthEI, 800 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_indEI, indEI, 36000 * sizeof(uint8_t), cudaMemcpyHostToDevice));
    }
}

void pushIEConnectivityToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_rowLengthIE, rowLengthIE, 200 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_indIE, indIE, 24800 * sizeof(uint16_t), cudaMemcpyHostToDevice));
    }
}

void pushIIConnectivityToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_rowLengthII, rowLengthII, 200 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_indII, indII, 8600 * sizeof(uint8_t), cudaMemcpyHostToDevice));
    }
}

void pushtauPlusEEToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_tauPlusEE, tauPlusEE, 101600 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushtauMinusEEToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_tauMinusEE, tauMinusEE, 101600 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushtauCEEToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_tauCEE, tauCEE, 101600 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushtauDEEToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_tauDEE, tauDEE, 101600 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushaPlusEEToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_aPlusEE, aPlusEE, 101600 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushaMinusEEToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_aMinusEE, aMinusEE, 101600 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushwMinEEToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_wMinEE, wMinEE, 101600 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushwMaxEEToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_wMaxEE, wMaxEE, 101600 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushgEEToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_gEE, gEE, 101600 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushcEEToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_cEE, cEE, 101600 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushinSynEEToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynEE, inSynEE, 800 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushEEStateToDevice(bool uninitialisedOnly) {
    pushtauPlusEEToDevice(uninitialisedOnly);
    pushtauMinusEEToDevice(uninitialisedOnly);
    pushtauCEEToDevice(uninitialisedOnly);
    pushtauDEEToDevice(uninitialisedOnly);
    pushaPlusEEToDevice(uninitialisedOnly);
    pushaMinusEEToDevice(uninitialisedOnly);
    pushwMinEEToDevice(uninitialisedOnly);
    pushwMaxEEToDevice(uninitialisedOnly);
    pushgEEToDevice(uninitialisedOnly);
    pushcEEToDevice(uninitialisedOnly);
    pushinSynEEToDevice(uninitialisedOnly);
}

void pushtauPlusEIToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_tauPlusEI, tauPlusEI, 36000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushtauMinusEIToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_tauMinusEI, tauMinusEI, 36000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushtauCEIToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_tauCEI, tauCEI, 36000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushtauDEIToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_tauDEI, tauDEI, 36000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushaPlusEIToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_aPlusEI, aPlusEI, 36000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushaMinusEIToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_aMinusEI, aMinusEI, 36000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushwMinEIToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_wMinEI, wMinEI, 36000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushwMaxEIToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_wMaxEI, wMaxEI, 36000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushgEIToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_gEI, gEI, 36000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushcEIToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_cEI, cEI, 36000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushinSynEIToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynEI, inSynEI, 200 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushEIStateToDevice(bool uninitialisedOnly) {
    pushtauPlusEIToDevice(uninitialisedOnly);
    pushtauMinusEIToDevice(uninitialisedOnly);
    pushtauCEIToDevice(uninitialisedOnly);
    pushtauDEIToDevice(uninitialisedOnly);
    pushaPlusEIToDevice(uninitialisedOnly);
    pushaMinusEIToDevice(uninitialisedOnly);
    pushwMinEIToDevice(uninitialisedOnly);
    pushwMaxEIToDevice(uninitialisedOnly);
    pushgEIToDevice(uninitialisedOnly);
    pushcEIToDevice(uninitialisedOnly);
    pushinSynEIToDevice(uninitialisedOnly);
}

void pushinSynIEToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynIE, inSynIE, 800 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushIEStateToDevice(bool uninitialisedOnly) {
    pushinSynIEToDevice(uninitialisedOnly);
}

void pushinSynIIToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynII, inSynII, 200 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushIIStateToDevice(bool uninitialisedOnly) {
    pushinSynIIToDevice(uninitialisedOnly);
}


// ------------------------------------------------------------------------
// copying things from device
// ------------------------------------------------------------------------
void pullESpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntE, d_glbSpkCntE, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkE, d_glbSpkE, 800 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullECurrentSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntE, d_glbSpkCntE, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkE, d_glbSpkE, glbSpkCntE[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullESpikeEventsFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntEvntE, d_glbSpkCntEvntE, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkEvntE, d_glbSpkEvntE, 800 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullECurrentSpikeEventsFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntEvntE, d_glbSpkCntEvntE, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkEvntE, d_glbSpkEvntE, glbSpkCntEvntE[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullESpikeTimesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(sTE, d_sTE, 800 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullEPreviousSpikeTimesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(prevSTE, d_prevSTE, 800 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullESpikeEventTimesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(seTE, d_seTE, 800 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullEPreviousSpikeEventTimesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(prevSETE, d_prevSETE, 800 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullVEFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(VE, d_VE, 800 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentVEFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(VE, d_VE, 800 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullUEFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(UE, d_UE, 800 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentUEFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(UE, d_UE, 800 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullDEFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(DE, d_DE, 800 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentDEFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(DE, d_DE, 800 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullEStateFromDevice() {
    pullVEFromDevice();
    pullUEFromDevice();
    pullDEFromDevice();
}

void pullstartStimECurrFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(startStimECurr, d_startStimECurr, 800 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullendStimECurrFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(endStimECurr, d_endStimECurr, 800 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullECurrStateFromDevice() {
    pullstartStimECurrFromDevice();
    pullendStimECurrFromDevice();
}

void pullISpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntI, d_glbSpkCntI, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkI, d_glbSpkI, 200 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullICurrentSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntI, d_glbSpkCntI, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkI, d_glbSpkI, glbSpkCntI[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullISpikeTimesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(sTI, d_sTI, 200 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullIPreviousSpikeTimesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(prevSTI, d_prevSTI, 200 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullVIFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(VI, d_VI, 200 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentVIFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(VI, d_VI, 200 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullUIFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(UI, d_UI, 200 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentUIFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(UI, d_UI, 200 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullIStateFromDevice() {
    pullVIFromDevice();
    pullUIFromDevice();
}

void pullstartStimICurrFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(startStimICurr, d_startStimICurr, 200 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullendStimICurrFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(endStimICurr, d_endStimICurr, 200 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullICurrStateFromDevice() {
    pullstartStimICurrFromDevice();
    pullendStimICurrFromDevice();
}

void pullEEConnectivityFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rowLengthEE, d_rowLengthEE, 800 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(indEE, d_indEE, 101600 * sizeof(uint16_t), cudaMemcpyDeviceToHost));
}

void pullEIConnectivityFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rowLengthEI, d_rowLengthEI, 800 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(indEI, d_indEI, 36000 * sizeof(uint8_t), cudaMemcpyDeviceToHost));
}

void pullIEConnectivityFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rowLengthIE, d_rowLengthIE, 200 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(indIE, d_indIE, 24800 * sizeof(uint16_t), cudaMemcpyDeviceToHost));
}

void pullIIConnectivityFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rowLengthII, d_rowLengthII, 200 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(indII, d_indII, 8600 * sizeof(uint8_t), cudaMemcpyDeviceToHost));
}

void pulltauPlusEEFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(tauPlusEE, d_tauPlusEE, 101600 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pulltauMinusEEFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(tauMinusEE, d_tauMinusEE, 101600 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pulltauCEEFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(tauCEE, d_tauCEE, 101600 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pulltauDEEFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(tauDEE, d_tauDEE, 101600 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullaPlusEEFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(aPlusEE, d_aPlusEE, 101600 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullaMinusEEFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(aMinusEE, d_aMinusEE, 101600 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullwMinEEFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(wMinEE, d_wMinEE, 101600 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullwMaxEEFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(wMaxEE, d_wMaxEE, 101600 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullgEEFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(gEE, d_gEE, 101600 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullcEEFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(cEE, d_cEE, 101600 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullinSynEEFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynEE, d_inSynEE, 800 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullEEStateFromDevice() {
    pulltauPlusEEFromDevice();
    pulltauMinusEEFromDevice();
    pulltauCEEFromDevice();
    pulltauDEEFromDevice();
    pullaPlusEEFromDevice();
    pullaMinusEEFromDevice();
    pullwMinEEFromDevice();
    pullwMaxEEFromDevice();
    pullgEEFromDevice();
    pullcEEFromDevice();
    pullinSynEEFromDevice();
}

void pulltauPlusEIFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(tauPlusEI, d_tauPlusEI, 36000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pulltauMinusEIFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(tauMinusEI, d_tauMinusEI, 36000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pulltauCEIFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(tauCEI, d_tauCEI, 36000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pulltauDEIFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(tauDEI, d_tauDEI, 36000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullaPlusEIFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(aPlusEI, d_aPlusEI, 36000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullaMinusEIFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(aMinusEI, d_aMinusEI, 36000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullwMinEIFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(wMinEI, d_wMinEI, 36000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullwMaxEIFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(wMaxEI, d_wMaxEI, 36000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullgEIFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(gEI, d_gEI, 36000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullcEIFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(cEI, d_cEI, 36000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullinSynEIFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynEI, d_inSynEI, 200 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullEIStateFromDevice() {
    pulltauPlusEIFromDevice();
    pulltauMinusEIFromDevice();
    pulltauCEIFromDevice();
    pulltauDEIFromDevice();
    pullaPlusEIFromDevice();
    pullaMinusEIFromDevice();
    pullwMinEIFromDevice();
    pullwMaxEIFromDevice();
    pullgEIFromDevice();
    pullcEIFromDevice();
    pullinSynEIFromDevice();
}

void pullinSynIEFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynIE, d_inSynIE, 800 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullIEStateFromDevice() {
    pullinSynIEFromDevice();
}

void pullinSynIIFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynII, d_inSynII, 200 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullIIStateFromDevice() {
    pullinSynIIFromDevice();
}


// ------------------------------------------------------------------------
// helper getter functions
// ------------------------------------------------------------------------
unsigned int* getECurrentSpikes(unsigned int batch) {
    return (glbSpkE);
}

unsigned int& getECurrentSpikeCount(unsigned int batch) {
    return glbSpkCntE[0];
}

unsigned int* getECurrentSpikeEvents(unsigned int batch) {
    return (glbSpkEvntE);
}

unsigned int& getECurrentSpikeEventCount(unsigned int batch) {
    return glbSpkCntEvntE[0];
}

scalar* getCurrentVE(unsigned int batch) {
    return VE;
}

scalar* getCurrentUE(unsigned int batch) {
    return UE;
}

scalar* getCurrentDE(unsigned int batch) {
    return DE;
}

unsigned int* getICurrentSpikes(unsigned int batch) {
    return (glbSpkI);
}

unsigned int& getICurrentSpikeCount(unsigned int batch) {
    return glbSpkCntI[0];
}

scalar* getCurrentVI(unsigned int batch) {
    return VI;
}

scalar* getCurrentUI(unsigned int batch) {
    return UI;
}


void copyStateToDevice(bool uninitialisedOnly) {
    pushEStateToDevice(uninitialisedOnly);
    pushECurrStateToDevice(uninitialisedOnly);
    pushIStateToDevice(uninitialisedOnly);
    pushICurrStateToDevice(uninitialisedOnly);
    pushEEStateToDevice(uninitialisedOnly);
    pushEIStateToDevice(uninitialisedOnly);
    pushIEStateToDevice(uninitialisedOnly);
    pushIIStateToDevice(uninitialisedOnly);
}

void copyConnectivityToDevice(bool uninitialisedOnly) {
    pushEEConnectivityToDevice(uninitialisedOnly);
    pushEIConnectivityToDevice(uninitialisedOnly);
    pushIEConnectivityToDevice(uninitialisedOnly);
    pushIIConnectivityToDevice(uninitialisedOnly);
}

void copyStateFromDevice() {
    pullEStateFromDevice();
    pullECurrStateFromDevice();
    pullIStateFromDevice();
    pullICurrStateFromDevice();
    pullEEStateFromDevice();
    pullEIStateFromDevice();
    pullIEStateFromDevice();
    pullIIStateFromDevice();
}

void copyCurrentSpikesFromDevice() {
    pullECurrentSpikesFromDevice();
    pullICurrentSpikesFromDevice();
}

void copyCurrentSpikeEventsFromDevice() {
    pullECurrentSpikeEventsFromDevice();
}

void allocateRecordingBuffers(unsigned int timesteps) {
    numRecordingTimesteps = timesteps;
     {
        const unsigned int numWords = 25 * timesteps;
         {
            CHECK_CUDA_ERRORS(cudaHostAlloc(&recordSpkE, numWords * sizeof(uint32_t), cudaHostAllocPortable));
            CHECK_CUDA_ERRORS(cudaMalloc(&d_recordSpkE, numWords * sizeof(uint32_t)));
            pushMergedNeuronUpdate1recordSpkToDevice(0, d_recordSpkE);
        }
    }
     {
        const unsigned int numWords = 7 * timesteps;
         {
            CHECK_CUDA_ERRORS(cudaHostAlloc(&recordSpkI, numWords * sizeof(uint32_t), cudaHostAllocPortable));
            CHECK_CUDA_ERRORS(cudaMalloc(&d_recordSpkI, numWords * sizeof(uint32_t)));
            pushMergedNeuronUpdate0recordSpkToDevice(0, d_recordSpkI);
        }
    }
}

void pullRecordingBuffersFromDevice() {
    if(numRecordingTimesteps == 0) {
        throw std::runtime_error("Recording buffer not allocated - cannot pull from device");
    }
     {
        const unsigned int numWords = 25 * numRecordingTimesteps;
         {
            CHECK_CUDA_ERRORS(cudaMemcpy(recordSpkE, d_recordSpkE, numWords * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        }
    }
     {
        const unsigned int numWords = 7 * numRecordingTimesteps;
         {
            CHECK_CUDA_ERRORS(cudaMemcpy(recordSpkI, d_recordSpkI, numWords * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        }
    }
}

void allocateMem() {
    int deviceID;
    CHECK_CUDA_ERRORS(cudaDeviceGetByPCIBusId(&deviceID, "0000:01:00.0"));
    CHECK_CUDA_ERRORS(cudaSetDevice(deviceID));
    
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // timers
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // local neuron groups
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkCntE, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkCntE, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkE, 800 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkE, 800 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkCntEvntE, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkCntEvntE, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkEvntE, 800 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkEvntE, 800 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&sTE, 800 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_sTE, 800 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&prevSTE, 800 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_prevSTE, 800 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&seTE, 800 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_seTE, 800 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&prevSETE, 800 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_prevSETE, 800 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rngE, 800 * sizeof(curandState)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&VE, 800 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_VE, 800 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&UE, 800 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_UE, 800 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&DE, 800 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_DE, 800 * sizeof(scalar)));
    // current source variables
    CHECK_CUDA_ERRORS(cudaHostAlloc(&startStimECurr, 800 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_startStimECurr, 800 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&endStimECurr, 800 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_endStimECurr, 800 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkCntI, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkCntI, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkI, 200 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkI, 200 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&sTI, 200 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_sTI, 200 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&prevSTI, 200 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_prevSTI, 200 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rngI, 200 * sizeof(curandState)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&VI, 200 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_VI, 200 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&UI, 200 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_UI, 200 * sizeof(scalar)));
    // current source variables
    CHECK_CUDA_ERRORS(cudaHostAlloc(&startStimICurr, 200 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_startStimICurr, 200 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&endStimICurr, 200 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_endStimICurr, 200 * sizeof(unsigned int)));
    
    // ------------------------------------------------------------------------
    // custom update variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // pre and postsynaptic variables
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynIE, 800 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynIE, 800 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynEE, 800 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynEE, 800 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynII, 200 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynII, 200 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynEI, 200 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynEI, 200 * sizeof(float)));
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rowLengthEE, 800 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rowLengthEE, 800 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&indEE, 101600 * sizeof(uint16_t), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_indEE, 101600 * sizeof(uint16_t)));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_colLengthEE, 800 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_remapEE, 101600 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rowLengthEI, 800 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rowLengthEI, 800 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&indEI, 36000 * sizeof(uint8_t), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_indEI, 36000 * sizeof(uint8_t)));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_colLengthEI, 200 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_remapEI, 24800 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rowLengthIE, 200 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rowLengthIE, 200 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&indIE, 24800 * sizeof(uint16_t), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_indIE, 24800 * sizeof(uint16_t)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rowLengthII, 200 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rowLengthII, 200 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&indII, 8600 * sizeof(uint8_t), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_indII, 8600 * sizeof(uint8_t)));
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaHostAlloc(&tauPlusEE, 101600 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_tauPlusEE, 101600 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&tauMinusEE, 101600 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_tauMinusEE, 101600 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&tauCEE, 101600 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_tauCEE, 101600 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&tauDEE, 101600 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_tauDEE, 101600 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&aPlusEE, 101600 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_aPlusEE, 101600 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&aMinusEE, 101600 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_aMinusEE, 101600 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&wMinEE, 101600 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_wMinEE, 101600 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&wMaxEE, 101600 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_wMaxEE, 101600 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&gEE, 101600 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_gEE, 101600 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&cEE, 101600 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_cEE, 101600 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&tauPlusEI, 36000 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_tauPlusEI, 36000 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&tauMinusEI, 36000 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_tauMinusEI, 36000 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&tauCEI, 36000 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_tauCEI, 36000 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&tauDEI, 36000 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_tauDEI, 36000 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&aPlusEI, 36000 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_aPlusEI, 36000 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&aMinusEI, 36000 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_aMinusEI, 36000 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&wMinEI, 36000 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_wMinEI, 36000 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&wMaxEI, 36000 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_wMaxEI, 36000 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&gEI, 36000 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_gEI, 36000 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&cEI, 36000 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_cEI, 36000 * sizeof(scalar)));
    
    pushMergedNeuronInitGroup0ToDevice(0, d_glbSpkCntI, d_glbSpkI, d_sTI, d_prevSTI, d_rngI, d_VI, d_UI, d_inSynII, d_inSynEI, 200);
    pushMergedNeuronInitGroup1ToDevice(0, d_glbSpkCntE, d_glbSpkE, d_glbSpkCntEvntE, d_glbSpkEvntE, d_sTE, d_seTE, d_prevSTE, d_prevSETE, d_rngE, d_VE, d_UE, d_DE, d_inSynIE, d_inSynEE, 800);
    pushMergedSynapseConnectivityInitGroup0ToDevice(0, d_rowLengthIE, d_indIE, 200, 800, 124);
    pushMergedSynapseConnectivityInitGroup1ToDevice(0, d_rowLengthII, d_indII, 200, 200, 43);
    pushMergedSynapseConnectivityInitGroup2ToDevice(0, d_rowLengthEI, d_indEI, 800, 200, 45);
    pushMergedSynapseConnectivityInitGroup3ToDevice(0, d_rowLengthEE, d_indEE, 800, 800, 127);
    pushMergedSynapseSparseInitGroup0ToDevice(0, d_tauMinusEI, d_cEI, d_gEI, d_wMaxEI, d_wMinEI, d_aMinusEI, d_aPlusEI, d_tauDEI, d_tauCEI, d_tauPlusEI, d_remapEI, d_colLengthEI, d_indEI, d_rowLengthEI, 800, 124, 45, 200);
    pushMergedSynapseSparseInitGroup1ToDevice(0, d_tauMinusEE, d_cEE, d_gEE, d_wMaxEE, d_wMinEE, d_aMinusEE, d_aPlusEE, d_tauDEE, d_tauCEE, d_tauPlusEE, d_remapEE, d_colLengthEE, d_indEE, d_rowLengthEE, 800, 127, 127, 800);
    pushMergedNeuronUpdateGroup0ToDevice(0, d_glbSpkCntI, d_glbSpkI, d_sTI, d_prevSTI, d_rngI, d_VI, d_UI, d_inSynEI, d_inSynII, d_startStimICurr, d_endStimICurr, d_stimTimesICurr, d_recordSpkI, 200);
    pushMergedNeuronUpdateGroup1ToDevice(0, d_VE, d_recordSpkE, d_stimTimesECurr, d_endStimECurr, d_startStimECurr, d_inSynIE, d_inSynEE, d_rewardTimestepsE, d_DE, d_UE, d_rngE, d_prevSETE, d_prevSTE, d_seTE, d_sTE, d_glbSpkEvntE, d_glbSpkCntEvntE, d_glbSpkE, d_glbSpkCntE, 800);
    pushMergedPresynapticUpdateGroup0ToDevice(0, d_inSynII, d_glbSpkCntI, d_glbSpkI, d_rowLengthII, d_indII, 200, 200, 43);
    pushMergedPresynapticUpdateGroup1ToDevice(0, d_inSynIE, d_glbSpkCntI, d_glbSpkI, d_rowLengthIE, d_indIE, 200, 800, 124);
    pushMergedPresynapticUpdateGroup2ToDevice(0, d_prevSTI, d_cEI, d_gEI, d_wMaxEI, d_wMinEI, d_aMinusEI, d_aPlusEI, d_tauDEI, d_tauCEI, d_tauMinusEI, d_tauPlusEI, d_indEI, d_rowLengthEI, d_prevSETE, d_prevSTE, d_seTE, d_sTI, d_sTE, d_DE, d_glbSpkEvntE, d_glbSpkCntEvntE, d_glbSpkE, d_glbSpkCntE, d_inSynEI, 800, 45, 200);
    pushMergedPresynapticUpdateGroup3ToDevice(0, d_prevSTE, d_cEE, d_gEE, d_wMaxEE, d_wMinEE, d_aMinusEE, d_aPlusEE, d_tauDEE, d_tauCEE, d_tauMinusEE, d_tauPlusEE, d_indEE, d_rowLengthEE, d_prevSETE, d_prevSTE, d_seTE, d_sTE, d_sTE, d_DE, d_glbSpkEvntE, d_glbSpkCntEvntE, d_glbSpkE, d_glbSpkCntE, d_inSynEE, 800, 127, 800);
    pushMergedPostsynapticUpdateGroup0ToDevice(0, d_rowLengthEI, d_cEI, d_gEI, d_wMaxEI, d_wMinEI, d_aMinusEI, d_aPlusEI, d_tauDEI, d_tauCEI, d_tauMinusEI, d_tauPlusEI, d_remapEI, d_colLengthEI, d_indEI, d_prevSETE, d_prevSTI, d_prevSTE, d_seTE, d_sTI, d_sTE, d_DE, d_glbSpkI, d_glbSpkCntI, 800, 124, 45, 200);
    pushMergedPostsynapticUpdateGroup1ToDevice(0, d_rowLengthEE, d_cEE, d_gEE, d_wMaxEE, d_wMinEE, d_aMinusEE, d_aPlusEE, d_tauDEE, d_tauCEE, d_tauMinusEE, d_tauPlusEE, d_remapEE, d_colLengthEE, d_indEE, d_prevSETE, d_prevSTE, d_prevSTE, d_seTE, d_sTE, d_sTE, d_DE, d_glbSpkE, d_glbSpkCntE, 800, 127, 127, 800);
    pushMergedNeuronPrevSpikeTimeUpdateGroup0ToDevice(0, d_glbSpkCntI, d_glbSpkI, d_prevSTI, 200);
    pushMergedNeuronPrevSpikeTimeUpdateGroup1ToDevice(0, d_glbSpkCntE, d_glbSpkCntEvntE, d_glbSpkE, d_prevSTE, d_glbSpkEvntE, d_prevSETE, 800);
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(0, d_glbSpkCntI);
    pushMergedNeuronSpikeQueueUpdateGroup1ToDevice(0, d_glbSpkCntE, d_glbSpkCntEvntE);
}

void freeMem() {
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // timers
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // local neuron groups
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntE));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntE));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkE));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkE));
    CHECK_CUDA_ERRORS(cudaFreeHost(recordSpkE));
    CHECK_CUDA_ERRORS(cudaFree(d_recordSpkE));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntEvntE));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntEvntE));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkEvntE));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkEvntE));
    CHECK_CUDA_ERRORS(cudaFreeHost(sTE));
    CHECK_CUDA_ERRORS(cudaFree(d_sTE));
    CHECK_CUDA_ERRORS(cudaFreeHost(prevSTE));
    CHECK_CUDA_ERRORS(cudaFree(d_prevSTE));
    CHECK_CUDA_ERRORS(cudaFreeHost(seTE));
    CHECK_CUDA_ERRORS(cudaFree(d_seTE));
    CHECK_CUDA_ERRORS(cudaFreeHost(prevSETE));
    CHECK_CUDA_ERRORS(cudaFree(d_prevSETE));
    CHECK_CUDA_ERRORS(cudaFree(d_rngE));
    CHECK_CUDA_ERRORS(cudaFreeHost(VE));
    CHECK_CUDA_ERRORS(cudaFree(d_VE));
    CHECK_CUDA_ERRORS(cudaFreeHost(UE));
    CHECK_CUDA_ERRORS(cudaFree(d_UE));
    CHECK_CUDA_ERRORS(cudaFreeHost(DE));
    CHECK_CUDA_ERRORS(cudaFree(d_DE));
    // current source variables
    CHECK_CUDA_ERRORS(cudaFreeHost(startStimECurr));
    CHECK_CUDA_ERRORS(cudaFree(d_startStimECurr));
    CHECK_CUDA_ERRORS(cudaFreeHost(endStimECurr));
    CHECK_CUDA_ERRORS(cudaFree(d_endStimECurr));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntI));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntI));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkI));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkI));
    CHECK_CUDA_ERRORS(cudaFreeHost(recordSpkI));
    CHECK_CUDA_ERRORS(cudaFree(d_recordSpkI));
    CHECK_CUDA_ERRORS(cudaFreeHost(sTI));
    CHECK_CUDA_ERRORS(cudaFree(d_sTI));
    CHECK_CUDA_ERRORS(cudaFreeHost(prevSTI));
    CHECK_CUDA_ERRORS(cudaFree(d_prevSTI));
    CHECK_CUDA_ERRORS(cudaFree(d_rngI));
    CHECK_CUDA_ERRORS(cudaFreeHost(VI));
    CHECK_CUDA_ERRORS(cudaFree(d_VI));
    CHECK_CUDA_ERRORS(cudaFreeHost(UI));
    CHECK_CUDA_ERRORS(cudaFree(d_UI));
    // current source variables
    CHECK_CUDA_ERRORS(cudaFreeHost(startStimICurr));
    CHECK_CUDA_ERRORS(cudaFree(d_startStimICurr));
    CHECK_CUDA_ERRORS(cudaFreeHost(endStimICurr));
    CHECK_CUDA_ERRORS(cudaFree(d_endStimICurr));
    
    // ------------------------------------------------------------------------
    // custom update variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // pre and postsynaptic variables
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynIE));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynIE));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynEE));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynEE));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynII));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynII));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynEI));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynEI));
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaFreeHost(rowLengthEE));
    CHECK_CUDA_ERRORS(cudaFree(d_rowLengthEE));
    CHECK_CUDA_ERRORS(cudaFreeHost(indEE));
    CHECK_CUDA_ERRORS(cudaFree(d_indEE));
    CHECK_CUDA_ERRORS(cudaFree(d_colLengthEE));
    CHECK_CUDA_ERRORS(cudaFree(d_remapEE));
    CHECK_CUDA_ERRORS(cudaFreeHost(rowLengthEI));
    CHECK_CUDA_ERRORS(cudaFree(d_rowLengthEI));
    CHECK_CUDA_ERRORS(cudaFreeHost(indEI));
    CHECK_CUDA_ERRORS(cudaFree(d_indEI));
    CHECK_CUDA_ERRORS(cudaFree(d_colLengthEI));
    CHECK_CUDA_ERRORS(cudaFree(d_remapEI));
    CHECK_CUDA_ERRORS(cudaFreeHost(rowLengthIE));
    CHECK_CUDA_ERRORS(cudaFree(d_rowLengthIE));
    CHECK_CUDA_ERRORS(cudaFreeHost(indIE));
    CHECK_CUDA_ERRORS(cudaFree(d_indIE));
    CHECK_CUDA_ERRORS(cudaFreeHost(rowLengthII));
    CHECK_CUDA_ERRORS(cudaFree(d_rowLengthII));
    CHECK_CUDA_ERRORS(cudaFreeHost(indII));
    CHECK_CUDA_ERRORS(cudaFree(d_indII));
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaFreeHost(tauPlusEE));
    CHECK_CUDA_ERRORS(cudaFree(d_tauPlusEE));
    CHECK_CUDA_ERRORS(cudaFreeHost(tauMinusEE));
    CHECK_CUDA_ERRORS(cudaFree(d_tauMinusEE));
    CHECK_CUDA_ERRORS(cudaFreeHost(tauCEE));
    CHECK_CUDA_ERRORS(cudaFree(d_tauCEE));
    CHECK_CUDA_ERRORS(cudaFreeHost(tauDEE));
    CHECK_CUDA_ERRORS(cudaFree(d_tauDEE));
    CHECK_CUDA_ERRORS(cudaFreeHost(aPlusEE));
    CHECK_CUDA_ERRORS(cudaFree(d_aPlusEE));
    CHECK_CUDA_ERRORS(cudaFreeHost(aMinusEE));
    CHECK_CUDA_ERRORS(cudaFree(d_aMinusEE));
    CHECK_CUDA_ERRORS(cudaFreeHost(wMinEE));
    CHECK_CUDA_ERRORS(cudaFree(d_wMinEE));
    CHECK_CUDA_ERRORS(cudaFreeHost(wMaxEE));
    CHECK_CUDA_ERRORS(cudaFree(d_wMaxEE));
    CHECK_CUDA_ERRORS(cudaFreeHost(gEE));
    CHECK_CUDA_ERRORS(cudaFree(d_gEE));
    CHECK_CUDA_ERRORS(cudaFreeHost(cEE));
    CHECK_CUDA_ERRORS(cudaFree(d_cEE));
    CHECK_CUDA_ERRORS(cudaFreeHost(tauPlusEI));
    CHECK_CUDA_ERRORS(cudaFree(d_tauPlusEI));
    CHECK_CUDA_ERRORS(cudaFreeHost(tauMinusEI));
    CHECK_CUDA_ERRORS(cudaFree(d_tauMinusEI));
    CHECK_CUDA_ERRORS(cudaFreeHost(tauCEI));
    CHECK_CUDA_ERRORS(cudaFree(d_tauCEI));
    CHECK_CUDA_ERRORS(cudaFreeHost(tauDEI));
    CHECK_CUDA_ERRORS(cudaFree(d_tauDEI));
    CHECK_CUDA_ERRORS(cudaFreeHost(aPlusEI));
    CHECK_CUDA_ERRORS(cudaFree(d_aPlusEI));
    CHECK_CUDA_ERRORS(cudaFreeHost(aMinusEI));
    CHECK_CUDA_ERRORS(cudaFree(d_aMinusEI));
    CHECK_CUDA_ERRORS(cudaFreeHost(wMinEI));
    CHECK_CUDA_ERRORS(cudaFree(d_wMinEI));
    CHECK_CUDA_ERRORS(cudaFreeHost(wMaxEI));
    CHECK_CUDA_ERRORS(cudaFree(d_wMaxEI));
    CHECK_CUDA_ERRORS(cudaFreeHost(gEI));
    CHECK_CUDA_ERRORS(cudaFree(d_gEI));
    CHECK_CUDA_ERRORS(cudaFreeHost(cEI));
    CHECK_CUDA_ERRORS(cudaFree(d_cEI));
    
}

size_t getFreeDeviceMemBytes() {
    size_t free;
    size_t total;
    CHECK_CUDA_ERRORS(cudaMemGetInfo(&free, &total));
    return free;
}

void stepTime() {
    updateSynapses(t);
    updateNeurons(t, (unsigned int)(iT % numRecordingTimesteps)); 
    iT++;
    t = iT*DT;
}

