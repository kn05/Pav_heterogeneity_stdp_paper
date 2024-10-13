#pragma once
#define EXPORT_VAR extern
#define EXPORT_FUNC
// Standard C++ includes
#include <random>
#include <string>
#include <stdexcept>

// Standard C includes
#include <cassert>
#include <cstdint>
#define DT 1.00000000000000000e+00f
typedef float scalar;
#define SCALAR_MIN 1.175494351e-38f
#define SCALAR_MAX 3.402823466e+38f

#define TIME_MIN 1.175494351e-38f
#define TIME_MAX 3.402823466e+38f

// ------------------------------------------------------------------------
// bit tool macros
#define B(x,i) ((x) & (0x80000000 >> (i))) //!< Extract the bit at the specified position i from x
#define setB(x,i) x= ((x) | (0x80000000 >> (i))) //!< Set the bit at the specified position i in x to 1
#define delB(x,i) x= ((x) & (~(0x80000000 >> (i)))) //!< Set the bit at the specified position i in x to 0

extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
EXPORT_VAR unsigned long long iT;
EXPORT_VAR float t;

// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
EXPORT_VAR double initTime;
EXPORT_VAR double initSparseTime;
EXPORT_VAR double neuronUpdateTime;
EXPORT_VAR double presynapticUpdateTime;
EXPORT_VAR double postsynapticUpdateTime;
EXPORT_VAR double synapseDynamicsTime;
// ------------------------------------------------------------------------
// local neuron groups
// ------------------------------------------------------------------------
#define spikeCount_E glbSpkCntE[0]
#define spike_E glbSpkE
#define glbSpkShiftE 0

EXPORT_VAR unsigned int* glbSpkCntE;
EXPORT_VAR unsigned int* d_glbSpkCntE;
EXPORT_VAR unsigned int* glbSpkE;
EXPORT_VAR unsigned int* d_glbSpkE;
EXPORT_VAR uint32_t* recordSpkE;
EXPORT_VAR uint32_t* d_recordSpkE;
#define spikeEventCount_E glbSpkCntEvntE[0]
#define spikeEvent_E glbSpkEvntE


EXPORT_VAR unsigned int* glbSpkCntEvntE;
EXPORT_VAR unsigned int* d_glbSpkCntEvntE;
EXPORT_VAR unsigned int* glbSpkEvntE;
EXPORT_VAR unsigned int* d_glbSpkEvntE;
EXPORT_VAR float* sTE;
EXPORT_VAR float* d_sTE;
EXPORT_VAR float* prevSTE;
EXPORT_VAR float* d_prevSTE;
EXPORT_VAR float* seTE;
EXPORT_VAR float* d_seTE;
EXPORT_VAR float* prevSETE;
EXPORT_VAR float* d_prevSETE;
EXPORT_VAR scalar* VE;
EXPORT_VAR scalar* d_VE;
EXPORT_VAR scalar* UE;
EXPORT_VAR scalar* d_UE;
EXPORT_VAR scalar* DE;
EXPORT_VAR scalar* d_DE;
EXPORT_VAR uint32_t* rewardTimestepsE;
EXPORT_VAR uint32_t* d_rewardTimestepsE;
// current source variables
EXPORT_VAR unsigned int* startStimECurr;
EXPORT_VAR unsigned int* d_startStimECurr;
EXPORT_VAR unsigned int* endStimECurr;
EXPORT_VAR unsigned int* d_endStimECurr;
EXPORT_VAR scalar* stimTimesECurr;
EXPORT_VAR scalar* d_stimTimesECurr;
#define spikeCount_I glbSpkCntI[0]
#define spike_I glbSpkI
#define glbSpkShiftI 0

EXPORT_VAR unsigned int* glbSpkCntI;
EXPORT_VAR unsigned int* d_glbSpkCntI;
EXPORT_VAR unsigned int* glbSpkI;
EXPORT_VAR unsigned int* d_glbSpkI;
EXPORT_VAR uint32_t* recordSpkI;
EXPORT_VAR uint32_t* d_recordSpkI;
EXPORT_VAR float* sTI;
EXPORT_VAR float* d_sTI;
EXPORT_VAR float* prevSTI;
EXPORT_VAR float* d_prevSTI;
EXPORT_VAR scalar* VI;
EXPORT_VAR scalar* d_VI;
EXPORT_VAR scalar* UI;
EXPORT_VAR scalar* d_UI;
// current source variables
EXPORT_VAR unsigned int* startStimICurr;
EXPORT_VAR unsigned int* d_startStimICurr;
EXPORT_VAR unsigned int* endStimICurr;
EXPORT_VAR unsigned int* d_endStimICurr;
EXPORT_VAR scalar* stimTimesICurr;
EXPORT_VAR scalar* d_stimTimesICurr;

// ------------------------------------------------------------------------
// custom update variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// pre and postsynaptic variables
// ------------------------------------------------------------------------
EXPORT_VAR float* inSynIE;
EXPORT_VAR float* d_inSynIE;
EXPORT_VAR float* inSynEE;
EXPORT_VAR float* d_inSynEE;
EXPORT_VAR float* inSynII;
EXPORT_VAR float* d_inSynII;
EXPORT_VAR float* inSynEI;
EXPORT_VAR float* d_inSynEI;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------
EXPORT_VAR const unsigned int maxRowLengthEE;
EXPORT_VAR unsigned int* rowLengthEE;
EXPORT_VAR unsigned int* d_rowLengthEE;
EXPORT_VAR uint16_t* indEE;
EXPORT_VAR uint16_t* d_indEE;
EXPORT_VAR unsigned int* d_colLengthEE;
EXPORT_VAR unsigned int* d_remapEE;
EXPORT_VAR const unsigned int maxRowLengthEI;
EXPORT_VAR unsigned int* rowLengthEI;
EXPORT_VAR unsigned int* d_rowLengthEI;
EXPORT_VAR uint8_t* indEI;
EXPORT_VAR uint8_t* d_indEI;
EXPORT_VAR unsigned int* d_colLengthEI;
EXPORT_VAR unsigned int* d_remapEI;
EXPORT_VAR const unsigned int maxRowLengthIE;
EXPORT_VAR unsigned int* rowLengthIE;
EXPORT_VAR unsigned int* d_rowLengthIE;
EXPORT_VAR uint16_t* indIE;
EXPORT_VAR uint16_t* d_indIE;
EXPORT_VAR const unsigned int maxRowLengthII;
EXPORT_VAR unsigned int* rowLengthII;
EXPORT_VAR unsigned int* d_rowLengthII;
EXPORT_VAR uint8_t* indII;
EXPORT_VAR uint8_t* d_indII;

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------
EXPORT_VAR scalar* tauPlusEE;
EXPORT_VAR scalar* d_tauPlusEE;
EXPORT_VAR scalar* tauMinusEE;
EXPORT_VAR scalar* d_tauMinusEE;
EXPORT_VAR scalar* tauCEE;
EXPORT_VAR scalar* d_tauCEE;
EXPORT_VAR scalar* tauDEE;
EXPORT_VAR scalar* d_tauDEE;
EXPORT_VAR scalar* aPlusEE;
EXPORT_VAR scalar* d_aPlusEE;
EXPORT_VAR scalar* aMinusEE;
EXPORT_VAR scalar* d_aMinusEE;
EXPORT_VAR scalar* wMinEE;
EXPORT_VAR scalar* d_wMinEE;
EXPORT_VAR scalar* wMaxEE;
EXPORT_VAR scalar* d_wMaxEE;
EXPORT_VAR scalar* gEE;
EXPORT_VAR scalar* d_gEE;
EXPORT_VAR scalar* cEE;
EXPORT_VAR scalar* d_cEE;
EXPORT_VAR scalar* tauPlusEI;
EXPORT_VAR scalar* d_tauPlusEI;
EXPORT_VAR scalar* tauMinusEI;
EXPORT_VAR scalar* d_tauMinusEI;
EXPORT_VAR scalar* tauCEI;
EXPORT_VAR scalar* d_tauCEI;
EXPORT_VAR scalar* tauDEI;
EXPORT_VAR scalar* d_tauDEI;
EXPORT_VAR scalar* aPlusEI;
EXPORT_VAR scalar* d_aPlusEI;
EXPORT_VAR scalar* aMinusEI;
EXPORT_VAR scalar* d_aMinusEI;
EXPORT_VAR scalar* wMinEI;
EXPORT_VAR scalar* d_wMinEI;
EXPORT_VAR scalar* wMaxEI;
EXPORT_VAR scalar* d_wMaxEI;
EXPORT_VAR scalar* gEI;
EXPORT_VAR scalar* d_gEI;
EXPORT_VAR scalar* cEI;
EXPORT_VAR scalar* d_cEI;

EXPORT_FUNC void pushESpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullESpikesFromDevice();
EXPORT_FUNC void pushECurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullECurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getECurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getECurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushESpikeEventsToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullESpikeEventsFromDevice();
EXPORT_FUNC void pushECurrentSpikeEventsToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullECurrentSpikeEventsFromDevice();
EXPORT_FUNC unsigned int* getECurrentSpikeEvents(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getECurrentSpikeEventCount(unsigned int batch = 0); 
EXPORT_FUNC void pushESpikeTimesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullESpikeTimesFromDevice();
EXPORT_FUNC void pushEPreviousSpikeTimesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullEPreviousSpikeTimesFromDevice();
EXPORT_FUNC void pushESpikeEventTimesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullESpikeEventTimesFromDevice();
EXPORT_FUNC void pushEPreviousSpikeEventTimesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullEPreviousSpikeEventTimesFromDevice();
EXPORT_FUNC void pushVEToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVEFromDevice();
EXPORT_FUNC void pushCurrentVEToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentVEFromDevice();
EXPORT_FUNC scalar* getCurrentVE(unsigned int batch = 0); 
EXPORT_FUNC void pushUEToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullUEFromDevice();
EXPORT_FUNC void pushCurrentUEToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentUEFromDevice();
EXPORT_FUNC scalar* getCurrentUE(unsigned int batch = 0); 
EXPORT_FUNC void pushDEToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullDEFromDevice();
EXPORT_FUNC void pushCurrentDEToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentDEFromDevice();
EXPORT_FUNC scalar* getCurrentDE(unsigned int batch = 0); 
EXPORT_FUNC void pushEStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullEStateFromDevice();
EXPORT_FUNC void allocaterewardTimestepsE(unsigned int count);
EXPORT_FUNC void freerewardTimestepsE();
EXPORT_FUNC void pushrewardTimestepsEToDevice(unsigned int count);
EXPORT_FUNC void pullrewardTimestepsEFromDevice(unsigned int count);
EXPORT_FUNC void pushstartStimECurrToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullstartStimECurrFromDevice();
EXPORT_FUNC void pushendStimECurrToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullendStimECurrFromDevice();
EXPORT_FUNC void pushECurrStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullECurrStateFromDevice();
EXPORT_FUNC void allocatestimTimesECurr(unsigned int count);
EXPORT_FUNC void freestimTimesECurr();
EXPORT_FUNC void pushstimTimesECurrToDevice(unsigned int count);
EXPORT_FUNC void pullstimTimesECurrFromDevice(unsigned int count);
EXPORT_FUNC void pushISpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullISpikesFromDevice();
EXPORT_FUNC void pushICurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullICurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getICurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getICurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushISpikeTimesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullISpikeTimesFromDevice();
EXPORT_FUNC void pushIPreviousSpikeTimesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullIPreviousSpikeTimesFromDevice();
EXPORT_FUNC void pushVIToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVIFromDevice();
EXPORT_FUNC void pushCurrentVIToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentVIFromDevice();
EXPORT_FUNC scalar* getCurrentVI(unsigned int batch = 0); 
EXPORT_FUNC void pushUIToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullUIFromDevice();
EXPORT_FUNC void pushCurrentUIToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentUIFromDevice();
EXPORT_FUNC scalar* getCurrentUI(unsigned int batch = 0); 
EXPORT_FUNC void pushIStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullIStateFromDevice();
EXPORT_FUNC void pushstartStimICurrToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullstartStimICurrFromDevice();
EXPORT_FUNC void pushendStimICurrToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullendStimICurrFromDevice();
EXPORT_FUNC void pushICurrStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullICurrStateFromDevice();
EXPORT_FUNC void allocatestimTimesICurr(unsigned int count);
EXPORT_FUNC void freestimTimesICurr();
EXPORT_FUNC void pushstimTimesICurrToDevice(unsigned int count);
EXPORT_FUNC void pullstimTimesICurrFromDevice(unsigned int count);
EXPORT_FUNC void pushEEConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullEEConnectivityFromDevice();
EXPORT_FUNC void pushEIConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullEIConnectivityFromDevice();
EXPORT_FUNC void pushIEConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullIEConnectivityFromDevice();
EXPORT_FUNC void pushIIConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullIIConnectivityFromDevice();
EXPORT_FUNC void pushtauPlusEEToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulltauPlusEEFromDevice();
EXPORT_FUNC void pushtauMinusEEToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulltauMinusEEFromDevice();
EXPORT_FUNC void pushtauCEEToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulltauCEEFromDevice();
EXPORT_FUNC void pushtauDEEToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulltauDEEFromDevice();
EXPORT_FUNC void pushaPlusEEToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullaPlusEEFromDevice();
EXPORT_FUNC void pushaMinusEEToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullaMinusEEFromDevice();
EXPORT_FUNC void pushwMinEEToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullwMinEEFromDevice();
EXPORT_FUNC void pushwMaxEEToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullwMaxEEFromDevice();
EXPORT_FUNC void pushgEEToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgEEFromDevice();
EXPORT_FUNC void pushcEEToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullcEEFromDevice();
EXPORT_FUNC void pushinSynEEToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynEEFromDevice();
EXPORT_FUNC void pushEEStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullEEStateFromDevice();
EXPORT_FUNC void pushtauPlusEIToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulltauPlusEIFromDevice();
EXPORT_FUNC void pushtauMinusEIToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulltauMinusEIFromDevice();
EXPORT_FUNC void pushtauCEIToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulltauCEIFromDevice();
EXPORT_FUNC void pushtauDEIToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulltauDEIFromDevice();
EXPORT_FUNC void pushaPlusEIToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullaPlusEIFromDevice();
EXPORT_FUNC void pushaMinusEIToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullaMinusEIFromDevice();
EXPORT_FUNC void pushwMinEIToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullwMinEIFromDevice();
EXPORT_FUNC void pushwMaxEIToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullwMaxEIFromDevice();
EXPORT_FUNC void pushgEIToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgEIFromDevice();
EXPORT_FUNC void pushcEIToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullcEIFromDevice();
EXPORT_FUNC void pushinSynEIToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynEIFromDevice();
EXPORT_FUNC void pushEIStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullEIStateFromDevice();
EXPORT_FUNC void pushinSynIEToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynIEFromDevice();
EXPORT_FUNC void pushIEStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullIEStateFromDevice();
EXPORT_FUNC void pushinSynIIToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynIIFromDevice();
EXPORT_FUNC void pushIIStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullIIStateFromDevice();
// Runner functions
EXPORT_FUNC void copyStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void copyConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void copyStateFromDevice();
EXPORT_FUNC void copyCurrentSpikesFromDevice();
EXPORT_FUNC void copyCurrentSpikeEventsFromDevice();
EXPORT_FUNC void allocateRecordingBuffers(unsigned int timesteps);
EXPORT_FUNC void pullRecordingBuffersFromDevice();
EXPORT_FUNC void allocateMem();
EXPORT_FUNC void freeMem();
EXPORT_FUNC size_t getFreeDeviceMemBytes();
EXPORT_FUNC void stepTime();

// Functions generated by backend
EXPORT_FUNC void updateNeurons(float t, unsigned int recordingTimestep); 
EXPORT_FUNC void updateSynapses(float t);
EXPORT_FUNC void initialize();
EXPORT_FUNC void initializeSparse();
}  // extern "C"
