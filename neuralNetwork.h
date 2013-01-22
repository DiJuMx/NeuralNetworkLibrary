#ifndef	NEURAL_NETWORK_H
#define	NEURAL_NETWORK_H

#define LIN_ACTIVATION 	0x0001	/* Linear activation, i.e. output = input */
#define SIG_ACTIVATION 	0x0002	/* Sigmoidal activation, i.e. output = 1/(1+e^(-input)) */
#define STP_ACTIVATION 	0x0003	/* Step (not available yet) */

#define BPROP_LEARNING 	0x0011 	/* Back Propagation */
#define HEBB_LEARNING  	0x0012 	/* Hebbian learning (not available yet) */

#define SCALE_FOR_NET  	0x0101	/* Scale human data for use in the network */
#define SCALE_FOR_HUMAN	0x0102	/* Scale network data for presentation to human */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

typedef struct dataset dataset;
typedef struct mlpNetwork mlpNetwork;

dataset * loadData(char* filename, char* name);
void destroyDataset(dataset* ptrDataset);

void setLearnParameters(mlpNetwork* Net, int emax, double learnRate, double momentum);
void getLearnParameters(mlpNetwork* net, FILE* stream);

void setWeights(mlpNetwork* net, double* weights);
void runNetworkOnce(mlpNetwork* net, dataset* data, int print);
void trainNetworkOnce(mlpNetwork* net, dataset* data, int print);

void trainNetwork(mlpNetwork* net, dataset* training, dataset* validation, int print);


void destroyNet(mlpNetwork* net);
mlpNetwork* createNetwork(int numLayers, int* numPerLayer, int inputs, int learnMethod, int defaultActivation);

#endif	/* NEURAL_NETWORK_H */	
