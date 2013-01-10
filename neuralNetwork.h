#ifndef	NEURAL_NETWORK_H
#define	NEURAL_NETWORK_H

#define LIN_ACTIVATION 0x0001	/* Linear activation, i.e. output = input */
#define SIG_ACTIVATION 0x0002	/* Sigmoidal activation, i.e. output = 1/(1+e^(-input)) */
#define STP_ACTIVATION 0x0003	/* Step (not available yet) */

#define BPROP_LEARNING 0x0011 	/* Back Propagation */
#define HEBB_LEARNING  0x0012 	/* Hebbian learning (not available yet) */

unsigned char 

typedef struct dataset dataset;
typedef struct mlpNetwork mlpNetwork;

dataset * loadData(char* filename, char* name);
void destroyDataset(dataset* ptrDataset);

void setLearnParameters(mlpNetwork* Net, double learnRate, double momentum);
void setWeights(mlpNetwork* net, double* weights);
void runNetwork(mlpNetwork* net, dataset* data, int print);

#endif	/* NEURAL_NETWORK_H */	
