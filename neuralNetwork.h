#ifndef	NEURAL_NETWORK_H
#define	NEURAL_NETWORK_H

typedef struct dataset dataset;

dataset * loadData(char* filename, char* name);
void destroyDataset(dataset* ptrDataset);

#endif	/* NEURAL_NETWORK_H */	
