#include "neuralNetwork.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

typedef struct dataMember{
	double*			inputs;			/* The input data */
	double*			targets;		/* The target outputs */
	double*			outputs;		/* The actual outputs */
	double*			errors;			/* The error in the output */
} dataMember;

struct dataset{
	dataMember*		members;		/* The members of the dataset */
	double*			maxScale;		/* What the max value of the ins/outs are */
	double*			minScale;		/* What the min value of the ins/outs are */
	double*			sumSqErrors;	/* The sum squared errors of all members */
	char* 			name;			/* The name of the data set */
	int 			numMembers;		/* The number of members in the set */
	int 			numInputs;		/* The number of inputs in the set */
	int 			numOutputs;		/* The number of outputs in the set */
};

typedef struct neuron{
	double			output;			/* The output of the neuron */
	double**		inputs;			/* An array of pointers to inputs */
	double*			weights;		/* An array of weights for the inputs + bias */
	int				numInputs;		/* The number of inputs to the neuron */
	short			type;			/* The activation type of the neuron */ 
} neuron;

struct mlpNetwork{
	neuron**		layers;			/* Layers of neurons */
	int*			numNeurons;		/* Number of neurons in each layer */
	int 			numLayers;		/* The number of layers in the network */
	double			learnRate;		/* The learning rate of the neurons */
	double			momentum;		/* The momentum of the neurons */
	short			learning;		/* The learning type of the network */
	int				epoch;			/* The current epoch */
	int				epochMax;		/* The maximum number of epochs */
};

/* 
	First, define the functions for loading and handling the data.
	Since we can't do anything without some data to do stuff with
*/

dataset * loadData(char* filename, char* name){
	FILE* ptrDataFile;
	dataset* ptrDataset;
	int numInputs, numOutputs, numMembers;
	int i,j;
	
	/* Open filename */
	if( (ptrDataFile = fopen(filename, "r"))==NULL){
		perror(NULL);
		return (NULL);
	}
		
	/* Load first line which contain settings */
	fscanf(ptrDataFile, "%d, %d, %d\n", &numMembers, &numInputs, &numOutputs);
		
	/* Setup the dataset */
	/* Allocate memory for the dataset */
	if( (ptrDataset = (dataset*) malloc( sizeof(dataset) ))==NULL){
		perror("Couldn't allocate the dataset");
		return (NULL);
	}
	
	/* Set the variables */
	ptrDataset->numMembers = numMembers;  
	ptrDataset->numInputs = numInputs;
	ptrDataset->numOutputs = numOutputs;
	
	/* Allocate memory for the arrays in the dataset */
	if( (ptrDataset->members = (dataMember*) malloc(numMembers * sizeof(dataMember) )) == NULL){
		perror("Couldn't allocate dataset");
		free(ptrDataset);
		return (NULL);
	}
	if( (ptrDataset->maxScale = (double*) malloc( (numInputs+numOutputs) * sizeof(double) )) == NULL){
		perror("Couldn't allocate dataset");
		free(ptrDataset->members);
		free(ptrDataset);
		return (NULL);
	}
	if( (ptrDataset->minScale = (double*) malloc( (numInputs+numOutputs) * sizeof(double) )) == NULL){
		perror("Couldn't allocate dataset");
		free(ptrDataset->maxScale);
		free(ptrDataset->members);
		free(ptrDataset);
		return (NULL);
	}
	if( (ptrDataset->sumSqErrors = (double*) malloc( numOutputs * sizeof(double) )) == NULL){
		perror("Couldn't allocate dataset");
		free(ptrDataset->maxScale);
		free(ptrDataset->minScale);
		free(ptrDataset->members);
		free(ptrDataset);
		return (NULL);
	}
	if( (ptrDataset->name = (char*) malloc( (strlen(name)+1) * sizeof(char) ))==NULL){
		perror("Couldn't allocate dataset");
		free(ptrDataset->sumSqErrors);
		free(ptrDataset->maxScale);
		free(ptrDataset->minScale);
		free(ptrDataset->members);
		free(ptrDataset);
		return (NULL);
	}
	
	/* load the rest of the data */
	/* Get the max and mins */
	for(i=0; i< numInputs+numOutputs-1; i++){
		fscanf(ptrDataFile, "%lf, ", ptrDataset->maxScale +i);
	}	
	fscanf(ptrDataFile, "%lf\n", ptrDataset->maxScale+numInputs+numOutputs-1);
	for(i=0; i< numInputs+numOutputs-1; i++){
		fscanf(ptrDataFile, "%lf, ", ptrDataset->minScale +i);
	}	
	fscanf(ptrDataFile, "%lf\n", ptrDataset->minScale+numInputs+numOutputs-1);
	
	/* Get the data */
	/* For each Member */
	for(i=0; i< numMembers; i++){
		/* Allocate the memory for the member */
		if( ((ptrDataset->members+i)->inputs = (double*) malloc( numInputs*sizeof(double) )) == NULL){
			perror("Couldn't allocate dataset");
			for(j=0; j<i; j++){
				free((ptrDataset->members+j)->inputs);
				free((ptrDataset->members+j)->targets);
				free((ptrDataset->members+j)->outputs);
				free((ptrDataset->members+j)->errors);
			}
			free(ptrDataset->sumSqErrors);
			free(ptrDataset->maxScale);
			free(ptrDataset->minScale);
			free(ptrDataset->members);
			free(ptrDataset);
			return (NULL);
		}
		if( ((ptrDataset->members+i)->targets = (double*) malloc( numOutputs*sizeof(double) )) == NULL){
			perror("Couldn't allocate dataset");
			for(j=0; j<i; j++){
				free((ptrDataset->members+j)->inputs);
				free((ptrDataset->members+j)->targets);
				free((ptrDataset->members+j)->outputs);
				free((ptrDataset->members+j)->errors);
			}
			free((ptrDataset->members+j)->inputs);
			free(ptrDataset->sumSqErrors);
			free(ptrDataset->maxScale);
			free(ptrDataset->minScale);
			free(ptrDataset->members);
			free(ptrDataset);
			return (NULL);
		}
		if( ((ptrDataset->members+i)->outputs = (double*) malloc( numOutputs*sizeof(double) )) == NULL){
			perror("Couldn't allocate dataset");
			for(j=0; j<i; j++){
				free((ptrDataset->members+j)->inputs);
				free((ptrDataset->members+j)->targets);
				free((ptrDataset->members+j)->outputs);
				free((ptrDataset->members+j)->errors);
			}
			free((ptrDataset->members+i)->inputs);
			free((ptrDataset->members+i)->targets);
			free(ptrDataset->sumSqErrors);
			free(ptrDataset->maxScale);
			free(ptrDataset->minScale);
			free(ptrDataset->members);
			free(ptrDataset);
			return (NULL);
		}
		if( ((ptrDataset->members+i)->errors = (double*) malloc( numOutputs*sizeof(double) )) == NULL){
			perror("Couldn't allocate dataset");
			for(j=0; j<i; j++){
				free((ptrDataset->members+j)->inputs);
				free((ptrDataset->members+j)->targets);
				free((ptrDataset->members+j)->outputs);
				free((ptrDataset->members+j)->errors);
			}
			free((ptrDataset->members+i)->inputs);
			free((ptrDataset->members+i)->targets);
			free((ptrDataset->members+i)->outputs);
			free(ptrDataset->sumSqErrors);
			free(ptrDataset->maxScale);
			free(ptrDataset->minScale);
			free(ptrDataset->members);
			free(ptrDataset);
			return (NULL);
		}
		
		/* Read the inputs */
		for(j=0; j<numInputs; j++){
			fscanf(ptrDataFile, "%lf, ", (ptrDataset->members+i)->inputs+i);
		}
		
		/* Read the outputs */
		for(j=0; j<numOutputs-1; j++){
			fscanf(ptrDataFile, "%lf, ", (ptrDataset->members+i)->targets+i);
			*((ptrDataset->members+i)->outputs+i) = 0.0;
			*((ptrDataset->members+i)->errors+i) = 0.0;
		}
		fscanf(ptrDataFile, "%lf\n", (ptrDataset->members+i)->targets+numOutputs-1);
	}
	
	/* Make sure the file is closed */
	fclose(ptrDataFile);
	
	/* Finally, return the pointer to the dataset */
	return (ptrDataset);
} 

void destroyDataset(dataset* ptrDataset){
	int i;
	/* First loop through the members and free the arrays in them */
	for(i=0; i< (ptrDataset->numMembers); i++){
		free( (ptrDataset->members +i)->inputs	);
		free( (ptrDataset->members +i)->outputs	);
		free( (ptrDataset->members +i)->targets	);
		free( (ptrDataset->members +i)->errors	);
	}
	
	/* Then free the arrays in the data set */
	free( ptrDataset->members 	  );
	free( ptrDataset->maxScale 	  );
	free( ptrDataset->minScale	  );
	free( ptrDataset->sumSqErrors );
	free( ptrDataset->name 		  );
	
	/* Then free the dataset itself */	
	free(ptrDataset);
}
/*
	Then, define the functions for general tasks
*/
void setLearnParameters(mlpNetwork* net, int emax, double learnRate, double momentum){
	if (emax >= 0) net->epochMax = emax;
	if (learnRate >= 0.0) net->learnRate = learnRate;
	if (momentum >= 0.0) net->momentum = momentum;
}

/*
	This function is used to set the weights for each neuron
*/
void setWeights(mlpNetwork* net, double* weights){
	int i, j, k;
	int wCnt=0;
	neuron* layer;
	neuron* nTemp;
	
	/* For each layer */
	for(i=0; i<net->numLayers; i++){
		/* For each neuron in the layer */
		for(j=0; j< net->numNeurons[i]; j++){
			/* For each weight for the neuron */
			layer = net->layers[i];
			for(k=0; k<= (layer+j)->numInputs; k++){
				nTemp= layer+j;
				/* Update the weight */
				nTemp->weights[k] = weights[wCnt++];
			}				
		}
	}
}

/*
	Helper function which runs the network on a single data member
*/

void connectInputs(mlpNetwork* net, double** inPtrs, int layer){
	int i,j;
	neuron* nTemp;
	
	/* For each neuron in the the layer*/
	for(i=0; i< net->numNeurons[layer]; i++){
		/* For each input */
		nTemp = net->layers[layer]+i;
		for(j=0; j< nTemp->numInputs; j++){
			/* Set the pointer to the input variable */
			nTemp->inputs[j] = inPtrs[j];	
		}
	}
}

void computeNeuron(neuron* cell){
	int i;
	double input=0.0;
	
	/* Reset cell output to 0 */
	cell->output=0.0;
	/* Sum the inputs */
	for(i=0; i<= cell->numInputs; i++){
		if(i==0){ /* The bias */
			input = 1.0;
		}else{
			input = *(cell->inputs[i]);
		}
		cell->output += input * cell->weights[i];
	}
	
	if(cell->type == LIN_ACTIVATION){
		/* Do Nothing (output = sum of inputs) */
	}else if(cell->type == SIG_ACTIVATION){
		/* Apply sigmoid function */
		cell->output = 1.0 / (1.0 + exp(-cell->output)); 
	}else{
		/* Set ouput to 0.0*/
		cell->output = 0.0;
	}
}


void computeNetwork(mlpNetwork* net, dataMember* datum, int numIn, int numOut){
	int i,j;
	neuron* out;
	double** inPtrs;
	
	/* Store the addresses of the inputs in an array */
	inPtrs = (double**) malloc(numIn * sizeof(double*));
	for(i=0; i< numIn; i++){
		inPtrs[i] = datum->inputs+i;
	}
	/* Then map the inputs for the first layer to them */
	connectInputs(net, inPtrs, 0);
	
	/* For each layer */
	for(i=0; i< net->numLayers; i++){
		/* For each neuron in that layer */
		for(j=0; j< net->numNeurons[i]; j++){
			/* Compute the neuron output */
			computeNeuron(*(net->layers+i)+j);
		}
	}
	
	/* For each output */
	for(i=0; i<numOut; i++){
		out = *(net->layers+ net->numLayers -1)+i;
		/* Store the output in the datamember */
		datum->outputs[i] = out->output;
		/* Calculate and store the error (target - output) */
		datum->errors[i] = datum->targets[i] - datum->outputs[i];
	}
}

/*
	Function called by the user to run the network on a given dataset
*/

void runNetwork(mlpNetwork* net, dataset* data, int print){
	int i, j;
	
	for(i=0; i< data->numMembers; i++){
		/* Call computeNetwork on the data member */
	}
	
	/* Calculate the sumSqError for each output */
}


/*
	Next, define the functions for creating the network.
	There are two scenarios:
	1.	Sarting from scratch
	2.	Loading a previous network (For now ignore this)
*/


