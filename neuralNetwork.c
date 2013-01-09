#include "neuralNetwork.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

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
	unsigned int 	numMembers;		/* The number of members in the set */
	unsigned int 	numInputs;		/* The number of inputs in the set */
	unsigned int 	numOutputs;		/* The number of outputs in the set */
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
	ptrDataset = (dataset*) malloc( sizeof(dataset) );
	
	/* Set the variables */
	ptrDataset->numMembers = numMembers;  
	ptrDataset->numInputs = numInputs;
	ptrDataset->numOutputs = numOutputs;
	
	/* Allocate memory for the arrays in the dataset */
	ptrDataset->members = (dataMember*) malloc(numMembers * sizeof(dataMember) );
	ptrDataset->maxScale = (double*) malloc( (numInputs+numOutputs) * sizeof(double) );
	ptrDataset->minScale = (double*) malloc( (numInputs+numOutputs) * sizeof(double) );
	ptrDataset->sumSqErrors = (double*) malloc( numOutputs * sizeof(double) );
	ptrDataset->name = (char*) malloc( (strlen(name)+1) * sizeof(char) );
	
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
		(ptrDataset->members+i)->inputs = (double*) malloc( numInputs*sizeof(double) );
		(ptrDataset->members+i)->targets = (double*) malloc( numOutputs*sizeof(double) );
		(ptrDataset->members+i)->outputs = (double*) malloc( numOutputs*sizeof(double) );
		(ptrDataset->members+i)->errors = (double*) malloc( numOutputs*sizeof(double) );
		
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
	unsigned int i;
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
	Then, define the functions for maintaining the network,
	and setting network parameters
*/

/*
	Next, define the functions for creating the network.
	There are two scenarios:
	1.	Sarting from scratch
	2.	Loading a previous network
*/


