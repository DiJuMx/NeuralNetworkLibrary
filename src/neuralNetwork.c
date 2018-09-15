#include "neuralNetwork.h"

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
	double			delta;			/* The delta of the neuron */
	double**		inputs;			/* An array of pointers to inputs */
	double*			weights;		/* An array of weights for the inputs + bias */
	double*			deltaWeights;	/* An array of weight changes for inputs + bias */
	int				numInputs;		/* The number of inputs to the neuron */
	int				type;			/* The activation type of the neuron */ 
} neuron;

struct mlpNetwork{
	neuron**		layers;			/* Layers of neurons */
	int*			numNeurons;		/* Number of neurons in each layer */
	int 			numLayers;		/* The number of layers in the network */
	double			learnRate;		/* The learning rate of the neurons */
	double			momentum;		/* The momentum of the neurons */
	int				learning;		/* The learning type of the network */
	int				epoch;			/* The current epoch */
	int				epochMax;		/* The maximum number of epochs */
};

double sqr(double val){ return (val*val); }

double scale(double val, double min, double max, int type){
	if(min == max) return (val); /* i.e. Don't Scale this data */
	
	if(type == SCALE_FOR_NET){ /* i.e. Take data, convert to values for network */
		return ( 0.1 + (0.8 * (val - min) / (max-min) ) );
	}else if(type == SCALE_FOR_HUMAN){ /* i.e. Take network values and convert for human */
		return ( min + ((max-min) * (val - 0.1) / 0.8));
	}else if(type == SCALE_ERROR_OUT){
		return (val *(max-min) / 0.8);
	}else{
		return (val);
	}	
}

/* 
	First, define the functions for loading and handling the data.
	Since we can't do anything without some data to do stuff with
*/

dataset * loadData(char* filename, char* name){
	FILE* ptrDataFile;
	dataset* ptrDataset;
	int numInputs, numOutputs, numMembers;
	int i,j;
	double temp=0.0,min,max;
	char check = 0x00;
	
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
		perror("Couldn't allocate the dataset\n");
		return (NULL);
	}
	
	/* Set the variables */
	ptrDataset->numMembers = numMembers;  
	ptrDataset->numInputs = numInputs;
	ptrDataset->numOutputs = numOutputs;
	
	/* Allocate memory for the arrays in the dataset */
	if((ptrDataset->members = (dataMember*) malloc(numMembers * sizeof(dataMember))) != NULL)
		check |= 0x01;
	if((ptrDataset->maxScale = (double*) malloc( (numInputs+numOutputs) * sizeof(double))) != NULL)
		check |= 0x02;
	if((ptrDataset->minScale = (double*) malloc( (numInputs+numOutputs) * sizeof(double))) != NULL)
		check |= 0x04;
	if((ptrDataset->sumSqErrors = (double*) malloc( numOutputs * sizeof(double))) != NULL)
		check |= 0x08;
	if((ptrDataset->name = (char*) malloc( (strlen(name)+1) * sizeof(char))) !=NULL)
		check |= 0x10;
	
	if(check<31){
		printf("Couldn't allocate dataset\n");
		if(check & 0x01) free(ptrDataset->sumSqErrors);
		if(check & 0x02) free(ptrDataset->maxScale);
		if(check & 0x04) free(ptrDataset->minScale);
		if(check & 0x08) free(ptrDataset->members);
		if(check & 0x10) free(ptrDataset->name);
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
		check = 0x00;
		if( ((ptrDataset->members+i)->inputs = (double*) malloc( numInputs*sizeof(double) )) != NULL)
			check |= 0x01;
		if( ((ptrDataset->members+i)->targets = (double*) malloc( numOutputs*sizeof(double) )) != NULL)
			check |= 0x03;
		if( ((ptrDataset->members+i)->outputs = (double*) malloc( numOutputs*sizeof(double) )) != NULL)
			check |= 0x04;
		if( ((ptrDataset->members+i)->errors = (double*) malloc( numOutputs*sizeof(double) )) != NULL)
			check |= 0x08;
		
		if(check<15){
			printf("Couldn't allocate dataset\n");
			/* Deallocate the previous loops */
			for(j=0; j<i; j++){
				free((ptrDataset->members+j)->inputs);
				free((ptrDataset->members+j)->targets);
				free((ptrDataset->members+j)->outputs);
				free((ptrDataset->members+j)->errors);
			}
			
			if(check & 0x01) free((ptrDataset->members+i)->inputs);
			if(check & 0x02) free((ptrDataset->members+i)->targets);
			if(check & 0x04) free((ptrDataset->members+i)->outputs);
			if(check & 0x08) free((ptrDataset->members+i)->errors);
			
			free(ptrDataset->sumSqErrors);
			free(ptrDataset->maxScale);
			free(ptrDataset->minScale);
			free(ptrDataset->members);
			free(ptrDataset);
			return (NULL);
		}
		
		/* Read the inputs */
		for(j=0; j<numInputs; j++){
			max = ptrDataset->maxScale[j];
			min = ptrDataset->minScale[j];
			fscanf(ptrDataFile, "%lf, ", &temp);
			(ptrDataset->members+i)->inputs[j] = scale(temp,min,max,SCALE_FOR_NET);			
		}
		
		/* Read the outputs */
		for(j=0; j<numOutputs-1; j++){
			max = ptrDataset->maxScale[numInputs+j];
			min = ptrDataset->minScale[numInputs+j];
			fscanf(ptrDataFile, "%lf, ", &temp);
			(ptrDataset->members+i)->targets[j] = scale(temp,min,max,SCALE_FOR_NET);
			(ptrDataset->members+i)->outputs[j] = 0.0;
			(ptrDataset->members+i)->errors[j] = 0.0;
		}
		fscanf(ptrDataFile, "%lf\n", &temp);
		max = ptrDataset->maxScale[numInputs+j];
		min = ptrDataset->minScale[numInputs+j];
		(ptrDataset->members+i)->targets[j] = scale(temp,min,max,SCALE_FOR_NET);
		(ptrDataset->members+i)->outputs[j] = 0.0;
		(ptrDataset->members+i)->errors[j] = 0.0;
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

void printDataMember(dataset* data, int member){
	int i;
	for(i=0; i<data->numInputs;i++){
		printf("%lf, ", data->maxScale[i]);
	}
	printf("\n");
	for(i=0; i<data->numInputs;i++){
		printf("%lf, ", data->minScale[i]);
	}
	printf("\n");
	for(i=0; i<data->numInputs;i++){
		printf("%lf, ", (data->members+member)->inputs[i]);
	}
	printf("\n");
}

/*
	Then, define the functions for general tasks
*/

void calcSSE(dataset* set){
	int i, j;
	dataMember* datum;
	for(i=0; i<set->numOutputs; i++){
		set->sumSqErrors[i] = 0.0;
	}
	for(i=0; i< set->numMembers; i++){
		datum = set->members+i;
		for(j=0; j< set->numOutputs; j++){
			set->sumSqErrors[j] += sqr(datum->errors[j]);
		}
	}
	
	for(i=0; i<set->numOutputs; i++){
		set->sumSqErrors[i] = set->sumSqErrors[i] / (set->numMembers);
	}
	
}

double getSSE(dataset* set){
	int i;
	double total=0.0;
	calcSSE(set);
	for(i=0; i< set->numOutputs; i++) total += set->sumSqErrors[i];
	return (total);
}

void setLearnParameters(mlpNetwork* net, int emax, double learnRate, double momentum){
	if (emax > 0) net->epochMax = emax;
	else net->epochMax = 1000;
	if (learnRate >= 0.0) net->learnRate = learnRate;
	else net->learnRate = 0.5;
	if (momentum >= 0.0) net->momentum = momentum;
	else net->momentum = 0.5;
}

void getLearnParameters(mlpNetwork* net, FILE* stream){
	if(stream == NULL) stream = stdout;
	fprintf(stream, "Max Epoch: %6d, Learning Rate: %06.3lf, Momentum: %06.3lf\n", 
	                net->epochMax, net->learnRate, net->momentum);
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
		layer = net->layers[i];
		/* For each neuron in the layer */
		for(j=0; j< net->numNeurons[i]; j++){
			/* For each weight for the neuron */
			nTemp= layer+j;
			for(k=0; k<= (nTemp)->numInputs; k++){
				/* Set the weight */
				if(weights != NULL){
					nTemp->weights[k] = weights[wCnt++];
				}else{
					nTemp->weights[k] = ((double) rand() * 2.0 / RAND_MAX)-1;
				}
				/* Set the previous change to 0 */
				nTemp->deltaWeights[k] = 0;
			}				
		}
	}
}

void getWeights(mlpNetwork* net, FILE* stream){
	int i, j, k;
	neuron* layer;
	neuron* nTemp;
	
	if(stream == NULL) stream = stdout;
	
	/* For each layer */
	for(i=0; i<net->numLayers; i++){
		/* For each neuron in the layer */
		fprintf(stream, "Layer %2d:\n", i);
		layer = net->layers[i];
		for(j=0; j< net->numNeurons[i]; j++){
			/* For each weight for the neuron */
			nTemp= layer+j;
			fprintf(stream, "{");
			for(k=0; k<= (nTemp)->numInputs; k++){
				/* Print the weight */
				fprintf(stream, "%6.5lf", nTemp->weights[k]);
				if(k < (nTemp)->numInputs) fprintf(stream, ", ");
			}
			fprintf(stream, "}\n");
		}
		fprintf(stream, "\n");
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

void adaptNetwork(mlpNetwork* net, double* errors, double* inputs){
	int i, j, k;
	neuron* nTemp;
	double err=0.0;
	double input =0.0;
	
	/* First, calculate the deltas */
	
	/* For each layer from the last to the first */
	for(i=net->numLayers-1; i>=0; i--){
		/* For each neuron in the layer */
		for(j=0; j< net->numNeurons[i]; j++){
			
			/* If we're on the final layer, use the errors */
			if(i == net->numLayers-1){
				err = errors[j];
			}else{/* Otherwise use the sum of next layers (deltas * weights) */
				err = 0.0;
				/* For each neuron in the next layer */
				for(k=0; k< net->numNeurons[i+1]; k++){
					nTemp = (net->layers[i+1])+k;
					/* sum the delta * weight to neuron in this layer */
					err += nTemp->delta * nTemp->weights[j+1];
				}
			}
			
			nTemp = (net->layers[i])+j;
			
			/* Delta = error for linear */
			nTemp->delta = err;
			if(nTemp->type == SIG_ACTIVATION){ /* If sigmoidal, do some extra processing */
				nTemp->delta = nTemp->delta * (1-nTemp->output) * nTemp->output;
			}
		}
	} 
	/* Then, calculate the required deltaWeights */
	/* For each layer in the network */
	for(i=0; i< net->numLayers; i++){
		/* For each neuron in the layer */
		for(j=0; j< net->numNeurons[i]; j++){
			/*For each weight in the neuron */
			nTemp = (net->layers[i])+j;
			for(k=0; k<= nTemp->numInputs; k++){
				/* if k==0, it's the bias */
				if(k==0){
					input=1.0;;
				}else{
					/* If we're on the first layer, use the supplied inputs */
					if(i==0){
						input = inputs[k-1];
					}else{ /* Otherwise use the previous layer's outputs */
						input = *(nTemp->inputs[k-1]);
					}
				}
				
				/* calculate the deltaWeights value for this neurons weights */
				nTemp->deltaWeights[k] = net->learnRate * input * nTemp->delta
				                       + net->momentum * nTemp->deltaWeights[k];
				
				nTemp->weights[k] += nTemp->deltaWeights[k];
			}
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
			input = *(cell->inputs[i-1]);
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
		out = net->layers[net->numLayers-1]+i;
		/* Store the output in the datamember */
		datum->outputs[i] = out->output;
		/* Calculate and store the error (target - output) */
		datum->errors[i] = datum->targets[i] - datum->outputs[i];
	}
	
	free(inPtrs);
}

/*
	Function called by the user to run the network on a given dataset once
*/
void runNetworkOnce(mlpNetwork* net, dataset* data, FILE* stream, int print){
	int i,k;
	dataMember* member;
	double max, min;
	int space;
	
	if(stream == NULL) stream = stdout;
	
	
	if(print>0){
		switch(print){
			case 2:	/* Outputs and targets */
				fprintf(stream, "%-*s|%-*s\n", (9*data->numInputs)-2, "Inputs: ",
										19*data->numOutputs , "(Outputs,Targets): ");
				break;
			case 3:	/* Outputs and errors */
				fprintf(stream, "%-*s|%-*s\n", (9*data->numInputs)-2, "Inputs: ",
										19*data->numOutputs , "(Outputs,Errors):  ");
				break;
			case 4:	/* Outputs, target and errors */
				fprintf(stream, "%-*s|%-*s\n", (9*data->numInputs)-2, "Inputs: ", 
										28*data->numOutputs , "(Outputs,Targets,Errors):   ");
				break;
			case 1: /* Just the outputs */
			default:
				fprintf(stream, "%-*s|%-*s\n", (9*data->numInputs)-2, "Inputs: ", 
										10*data->numOutputs ,"(Outputs):");
				break;
		}
		printf("\n");
	}
	
	/* Then calculate the outputs for each output */
	for(i=0; i< data->numMembers; i++){
		/* Call computeNetwork on the data member */
		member = data->members+i;
		computeNetwork(net, member, data->numInputs, data->numOutputs);
		
		/* If want output, print the inputs and associated outputs */
		if(print > 0){
			for(k=0; k< data->numInputs; k++){
				max = data->maxScale[k];
				min = data->minScale[k];
				if(k==0)fprintf(stream, "%7.4lf", scale(member->inputs[k], min, max, SCALE_FOR_HUMAN));
				else	fprintf(stream, ", %7.4lf", scale(member->inputs[k], min, max, SCALE_FOR_HUMAN));
			}
			printf("|");
			for(k=0; k< data->numOutputs; k++){
				max = data->maxScale[data->numInputs+k];
				min = data->minScale[data->numInputs+k];
				switch(print){
					case 2:	/* Outputs and targets */
						fprintf(stream, "(%7.4lf, %7.4lf) ", scale(member->outputs[k],min,max,SCALE_FOR_HUMAN), scale(member->targets[k],min,max,SCALE_FOR_HUMAN) );
						break;
					case 3:	/* Outputs and errors */
						fprintf(stream, "(%7.4lf, %7.4lf) ", scale(member->outputs[k],min,max,SCALE_FOR_HUMAN), scale(member->errors[k],min,max,SCALE_ERROR_OUT));
						break;
					case 4:	/* Outputs, target and errors */
						fprintf(stream, "(%7.4lf, %7.4lf, %7.4lf) ", scale(member->outputs[k],min,max,SCALE_FOR_HUMAN), scale(member->targets[k],min,max,SCALE_FOR_HUMAN), scale(member->errors[k],min,max,SCALE_ERROR_OUT) );
						break;
					case 1: /* Just the outputs */
					default:
						fprintf(stream, "(%7.4lf) ", scale(member->outputs[k],min,max,SCALE_FOR_HUMAN) );
						break;				
				}
			}
			fprintf(stream, "\n");
		}		
		
	}
	
}

/*
	Function called by the user to train the network once
*/
void trainNetworkOnce(mlpNetwork* net, dataset* data, int print){
	int i;
	/* Consider mode (batch / online) */
	
	/* For each datamemember, compute then adapt */
	for(i=0; i< data->numMembers; i++){
		computeNetwork(net, data->members+i, data->numInputs, data->numOutputs);
		adaptNetwork(net, (data->members+i)->errors, (data->members+i)->inputs);
		if(print == 1) printf("Total SSE: %9.5lf\n", getSSE(data) );
	}
	
}

void trainNetwork(mlpNetwork* net, dataset* training, dataset* validation, FILE* stream,  int print){
	double currAvg = 0.0, prevAvg = 10000000.0;
	
	if(stream == NULL) stream = stdout;
	
	for(net->epoch=0; net->epoch < net->epochMax; net->epoch++){
		
		trainNetworkOnce(net, training, 0);
		
		/* If we're using the validation set */
		if(validation != NULL){
			runNetworkOnce(net, validation, stream, 0);
			currAvg += getSSE(validation);
			
			/* If epoch at least 150 */
			if( net->epoch > 149 ){
				/* Every 10th epoch */
				if( net->epoch % 10 == 0){
					/* If the average has risen */
					if(currAvg > prevAvg){
						break;					
					}
					prevAvg = currAvg;
					currAvg = 0.0;
				}
			}
		}
	}
	
	if(print == 1) fprintf(stream, "Epochs: %6d, Training SSE: %9.5lf", net->epoch, getSSE(training));
	if(validation != NULL) fprintf(stream, ", Valid SSE: %9.5lf", getSSE(validation));
	fprintf(stream, "\n");
}

/*
	Next, define the functions for creating the network.
	There are two scenarios:
	1.	Starting from scratch
	2.	Loading a previous network (For now ignore this)
*/

void destroyNet(mlpNetwork* net){
	int i, j;
	neuron* nTemp;
	
	/* First deallocate the neurons int he layers */
	for(i=0; i< net->numLayers; i++){
		/* For each neuron */
		for(j=0; j< net->numNeurons[i]; j++){
			nTemp = (net->layers[i])+j;
			free(nTemp->weights);
			free(nTemp->inputs);
			free(nTemp->deltaWeights);
		}
		/* Followed by the layers*/
		free(net->layers[i]);
	}
	/* Finally, The array of layers, the number of 
	neurons per layer, and the net itself */
	free(net->layers);
	free(net->numNeurons);
	free(net);
}

mlpNetwork* createNetwork(int numLayers, int* numPerLayer, int inputs, int learnMethod, int defaultActivation){
	int i,j,k,m;
	char check =0x00;
	mlpNetwork* net;
	neuron* nTemp;
	 double** inPtrs;
	 
	/* Check Validity */
	if(numLayers < 1 || inputs < 1){
		printf("\n Must specify 1 or more Layers and/or inputs \n");
		return (NULL);
	}
	if(learnMethod != BPROP_LEARNING){
		if(learnMethod == HEBB_LEARNING) printf("The hebbian learning method is not implemented yet\n");
		else printf("The learning method is not recognised\n");
		return (NULL);
	}
	if(   defaultActivation != LIN_ACTIVATION
	   && defaultActivation != SIG_ACTIVATION){
		printf("Activation method not recognised\n");
		return (NULL);
	}
		
	if((net = (mlpNetwork*) malloc(sizeof(mlpNetwork))) == NULL) return (NULL);
	net->numLayers = numLayers;
	net->learning = learnMethod;
	
	/* Set a default in case they don't get set */
	net->learnRate = 0.5;
	net->momentum = 0.5;
	
	if((net->layers = (neuron**) malloc(numLayers * sizeof(neuron*))) != NULL) check |= 0x01;
	
	if((net->numNeurons = (int*) malloc(numLayers * sizeof(int))) != NULL) check |= 0x02;
	
	if(check < 3){
		printf("Couldn't create network\n");
		if(check & 0x01) free(net->layers);
		if(check & 0x02) free(net->numNeurons);
		free(net);
		return (NULL);
	}
	
	/* Create neurons */
	
	/* First, create the layers */
	for(i=0; i<numLayers; i++){
		/* Store the number of neurons in the layer */
		net->numNeurons[i] = numPerLayer[i];
		/* Allocate the neurons for the layer */
		if((net->layers[i] = (neuron*) malloc(numPerLayer[i] * sizeof(neuron))) == NULL){
			/* If couldn't allocate */
			printf("Couldn't create network\n");
			/* Deallocate previously allocated layers */
			for(j=0; j<i; j++){
				free(net->layers[j]);
			}
			/* Deallocate other arrays */
			free(net->layers);
			free(net->numNeurons);
			free(net);
			return (NULL);
		}
		
		/* For each neuron in the layer */
		
		for(j=0; j< numPerLayer[i]; j++){
			/* Assign a temporary pointer */
			nTemp = (net->layers[i])+j;
			
			/* Set the number of inputs */
			if(i==0) nTemp->numInputs = inputs; /* If it's the input layer */
			else nTemp->numInputs = numPerLayer[i-1]; /* If it's not the input layer */
			
			/* Set the activation type */
			nTemp->type = defaultActivation;
			
			/* Allocate the memory for the weights and inputs */
			check=0x00;
			if(( nTemp->weights = (double*) malloc((nTemp->numInputs +1) * sizeof(double))) != NULL) check |= 0x01;
			if(( nTemp->inputs = (double**) malloc((nTemp->numInputs) * sizeof(double*))) != NULL) check |= 0x02;
			
			/* Allocate memory for the weight change array */
			if(( nTemp->deltaWeights = (double*) malloc((nTemp->numInputs+1)*sizeof(double))) != NULL) check |= 0x04;
			
			/* If they failed */
			if(check<0x07){
				printf("Couldn't create network\n");
				/* Need to deallocate succeded ones*/
				if(check & 0x01) free(nTemp->weights);
				if(check & 0x02) free(nTemp->inputs);
				if(check & 0x04) free(nTemp->deltaWeights);
				/* First, all previous neurons/weights in this layer */
				for(k=0; k<j; k++){
					nTemp = (net->layers[i])+k;
					free(nTemp->weights);
					free(nTemp->inputs);
				}
				/* Then, this layer */
				free(net->layers[i]);
				/* Then, all the neurons in the previous layers */
				for(m=0; m<i; m++){
					/* For each neuron */
					for(k=0; k<numPerLayer[m]; k++){
						nTemp = (net->layers[m])+k;
						free(nTemp->weights);
						free(nTemp->inputs);
						free(nTemp->deltaWeights);
					}
					/* Followed by those layers*/
					free(net->layers[m]);
				}
				/* Finally, The array of layers, the number of 
				neurons per layer, and the net itself */
				free(net->layers);
				free(net->numNeurons);
				free(net);
				return (NULL);
			}
			
			
		}
		
		
	}
	
	/*connect layers */	
	for(i=1; i<numLayers; i++){
		
		if((inPtrs = (double**) malloc((net->numNeurons[i-1]) * sizeof(double))) == NULL){
			printf("Couldn't create the network\n");
			destroyNet(net);
			return (NULL);
		}
		for(j=0; j< net->numNeurons[i-1]; j++){
			nTemp = (net->layers[i-1])+j;
			inPtrs[j] = &(nTemp->output);
		}
		connectInputs(net, inPtrs, i);
		free(inPtrs);
	}
	
	setWeights(net, NULL);
	return (net);
}

