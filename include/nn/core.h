/** @file nn/core.h
 *  @brief This file provides the generic interfaces and structures and
 *  forms the core of the neural network library.
 */

#ifndef NN_CORE_H
#define NN_CORE_H


#ifdef __cplusplus
extern "C" {
#endif

/** @brief Status codes for the library
 */
enum nnStatusCodes {
        NN_SUCCESS = 0x0,       /**< @brief Generic success statuscode */
};

/** @brief The nnUnit structure represents a single neuron, or unit.
 */
struct nnUnit {
        /** The units inputs variable should point to the array of input pointers in the layer */
        double ** inputs;
        /** The units output variable should point to the appropriate element in the layer output array */
        double *  output;
        /** The units gInputs variable should point to the array of gradient input pointers in the layer */
        double ** gInputs;
        /** The unit gOutput should point to the appropriate element in the layer gradient output array */
        double *  gOutput; 

        /** Function pointer for the forward pass or activation function */
        int (*fPtrForward)(struct nnUnit * unit);
        /** Function pointer for the backward pass or gradient calculation function */
        int (*fPtrBackward)(struct nnUnit * unit);

        /** This pointer contains any additional functionality required for correct operation of the unit */
        void * extra;
};


/** @brief The nnLayer structure acts as a wrapper around a collection of nnUnits.
 */
struct nnLayer {
        /** The array of neural units within the layer */
        struct nnUnit* units;
        /** An array of pointers which reference the outputs in the previous layers  */
        double ** inputs;
        /** An array of values the units in the layer should populate with activations */
        double *  output;
        /** An array of pointers which reference the gradients of outputs in the previous layers */
        double ** gInputs;
        /** An array of values accumulating the gradients from the subsequent layers */
        double *  gOutput;

        /** The number of elements in the input and gInput arrays */
        int numInputs;
        /** The number of elements in the input and gInput arrays */
        int numOutputs;
        /** The number of units within the layer */
        int numUnits;
};

/** @brief The nnNetwork structure groups together all of the nnLayers within the model
 */
struct nnNetwork {
        /** Array of layers in the network. */
        struct nnLayer * layers;
};


#ifdef __cplusplus
}
#endif

#endif /* NN_CORE_H */
