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

/** @brief A container to hold the arrays associated with a nnBundle's output
 */
struct nnBundleOutput {
        int     number;
        float * values;
        float * gradients;
};

/** @brief A Bundle is the equivalent of a single layer of a specific neuron type.
 */
struct nnBundle {
        /** A container holding the bundle output */
        struct nnBundleOutput   output;

        /** An array of pointers to the outputs of previous bundles */
        struct nnBundleOutput** inputs;

        /** Number of elements in the inputs array */
        int                     numInputBundles;

        /** Non-common data */
        void *                  data;

        /** Forward pass function */
        enum nnStatusCode (*forward)(struct nnBundle * bundle);

        /** Backward pass function */
        enum nnStatusCode (*backward)(struct nnBundle * bundle);
};

/** @brief The nnNetwork structure groups together all of the nnLayers within the model.
 */
struct nnNetwork {
        /** An array of the layers within the entire network */
        struct nnBundle * layers;
};


#ifdef __cplusplus
}
#endif

#endif /* NN_CORE_H */
