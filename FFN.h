// FFN.h
#ifndef FFN_H
#define FFN_H

#include <string.h>
#include "tensor.h"

// 1. Define FFN (N-Layer)
typedef struct {
    int depth; // depth of FFN
    int* shape; // [768, 32,10]
    Tensor** weights;
    Tensor** biases;
} FFN;

// 2. Basic functions (Create, Save & Free)
FFN* create_ffn(int depth, int* shape);
void save_ffn(FFN* ffn, string save_path); 
void free_ffn(FFN* ffn);

// 3. Utils
int get_size(FFN* ffn);

// 4. Operation functions
void ffn_forward(FFN* ffn);
void ffn_backward(FFN* ffn);

#endif // NEURALNET_H
