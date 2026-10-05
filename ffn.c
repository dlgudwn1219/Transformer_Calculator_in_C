#include <stdio.h>
#include <string.h>
#include "tensor.h"
#include "FFN.h"

// Create ffn
FFN* create_ffn(int depth, int* shape){
    FFN* ffn = (FFN*)malloc(sizeof(FFN));
    
    ffn->depth = depth; // # of Weight matrix
    
    int num_layers = depth + 1;
    
    ffn->shape = (int*)malloc(num_layers * sizeof(int));
    memcpy(ffn->shape, shape, num_layers * sizeof(int));

    ffn->weights = (Tensor**)malloc(depth * sizeof(Tensor*));
    ffn->biases = (Tensor**)malloc(depth * sizeof(Tensor*));

    for (int i = 0; i < depth; i++){
        create_tensor(ffn->weights[i]);
        create_tensor(ffn->biases[i]);
    }
}

void save_ffn(FFN* ffn, string save_path){
}

void free_ffn(FFN* ffn){
    if (ffn){
        // 1. Free Wegihts & Biases first
        for (int i = 0; i < depth; i++){
            free_tensor(ffn->weights[i]);
            free_tensor(ffn->biases[i]);
        }
        free(ffn->shape);
        free(f);
    }
}

long get_size(FFN* ffn){
    long size = 0;
    for (int i = 0; i < depth; i++){
        size += ffn->shape[i] * ffn->shape[i+1] + ffn->shape[i+1];
    }
    return size;
}

void ffn_forward(FFN* ffn, Tensor* A){
    // OHHHH!! I got to know that the 'Z's eat so much memory
    for (int i = 
}

void ffn_backward(FFN* ffn);
