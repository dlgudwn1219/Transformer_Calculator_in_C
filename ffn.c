#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include "tensor.h"
#include "FFN.h"

// Create ffn
FFN* create_ffn(int depth, int* shape){
    FFN* ffn = (FFN*)malloc(sizeof(FFN));
    
    ffn->depth = depth; // # of Weight matrix
        
    ffn->shape = (int*)malloc((depth + 1) * sizeof(int));
    memcpy(ffn->shape, shape, (depth + 1) * sizeof(int));

    ffn->weights = (Tensor**)malloc(depth * sizeof(Tensor*));
    ffn->biases = (Tensor**)malloc(depth * sizeof(Tensor*));

    for (int i = 0; i < depth; i++){
        int shape_i[2] = {ffn->shape[i], ffn->shape[i+1]};
        ffn->weights[i] = create_tensor(2, shape_i);
        ffn->biases[i] = create_tensor(2, shape_i);
    }

    ffn->results= NULL;

    return ffn;
}

void save_ffn(FFN* ffn, char* save_path){
}

void free_ffn(FFN* ffn){
    if (ffn){
        // 1. Free Wegihts & Biases
        for (int i = 0; i < ffn->depth; i++){
            free_tensor(ffn->weights[i]);
            free_tensor(ffn->biases[i]);
        }

        free(ffn->shape);

        // 2. Free whole computation graph
        if (ffn->results){
            for (int i = 0; i < ffn->depth; i++) free_tensor(ffn->results[i]);
            free(ffn->results);
        }

        free(ffn);
    }
}

long get_size(FFN* ffn){
    long size = 0;
    for (int i = 0; i < ffn->depth; i++){
        size += ffn->shape[i] * ffn->shape[i+1] + ffn->shape[i+1];
    }
    return size;
}

Tensor* ffn_forward(FFN* ffn, Tensor* A){
    // OHHHH!! I got to know that the 'Z's eat so much memory
    // 1. Allocate results(computation graph) memory
    if (ffn->results == NULL){
        printf("Allocating memory for computation graph...\n");
        ffn->results = (Tensor**)malloc(ffn->depth * sizeof(Tensor*));
            
        for (int i = 0; i < ffn->depth; i++){
            // Allocat
            int* shape_i = (int*)malloc(A->ndim * sizeof(int));
            for (int j = 0; j < A->ndim - 1; j++) shape_i[j] = A->shape[j];
            shape_i[A->ndim-1] = ffn->shape[i+1];

            ffn->results[i] = create_tensor(A->ndim, shape_i);
            free(shape_i);
        }
        printf("Allocated memory for compuation graph!!\n");
    }
    
    // 2. Do the forwarding process
    matmul_forward(A, ffn->weights[0], ffn->results[0]);
    add_forward(ffn->results[0], ffn->biases[0], ffn->results[0]);

    printf("First layer computed..\n");
    for (int i = 0; i < ffn->depth-1; i++){
        printf("layer%d computing..\n", i);
        matmul_forward(ffn->results[i], ffn->weights[i], ffn->results[i+1]);
    }

    return ffn->results[ffn->depth-1];
}

void ffn_backward(FFN* ffn){
    if (ffn->results == NULL){
        printf("ERROR: forward first to do backward\n");
        return;
    }
    
    // 1. Y and Y^
    

    // 2. Backprop over W and Zs..

    for (int i = ffn->depth -1; i > 0; i--)
        matmul_backward(ffn->results[i], ffn->results[i-1], ffn->weights[i-1]);

    return;
}
