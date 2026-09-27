#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tensor.h"

// 1. Create Tensor
Tensor* create_tensor(int ndim, int* shape){
	Tensor* t = (Tensor*)malloc(sizeof(Tensor));
	t->ndim = ndim;

    t->shape = (int*)malloc(ndim * sizeof(int));
    memcpy(t->shape, shape, ndim * sizeof(int));
    
    int size = 1;
    for (int i = 0; i < ndim; i++) size *= shape[i];

    t->size = size;

	t->value = (float*)calloc(t->size, sizeof(float));
	t->grad = (float*)calloc(t->size, sizeof(float));

    t->rows = t->shape[t->ndim - 2];
    t->cols = t->shape[t->ndim - 1];
	return t;
}

// 2. Freeing Tensor
void free_tensor(Tensor* t){
    if (t){
        free(t->shape);
		free(t->value);
		free(t->grad);
		free(t);
	}
}

// 3. Utils
int get_batch_size(Tensor* t) {
    int batch_size = 1;
    for (int i = 0; i < t->ndim - 2; i++) batch_size *= t->shape[i];

    return batch_size;
}
