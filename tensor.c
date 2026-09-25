#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tensor.h"

// 1. Create Tensor
Tensor* create_tensor(int rows, int cols){
	Tensor* t = (Tensor*)malloc(sizeof(Tensor));
	t->rows = rows;
	t->cols = cols;
	t->size = rows * cols;

	t->value = (float*)calloc(t->size, sizeof(float));
	t->grad = (float*)calloc(t->size, sizeof(float));

	return t;
}

// 2. Freeing Tensor
void free_tensor(Tensor* t) {
	if (t) {
		free(t->value);
		free(t->grad);
		free(t);
	}
}
