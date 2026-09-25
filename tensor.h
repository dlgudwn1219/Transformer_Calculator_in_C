// tensor.h
#ifndef TENSOR_H
#define TENSOR_H

// 1. Define Tensor
typedef struct {
	int rows;
	int cols;
	int size; // rows * cols
	float* value;
	float* grad;
} Tensor;

// 2. Basic functions (Create & Free)
Tensor* create_tensor(int rows, int cols);
void free_tensor(Tensor* t);

// 3. Operation functions
void matmul_forward(Tensor* A, Tensor* B, Tensor* C);
void matmul_backward(Tensor* A, Tensor* B, Tensor* C);

void add_forward(Tensor* A, Tensor* B, Tensor* C);
void add_backward(Tensor* A, Tensor* B, Tensor* C);


#endif // TENSOR_H
