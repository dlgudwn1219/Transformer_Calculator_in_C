// tensor.h
#ifndef TENSOR_H
#define TENSOR_H

// 1. Define Tensor
typedef struct {
    int ndim;
    int* shape;
	int size;
	float* value;
	float* grad;

    int rows;
    int cols;
} Tensor;

// 2. Basic functions (Create & Free)
Tensor* create_tensor(int ndim, int* shape);
void free_tensor(Tensor* t);

// 3. Utils
int get_batch_size(Tensor* A); // return batch_size

// 3. Operation functions
void matmul_forward(Tensor* A, Tensor* B, Tensor* C);
void matmul_backward(Tensor* A, Tensor* B, Tensor* C);

void add_forward(Tensor* A, Tensor* B, Tensor* C);
void add_backward(Tensor* A, Tensor* B, Tensor* C);


#endif // TENSOR_H
