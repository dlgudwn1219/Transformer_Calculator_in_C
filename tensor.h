// tensor.h
#ifndef TENSOR_H
#define TENSOR_H

// 1. Define Tensor
typedef struct {
    int ndim;
    int* shape;
	int size;
	float* values;
	float* grads;

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

void scale_tensor(Tensor* A, float scale);
void transpose(Tensor* in, Tensor* out);

void relu_forward(Tensor* X, Tensor* Y);
void relu_backward(float* x, float* dout, float* dx, int size);

void sgd_update(Tensor* t, float lr);

float softmax_crossentropy_forward(float* logits, int target_class, float* probs, int num_classes);
void apply_softmax(Tensor* Y_hat);

void softmax_crossentropy_backward(float* probs, int target_class, float* dlogits, int num_classes);

#endif // TENSOR_H
