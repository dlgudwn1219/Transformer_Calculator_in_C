#include <string.h>
#include "tensor.h"

// 1. Matrix Multiplication (Forward propagation): C = A @ B
void matmul_forward(Tensor* A, Tensor* B, Tensor*C){
	memset(C->value, 0, C->size * sizeof(float));

	// C[i][k] = /Sum{A[i][j] * B[j][k]}
	for (int i = 0; i < A -> rows; i++){
		for (int k = 0; k < B -> cols; k++){
			float sum = 0.0f;
			for (int j = 0; j < A->cols; j++){
				int a_idx = i * A->cols + j;
				int b_idx = j * B->cols + k;
				sum += A->value[a_idx] * B->value[b_idx];
			}
			int c_idx = i * C->cols + k;
			C->value[c_idx] = sum;
		}
	}
}

// 2. Matrix Backprop
// dL/dA = dL/dC @ B^T
// dL/dB = A^T @ dL/dC

void matmul_backward(Tensor* A, Tensor* B, Tensor*C){

	// 1. dA = dC @ B^T
	for (int i = 0; i < A->rows; i++){
        for (int k = 0; k < A->cols; k++){
            float sum = 0;
            for (int j = 0; j < C->cols; j++){
                int c_idx = i * C->cols + j;
                int b_idx = k * B->cols + j; // B^T[j][k] = B[k][j]
                sum += C->value[c_idx] * B->value[b_idx];
            }
            int a_idx = i * A->cols + k;
            A->grad[a_idx] += sum; // Grad should always be accumulated
        }
    }

    // 2. dB = A^T * dC
    for (int i = 0; i < B->rows; i++){
        for (int k = 0; k < B->cols; k++){
            float sum = 0.0f;
            for (int j = 0; j < C->rows; j++){
                int a_idx = j * A->cols + i;
                int c_idx = j * C->cols + k;
                sum += A->value[a_idx] * C->value[c_idx];
            }
            int b_idx = i * B->cols + k;
            B->grad[b_idx] += sum;
        }
    }
}

void add_forward(Tensor* A, Tensor*B, Tensor*C){
    for (int i = 0; i < C->size; i++){
        C->value[i] = A->value[i] + B->value[i];
    }
}

void add_backward(Tensor* A, Tensor* B, Tensor* C){
    for (int i = 0; i < C->size; i++){
        A->grad[i] += C->grad[i];
        B->grad[i] += C->grad[i];
    }
}

void apply_softmax(Tensor* t){
}

void apply_layernorm(Tensor* t, float* gamma, float* beta, float eps){
}


