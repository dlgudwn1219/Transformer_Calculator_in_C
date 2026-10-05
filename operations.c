#include <string.h>
#include <math.h>
#include <float.h>
#include <stddef.h> // for NULL
#include "tensor.h"



// 1. Matrix Multiplication (Forward, Backward): C = A @ B

// Forward
void matmul_forward(Tensor* A, Tensor* B, Tensor*C){
	memset(C->value, 0, C->size * sizeof(float));

    int I = A->shape[A->ndim - 2];
    int J = A->shape[A->ndim - 1];
    int K = B->shape[B->ndim - 1];

    // All batch size except the last two dimensions
    int batch_size = get_batch_size(A);

	// C[I][K] = /Sum{A[I][J] * B[J][K]}
    for (int b = 0; b < batch_size; b++){
	    for (int i = 0; i < I; i++){
	    	for (int k = 0; k < K; k++){
	    		float sum = 0.0f;
    			for (int j = 0; j < J; j++){
			    	int a_idx = b * I*J + i * A->cols + j;
		    		int b_idx = b * J*K + j * B->cols + k;
	    			sum += A->value[a_idx] * B->value[b_idx];
    			}
			    int c_idx = b* I*K + i * C->cols + k;
		    	C->value[c_idx] = sum;
	    	}
    	}
    }
}

// Backward
// dL/dA = dL/dC @ B^T
// dL/dB = A^T @ dL/dC
void matmul_backward(Tensor* A, Tensor* B, Tensor*C){    
    int I = A->shape[A->ndim - 2];
    int J = A->shape[A->ndim - 1];
    int K = B->shape[B->ndim - 1];
    
    // All batch size except the last two dimensions
    int batch_size = get_batch_size(A);
    
	// 1. dA = dC @ B^T
    // [I,J] = [I, K] @ [K, J]
    for (int b = 0; b < batch_size; b++){
    	for (int i = 0; i < I; i++){
            for (int j = 0; j < J; j++){
                float sum = 0;
                for (int k = 0; k < K; k++){
                    int c_idx = b * I*K + i * K + k;
                    int b_idx = b * K*J + j * K + k; // B^T[k][j] = B[j][k]
                    sum += C->value[c_idx] * B->value[b_idx];
                }
                int a_idx = i * A->cols + j;
                A->grad[a_idx] += sum; // Grad should always be accumulated
            }
        }
    }

    // 2. dB = A^T * dC 
    // [J, K] = [J, I] @ [I, K]
    
    for (int b = 0; b < batch_size; b++){
        for (int j = 0; j < J; j++){
            for (int k = 0; k < K; k++){
                float sum = 0.0f;
                for (int i = 0; i < I; i++){
                    int a_idx = b * I*J + i * J + j; // A^T[j][i] = A[i][j]
                    int c_idx = b * I*K + i * K + k;
                    sum += A->value[a_idx] * C->value[c_idx];
                }
            int b_idx = b * J*K + j * K + k;
                B->grad[b_idx] += sum;
            }
        }
    }
}

// 2. Matrix addition (Forward, Backward): C = A + B

// Forward
void add_forward(Tensor* A, Tensor*B, Tensor*C){
    for (int i = 0; i < C->size; i++){
        C->value[i] = A->value[i] + B->value[i];
    }
}

// Backward
void add_backward(Tensor* A, Tensor* B, Tensor* C){
    for (int i = 0; i < C->size; i++){
        A->grad[i] += C->grad[i];
        B->grad[i] += C->grad[i];
    }
}

void apply_layernorm(Tensor* t, float* gamma, float* beta, float eps){
    int last_dim = t->shape[t->ndim - 1];
    int num_rows = t->size / last_dim;

    for (int i = 0; i < num_rows; i++){
        float* row = t->value + i * last_dim;
        
        // 1. Mean
        float sum = 0.0f;
        for (int j = 0; j < last_dim; j++) sum += row[j];
        float mean = sum / (float)last_dim;

        // 2. Variance
        float var = 0.0f;
        for (int j = 0; j < last_dim; j++) var += (row[j] - mean) * (row[j] - mean);

        // 3. Normalize
        for (int j = 0; j < last_dim; j++){
            float normalized = (row[j] - mean) / sqrtf(var + eps);
            
            // Initialize gamma and beta if NULL
            float g= (gamma != NULL) ? gamma[j] : 1;
            float b = (beta != NULL) ? beta[j] : 0;

            row[j] = normalized * g + b;

        }
    }
}

void scale_tensor(Tensor* t, float scale){
    for (int i = 0; i < t->size; i++){
        t->value[i] *= scale;
    }
}

void transpose(Tensor* in, Tensor* out){
    int batch_size = get_batch_size(in);
    int I = in->shape[in->ndim -2];
    int J = in->shape[in->ndim -1];

    for (int b = 0; b < batch_size; b++){
        for (int i = 0; i < I; i++){
            for (int j = 0; j < J; j++){
                int in_index = b * I*J + i * J + j;
                int out_index = b* I*J + j * I + i;
                out->values[out_index] = in->values[in_index];
            }
        }
    }
}


// RelU backward & forward
void relu_forward(Tensor* X, Tensor* Y) {
    for (int i = 0; i < X->size; i++) {
        Y->values[i] = X->values[i] > 0.0f ? X->values[i] : 0.0f;
    }
}

void relu_backward(float* x, float* dout, float* dx, int size) {
    for (int i = 0; i < size; i++){
        dx[i] = x[i] > 0.0f ? dout[i] : 0.0f;
    }
}

void sgd_update(Tensor* t, float lr){
    for (int i = 0; i < t->size; i++){
        t->values[i] -= lr * t->grads[i];
    }
}

// Softmax + Cross Entropy 

void apply_softmax(Tensor* t){
    int last_dim = t->shape[t->ndim - 1];
    int num_rows = t->size / last_dim;

    float max_val = -FLT_MAX;
    for (int i = 0; i < num_rows; i++){
        float* row = t->value + i * last_dim; // t->value is also a pointer

        // 1. Max trick: avoid exp overflow
        float max_val = -FLT_MAX;
        for (int j = 0; j < last_dim; j++)
            if (row[j] > max_val)
                max_val = row[j];

        float sum = 0.0f;
        for (int j = 0; j < last_dim; j++){
            row[j] = expf(row[j] - max_val);
            sum += row[j];
        }
        
        for (int j = 0; j < last_dim; j++) {
            row[j] /= sum;
        }
    }
}

float softmax_crossentropy_forward(float* logits, int target_class, float* probs, int num_classes){
    // Max trick
    float max_val = logits[0];
    
    for (int i = 1; i < num_classes; i++){
        if (logits[i] > max_val)
            max_val = logits[i];
    }

    //Softmax
    float sum_exp = 0.0f;
    for (int i = 0; i < num_classes; i++){
        probs[i] = expf(logits[i] - max_val);
        sum_exp == probs[i];
    }

    for (int i = 0; i < num_classes; i++){
        probs[i] /= sum_exp;
    }

    return -logf(probs[target_class] + 1e-7);
}

void softmax_crossentropy_backward(float* probs, int target_class, float* dlogits, int num_classes) {
    for (int i = 0; i < num_classes; i++){
        dlogits[i] = probs[i];
    }
    dlogits[target_calss] -= 1.0f;
}


