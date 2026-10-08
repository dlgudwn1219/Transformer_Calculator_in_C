#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#include "tensor.h"
#include "FFN.h"

void print_tensor_value(const char* name, Tensor* t){
    printf("--- %s ---\n", name);
    printf("shape:");
    for (int i = 0; i < t->ndim; i++) printf("%d ", t->shape[i]);
    printf("\n");

    int I = t->shape[t->ndim - 2];
    int J = t->shape[t->ndim - 1];
    
    int batch_size = get_batch_size(t);

    for (int b = 0; b < batch_size; b++){
        printf("batch %d\n", b);
        for (int i = 0; i < t->rows; i++){
            for (int j = 0; j < t->cols; j++)
                printf("%6.2f ", t->values[i * t->cols + j]);
            printf("\n");
        }
        printf("\n");
    }
}

void print_tensor_grad(const char* name, Tensor* t){
    printf("--- %s ---\n", name);
    for (int i = 0; i < t->rows; i++){
        for (int j = 0; j < t->cols; j++)
            printf("%6.2f ", t->grads[i * t->cols + j]);
        printf("\n");
    }
    printf("\n");
}

int main(){
    bool test_matrix_mul = false;
    bool test_ffn = true;

    if (test_matrix_mul){
       // 1. Create Tensor
        int A_shape[4] = {1, 1, 3, 2};
        int B_shape[4] = {1, 1, 2, 3};
        int C_shape[4] = {1, 1, 3, 3};

        Tensor* A = create_tensor(4, A_shape);
        Tensor* B = create_tensor(4, B_shape);
        Tensor* C = create_tensor(4, C_shape);

        // 2. Fill A, B with test data
        for (int i = 0; i < 6; i++) A->values[i] = i+1;
        for (int i = 0; i < 6; i++) B->values[i] = 6-i;

        // 3. Forward
        matmul_forward(A, B, C);
        printf("[Forward pass]\n");
        print_tensor_value("A value", A);
        print_tensor_value("B value", B);
        print_tensor_value("C = A @ B value: ", C);

        // 4. Backward
        for (int i = 0; i < C->size; i++) C->grads[i] = 1.0f;
        matmul_backward(A, B, C);

        print_tensor_grad("A grad", A); 
        print_tensor_grad("B grad", B);
        print_tensor_grad("C = A @ B grad: ", C);
    
        free_tensor(A);
        free_tensor(B);
        free_tensor(C);
    }
    
    if (test_ffn){
        // 5. Define ffn
        int X_shape[3] = {3, 1, 5};
        int Y_shape[3] = {3, 1, 3};
        int ffn_shape[3] = {5, 4, 3};

        Tensor* X = create_tensor(3, X_shape);
        for (int b = 0; b < 3; b++)
            for (int i = 0; i < 5; i++)
                X->values[b*5+i] = ((b + i) % 5) / 5;
    
        Tensor* Y = create_tensor(3, Y_shape);
        for (int b = 0; b < 3; b++)
            for (int i = 0; i < 3; i++)
                Y->values[b*5+i] = (i == b ? 1 : 0);

        FFN* simple_net = create_ffn(2, ffn_shape);

        // Initialize weight matrix to 1
        for (int i = 0; i < simple_net->depth; i++){
            int size_i = simple_net->shape[i] * simple_net->shape[i+1];
        for (int j = 0; j < size_i; j++)
            simple_net->weights[i]->values[j] = ((float)(j * 47 % 11) / 12.0f);
        }

        // 6. ffn forward
        printf("Forwarding network..\n");
        ffn_forward(simple_net, X);

        printf("Forwarding completed!!\n");
        print_tensor_value("X value", X);
        
        
        for (int i = 0; i < simple_net->depth; i++)
            print_tensor_value("W value", simple_net->weights[i]);
        
        for (int i = 0; i < simple_net->depth; i++)
            print_tensor_value("Z value", simple_net->results[i]);
        
        // ffn backward
        printf("Backwarding network...\n");
        ffn_backward(simple_net, Y);

        printf("Backward completed!!\n");
        // 5. Free Memory
        free_tensor(X);
        free_tensor(Y);
        free_ffn(simple_net);
    }

    return 0;
}
