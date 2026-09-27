#include <stdio.h>
#include "tensor.h"

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
                printf("%6.2f ", t->value[i * t->cols + j]);
            printf("\n");
        }
        printf("\n");
    }
}

void print_tensor_grad(const char* name, Tensor* t){
    printf("--- %s ---\n", name);
    for (int i = 0; i < t->rows; i++){
        for (int j = 0; j < t->cols; j++)
            printf("%6.2f ", t->grad[i * t->cols + j]);
        printf("\n");
    }
    printf("\n");
}

int main(){
    // 1. Create Tensor
    int A_shape[4] = {1, 1, 3, 2};
    int B_shape[4] = {1, 1, 2, 3};
    int C_shape[4] = {1, 1, 3, 3};

    Tensor* A = create_tensor(4, A_shape);
    Tensor* B = create_tensor(4, B_shape);
    Tensor* C = create_tensor(4, C_shape);

    // 2. Fill A, B with test data
    for (int i = 0; i < 6; i++) A->value[i] = i+1;
    for (int i = 0; i < 6; i++) B->value[i] = 6-i;

    // 3. Forward
    matmul_forward(A, B, C);
    printf("[Forward pass]\n");
    print_tensor_value("A value", A);
    print_tensor_value("B value", B);
    print_tensor_value("C = A @ B value: ", C);

    // 4. Backward
    for (int i = 0; i < C->size; i++) C->grad[i] = 1.0f;
    matmul_backward(A, B, C);

    print_tensor_grad("A grad", A); 
    print_tensor_grad("B grad", B);
    print_tensor_grad("C = A @ B grad: ", C);
    // 5. Free Memory
    free_tensor(A);
    free_tensor(B);
    free_tensor(C);

    return 0;
}
