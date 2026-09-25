#include <stdio.h>
#include "tensor.h"

void print_tensor_value(const char* name, Tensor* t){
    printf("--- %s ---\n", name);
    for (int i = 0; i < t->rows; i++){
        for (int j = 0; j < t->cols; j++)
            printf("%6.2f ", t->value[i * t->cols + j]);
        printf("\n");
    }
    printf("\n");
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
    Tensor* A = create_tensor(3,2);
    Tensor* B = create_tensor(2,3);
    Tensor* C = create_tensor(2,2);

    // 2. Fill A, B with test data
    for (int i = 0; i < 6; i++) A->value[i] = i+1;
    for (int i = 0; i < 6; i++) B->value[i] = 7-i;

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
