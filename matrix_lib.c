/*
    INF 1029 - Trab1 2024.1
    Michel Anísio Almeida - 1521767
    Alexandre César Brandão de Andrade - 2010292 
*/

#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include "matrix_lib.h"



int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
    if (matrix == NULL || matrix->rows == NULL) {
        return 0;
    }

    unsigned long int total_elements = matrix->height * matrix->width;
    __m256 v_scalar = _mm256_set1_ps(scalar_value);
    float *p = matrix->rows;  

    unsigned long int i;
    for (i = 0; i + 7 < total_elements; i += 8) {
        __m256 v_data = _mm256_loadu_ps(p);  
        __m256 v_result = _mm256_mul_ps(v_data, v_scalar); 
        _mm256_storeu_ps(p, v_result);  
        p += 8; 
    }

    for (; i < total_elements; i++) {
        *p *= scalar_value;  
        p++;  
    }

    return 1;
}


int matrix_matrix_mult(struct matrix *a, struct matrix *b, struct matrix *c) {
  if (a->width != b->height || c->height != a->height || c->width != b->width) return 0;

  for (int i = 0; i < a->height; i++) {
    for (int j = 0; j < b->width; j++) {
      __m256 v_sum = _mm256_setzero_ps();

      for (int k = 0; k < a->width; k += 8) {
        __m256 v_a = _mm256_loadu_ps(a->rows + i * a->width + k);
        __m256 v_b = _mm256_loadu_ps(b->rows + k * b->width + j);
        v_sum = _mm256_fmadd_ps(v_a, v_b, v_sum);
      }

        _mm256_storeu_ps(c->rows + i * c->width + j, v_sum);
    }
  }

    return 1;
}