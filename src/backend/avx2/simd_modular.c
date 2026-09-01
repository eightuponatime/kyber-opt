#include "simd_modular.h"
#include <immintrin.h>
#include <stdio.h>
#include <stdalign.h>

// we have 256 polynome
// Add(Vector256<Int16>, Vector256<Int16>)
// __m256i _mm256_add_epi16 (__m256i a, __m256i b)

void simd_mod_add(int16_t *result_arr, int16_t a[16], int16_t b[16]) {
    __m256i ymm_a = _mm256_loadu_si256((const __m256i *) a);
    __m256i ymm_b = _mm256_loadu_si256((const __m256i *) b);

    __m256i sum = _mm256_add_epi16(ymm_a, ymm_b);

    __m256i q = _mm256_set1_epi16(Q);

    __m256i q_minus_one = _mm256_set1_epi16(Q - 1);
    __m256i mask = _mm256_cmpgt_epi16(sum, q_minus_one);

    __m256i q_to_subtract = _mm256_and_si256(mask, q);

    __m256i reduced = _mm256_sub_epi16(sum, q_to_subtract);

    _mm256_storeu_si256((__m256i *)result_arr, reduced);
}

void simd_mod_sub(int16_t *result_arr, int16_t a[16], int16_t b[16]) {
    __m256i ymm_a = _mm256_loadu_si256((__m256i const *) a);
    __m256i ymm_b = _mm256_loadu_si256((__m256i const *) b);

    __m256i sub = _mm256_sub_epi16(ymm_a, ymm_b);

    __m256i zeros = _mm256_set1_epi16(0);
    __m256i cmp_res = _mm256_cmpgt_epi16(zeros, sub);

    __m256i q = _mm256_set1_epi16(Q);
    __m256i correction = _mm256_and_si256(cmp_res, q);

    __m256i result = _mm256_add_epi16(sub, correction);

    _mm256_storeu_si256((__m256i *)result_arr, result);
}


/*
(gdb) p b_odd
$5 = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}
(gdb) p a_even
$6 = {1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 0, 13, 0, 15, 0}
(gdb) p a_odd
$7 = {0, 2, 0, 4, 0, 6, 0, 8, 0, 10, 0, 12, 0, 14, 0, -16}
(gdb) p b_even
$8 = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0}
(gdb) p b_odd
$9 = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}
 */
void arr_processor (int16_t *num, int16_t *num_odd, int16_t *num_even) {
    for (size_t i = 0; i < 16; ++i) {
        if (i % 2 == 0) {
            num_even[i] = num[i];
            num_odd[i] = 0;
        } else {
            num_odd[i] = num[i];
            num_even[i] = 0;
        }
    }
}

void simd_mod_mul(int16_t *result_arr, int16_t a[16], int16_t b[16]) {

    int16_t a_odd[16];
    int16_t a_even[16];
    arr_processor(a, a_odd, a_even);

    int16_t b_odd[16];
    int16_t b_even[16];
    arr_processor(b, b_odd, b_even);

    __m256i qinv = _mm256_set1_epi16 (QINV);

    // ============ ODD ============
    __m256i ymm_a_odd = _mm256_loadu_si256((const __m256i *) a_odd);
    __m256i ymm_b_odd = _mm256_loadu_si256((const __m256i *) b_odd);

    __m256i mul_odd = _mm256_mullo_epi16 (ymm_a_odd, ymm_b_odd);
    __m256i reduction_coeff_odd = _mm256_madd_epi16(mul_odd, qinv);

    alignas(32) int32_t mul_results_odd[8];
    _mm256_store_si256 ((__m256i *)mul_results_odd, reduction_coeff_odd);
    // ============ EVEN ============
    __m256i ymm_a_even = _mm256_loadu_si256((const __m256i *) a_even);
    __m256i ymm_b_even = _mm256_loadu_si256((const __m256i *) b_even);

    __m256i mul_even = _mm256_mullo_epi16 (ymm_a_even, ymm_b_even);
    __m256i reduction_coeff_even = _mm256_madd_epi16(mul_even, qinv);

    alignas(32) int32_t mul_results_even[8];
    _mm256_store_si256 ((__m256i *)mul_results_even, reduction_coeff_even);
    // ==============================
    // t = (a - (int32_t)t * Q) >> 16;

    __m256i q = _mm256_set1_epi32(Q);
    // we don't need mul because this multiplication can't give
    // the result more than 2^32, the biggest val is 177 209 328 (53232*3329)
    // TODO: check if epi16 is ok for this case;
    __m256i mod_mul_odd = _mm256_mullo_epi16(reduction_coeff_odd, q);
    __m256i mod_mul_even = _mm256_mullo_epi16(reduction_coeff_even, q);

    __m256i sub_odd = _mm256_sub_epi32(mul_odd, reduction_coeff_odd);
    __m256i sub_even = _mm256_sub_epi32(mul_even, reduction_coeff_even);

    __m256i t_odd = _mm256_srli_si256 (sub_odd, 16);
    __m256i t_even = _mm256_srli_si256 (sub_even, 16);
}

int main(void) {
    int16_t a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -16};
    int16_t b[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int16_t result[16];

    simd_mod_mul(result, a, b);
}
