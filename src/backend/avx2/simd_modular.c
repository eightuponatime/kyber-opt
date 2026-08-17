#include "simd_modular.h"
#include <immintrin.h>

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
