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

// ============ MONTGOMERY REDUCE ============

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

typedef struct {
    __m256i i32_mul;
    __m256i reduction_coeff;
} ReductionCoeff;

ReductionCoeff get_reduction_coeff(int16_t a[16], int16_t b[16]) {
    __m256i ones = _mm256_set1_epi16 (1);
    __m256i qinv = _mm256_set1_epi16 (QINV);

    __m256i ymm_a = _mm256_loadu_si256((const __m256i *) a);
    __m256i ymm_b = _mm256_loadu_si256((const __m256i *) b);

    __m256i mul = _mm256_mullo_epi16 (ymm_a, ymm_b);
    __m256i i32_mul = _mm256_madd_epi16(mul, ones);
    __m256i reduction_coeff = _mm256_madd_epi16(mul, qinv);

    ReductionCoeff res = { i32_mul, reduction_coeff };
    return res;
}

__m256i shift_with_saved_sign(__m256i sub_result) {
    __m256i sign_mask = _mm256_set1_epi32(INT32_MIN);
    __m256i sign_exclude = _mm256_and_si256(sub_result,sign_mask);
    __m256i cmp_signs = _mm256_cmpeq_epi32(sign_exclude, sign_mask);
    __m256i inv_vec = _mm256_xor_si256(sub_result, cmp_signs);
    __m256i sign_exc_shift = _mm256_srli_epi32 (sign_exclude, 31);
    __m256i inc_by_one = _mm256_add_epi32 (inv_vec, sign_exc_shift);
    __m256i unsigned_t_odd = _mm256_srli_epi32 (inc_by_one, 16);
    __m256i signed_t_odd = _mm256_xor_si256 (unsigned_t_odd, cmp_signs);
    return _mm256_add_epi32(signed_t_odd, sign_exc_shift);
}


__m256i optimized_packing (__m256i res_odd, __m256i res_even) {
    __m128i lo_4_odd = _mm256_castsi256_si128(res_odd);
    __m128i hi_4_odd = _mm256_extracti128_si256(res_odd, 1);
    __m128i lo_4_even = _mm256_castsi256_si128(res_even);
    __m128i hi_4_even = _mm256_extracti128_si256(res_even, 1);

    __m128i shuffle_lo1_odd = _mm_shuffle_epi32(lo_4_odd, _MM_SHUFFLE(3,2,1,0));
    __m128i shuffle_lo1_ev = _mm_shuffle_epi32(lo_4_even, _MM_SHUFFLE(3,2,1,0));
    __m128i shuffle_lo2_odd = _mm_shuffle_epi32(lo_4_odd, _MM_SHUFFLE(0,1,3,2));
    __m128i shuffle_lo2_ev = _mm_shuffle_epi32(lo_4_even, _MM_SHUFFLE(0,1,3,2));

    __m128i shuffle_hi1_odd = _mm_shuffle_epi32(hi_4_odd, _MM_SHUFFLE(3,2,1,0));
    __m128i shuffle_hi1_ev = _mm_shuffle_epi32(hi_4_even, _MM_SHUFFLE(3,2,1,0));
    __m128i shuffle_hi2_odd = _mm_shuffle_epi32(hi_4_odd, _MM_SHUFFLE(0,1,3,2));
    __m128i shuffle_hi2_ev = _mm_shuffle_epi32(hi_4_even, _MM_SHUFFLE(0,1,3,2));

    __m128i res_lo1 = _mm_unpacklo_epi32(shuffle_lo1_odd, shuffle_lo1_ev);
    __m128i res_lo2 = _mm_unpacklo_epi32(shuffle_lo2_odd, shuffle_lo2_ev);
    __m128i res_hi1 = _mm_unpacklo_epi32(shuffle_hi1_odd, shuffle_hi1_ev);
    __m128i res_hi2 = _mm_unpacklo_epi32(shuffle_hi2_odd, shuffle_hi2_ev);

    __m128i lo = _mm_packs_epi32 (res_lo1, res_lo2);
    __m128i hi = _mm_packs_epi32 (res_hi1, res_hi2);

    __m256i packs = _mm256_set_m128i (hi, lo);

    return packs;
}

void opt_simd_mod_mul(int16_t *result_arr, int16_t a[16], int16_t b[16]) {

    int16_t a_odd[16];
    int16_t a_even[16];
    arr_processor(a, a_odd, a_even);

    int16_t b_odd[16];
    int16_t b_even[16];
    arr_processor(b, b_odd, b_even);

    // t = (a - (int32_t)t * Q) >> 16;

    ReductionCoeff rc_odd = get_reduction_coeff(a_odd, b_odd);
    __m256i reduction_coeff_odd = rc_odd.reduction_coeff;
    __m256i i32_mul_odd = rc_odd.i32_mul;

    ReductionCoeff rc_even = get_reduction_coeff(a_even, b_even);
    __m256i reduction_coeff_even = rc_even.reduction_coeff;
    __m256i i32_mul_even = rc_even.i32_mul;

    __m256i q = _mm256_set1_epi32(Q);

    __m256i mod_mul_odd = _mm256_mullo_epi32(reduction_coeff_odd, q);
    __m256i mod_mul_even = _mm256_mullo_epi32(reduction_coeff_even, q);

    __m256i sub_odd = _mm256_sub_epi32(i32_mul_odd, mod_mul_odd);
    __m256i sub_even = _mm256_sub_epi32(i32_mul_even, mod_mul_even);

    __m256i res_odd = shift_with_saved_sign(sub_odd);
    __m256i res_even = shift_with_saved_sign(sub_even);

    __m256i packed_result = optimized_packing(res_odd, res_even);
    _mm256_storeu_si256 ((__m256i *) result_arr, packed_result);
}

// ============ BARRETT REDUCE ============


void barrett_reduce(int16_t *result, int16_t arr[16]) {
    int16_t odd_arr[16];
    int16_t even_arr[16];
    arr_processor(arr, odd_arr, even_arr);

    __m256i odd_a = _mm256_loadu_si256((const __m256i *) odd_arr);
    __m256i even_a = _mm256_loadu_si256((const __m256i *) even_arr);

    const int16_t v = ((1L << 26) + Q / 2) / Q;
    __m256i v_vec = _mm256_set1_epi16 (v);

    const int32_t round_to_nearest = 1L << 25;
    __m256i rtn_vec = _mm256_set1_epi16(round_to_nearest);
/*
    t = ((int32_t)v * a + (1L << 25)) >> 26;
    t *= Q;
    return a - t;
    * */

}

// ============ UNOPTIMIZED DEMONSTRATIONAL PACKING ============
void simd_mod_mul(int32_t *result_arr, int16_t a[16], int16_t b[16]) {

    int16_t a_odd[16];
    int16_t a_even[16];
    arr_processor(a, a_odd, a_even);

    int16_t b_odd[16];
    int16_t b_even[16];

    arr_processor(b, b_odd, b_even);

    // t = (a - (int32_t)t * Q) >> 16;

    ReductionCoeff rc_odd = get_reduction_coeff(a_odd, b_odd);
    __m256i reduction_coeff_odd = rc_odd.reduction_coeff;
    __m256i i32_mul_odd = rc_odd.i32_mul;

    ReductionCoeff rc_even = get_reduction_coeff(a_even, b_even);
    __m256i reduction_coeff_even = rc_even.reduction_coeff;
    __m256i i32_mul_even = rc_even.i32_mul;

    __m256i q = _mm256_set1_epi32(Q);

    __m256i mod_mul_odd = _mm256_mullo_epi32(reduction_coeff_odd, q);
    __m256i mod_mul_even = _mm256_mullo_epi32(reduction_coeff_even, q);

    __m256i sub_odd = _mm256_sub_epi32(i32_mul_odd, mod_mul_odd);
    __m256i sub_even = _mm256_sub_epi32(i32_mul_even, mod_mul_even);

    __m256i res_odd = shift_with_saved_sign(sub_odd);
    __m256i res_even = shift_with_saved_sign(sub_even);

    // === packing ===

    alignas(32) int32_t odd_arr[8];
    _mm256_storeu_si256((__m256i *) odd_arr, res_odd);

    alignas(32) int32_t even_arr[8];
    _mm256_storeu_si256((__m256i *) even_arr, res_even);

    int n = sizeof(odd_arr) * 2 / sizeof(odd_arr[0]);
    for (int i = 0; i < n; ++i) {
        if (i % 2 == 0) {
            result_arr[i] = odd_arr[i / 2];
        } else {
            result_arr[i] = even_arr[i / 2];
        }
    }
}

int main(void) {
    int16_t a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -16};
    int16_t b[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    alignas(16) int16_t opt_result[16];
    barrett_reduce(opt_result, a);
}
