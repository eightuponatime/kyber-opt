#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#include "../src/backend/scalar/modular.h"
#include "../src/backend/avx2/simd_modular.h"

/*
** SCALAR MOD OPS
 */

static void test_mod_add(void) {
    struct {
        int16_t a;
        int16_t b;
    } cases[] = {
        {0, 0},
        {1, 1},
        {Q - 1, 0},
        {Q - 1, 1},
        {Q - 1, Q - 1},
        {Q / 2, Q / 2},
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        int32_t expected = (cases[i].a + cases[i].b) % Q;
        int16_t actual = mod_add(cases[i].a, cases[i].b);

        assert(actual == expected);
    }

    printf("modd_add: OK\n");
}

void test_mod_sub(void) {
    struct {
        int16_t a;
        int16_t b;
    } cases[] = {
        {0, 0},
        {1, 0},
        {0, 1},
        {Q - 1, 0},
        {0, Q - 1},
        {Q - 1, Q - 1},
        {Q / 2, Q / 2},
        {Q / 2, Q / 2 + 1},
        {Q / 2 + 1, Q / 2},
        {1, Q - 1},
        {Q - 1, 1},
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        int16_t expected = ((cases[i].a - cases[i].b) % Q + Q) % Q;
        int16_t actual = mod_sub(cases[i].a, cases[i].b);

        assert(expected == actual);
    }

    printf("mod_sub: OK\n");
}

void test_mod_mul (void) {
    struct {
        int16_t a;
        int16_t b;
    } cases[] = {
        {1, Q - 1},
        {Q - 1, 1},
        {Q / 2, Q / 2},
        {Q / 2, Q / 2 + 1}};

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        int16_t cur_a = cases[i].a;
        int16_t cur_b = cases[i].b;

        int32_t r = 1 << 16;
        int32_t r_inv = 0;
        for (int32_t x = 1; x < Q; ++x) {
            if ((r * x) % Q == 1) {
                r_inv = x;
                break;
            }
        }
        int32_t const_val = r_inv % Q; // 169
        int16_t expected = ((int32_t)cur_a * cur_b * const_val) % Q;
        if (expected > Q / 2) {
            expected -= Q;
        }

        int16_t actual = mod_mul(cur_a, cur_b);

        assert(expected == actual);
    }

    printf("mod_mul: OK\n");
}

void test_barret_reduction (void) {
    int16_t cases[] = {
    0, 1, -1,
    Q - 1, Q, Q + 1,
    Q / 2 - 1, Q / 2, Q / 2 + 1,
    INT16_MAX, INT16_MIN};

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        int16_t a = cases[i];
        int16_t actual = barrett_reduce(a);

        int32_t actual_mod = actual % Q;
        if (actual_mod < 0) {
            actual_mod += Q;
        }

        int32_t expected = a % Q;
        if (expected < 0) {
            expected += Q;
        }

        assert(actual_mod == expected);

        assert(actual >= -(Q / 2));
        assert(actual <= Q / 2);
    }

    printf("barrett_reduce: OK\n");
}

/*
** SIMD OPTIMIZED MOD OPS
 */

void test_simd_mod_add(void) {
    int16_t actual[16];

    int16_t test_a[16] = {1, 2, 3, 4, 5, 6, 7, 8,
    9, 10, 11, 12, 13, 14, 15, 16};

    int16_t test_b[16] = {1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1};

    simd_mod_add(actual, test_a, test_b);

    int16_t expected[16] = {2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 15, 16, 17};

    for (size_t i = 0; i < sizeof(expected) / sizeof(expected[0]); ++i) {
        assert(actual[i] == expected[i]);
    }

    printf("simd_mod_add: OK\n");
}

int main(void) {
    test_mod_add();
    test_mod_sub();
    test_mod_mul();
    test_barret_reduction();
    test_simd_mod_add();

    return 0;
}
