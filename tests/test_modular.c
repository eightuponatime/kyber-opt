#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "../src/backend/scalar/modular.h"

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

int main(void) {
    test_mod_add();
    test_mod_sub();

    return 0;
}
