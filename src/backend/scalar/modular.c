#include "modular.h"

int16_t mod_add(int16_t a, int16_t b) {
    int16_t res = a + b;
    if (res >= Q) {
        return res - Q;
    } else {
        return res;
    }
}

int16_t mod_sub(int16_t a, int16_t b) {
    int32_t res = a - b;
    if (res < 0) {
        return res + Q;
    } else {
        return res;
    }
}

int16_t montgomery_reduce(int32_t a) {
    int16_t t;

    t = (int16_t)a * QINV;
    t = (a - (int32_t)t * Q) >> 16;

    return t;
}

int16_t mod_mul(int16_t a, int16_t b) {
    return montgomery_reduce((int32_t)a * b);
}

int16_t barrett_reduce(int16_t a) {
    int16_t t;
    const int16_t v = ((1L << 26) + Q / 2) / Q;

    t = ((int32_t)v * a + (1L << 25)) >> 26;
    t *= Q;
    return a - t;

}
