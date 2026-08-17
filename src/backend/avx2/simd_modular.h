#ifndef SIMD_MODULAR_H
#define SIMD_MODULAR_H

#include <stdint.h>
#include "../modular_params.h"

void simd_mod_add(int16_t *result, int16_t a[16], int16_t b[16]);

#endif // SIMD_MODULAR_H
