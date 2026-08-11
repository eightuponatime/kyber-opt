#ifndef SCALAR_MODULAR_H
#define SCALAR_MODULAR_H

#include <stdint.h>
#include "../modular_params.h"

int16_t mod_add(int16_t a, int16_t b);
int16_t mod_sub(int16_t a, int16_t b);
int16_t mod_mul(int16_t a, int16_t b);
int16_t montgomery_reduce(int32_t a);
int16_t barrett_reduce(int16_t a);

#endif
