#ifndef SCALAR_MODULAR_H
#define SCALAR_MODULAR_H

#include <stdint.h>
#include "../modular_params.h"

int32_t mod_add(int32_t a, int32_t b);
int32_t mod_sub(int32_t a, int32_t b);
int32_t montgomery_reduce(int64_t a);
int32_t barrett_reduce(int32_t a);

#endif
