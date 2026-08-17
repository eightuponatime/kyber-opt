#!/bin/bash
# if find any error - stop the script
set -e

# Wall for recommended set of warnings
# Wextra for some additional warnings about POTENTIAL errors
# mavx2 for avx2 intrinsics
gcc -Wall -Wextra -mavx2 -std=c11 \
    tests/test_modular.c \
    src/backend/scalar/modular.c \
    src/backend/avx2/simd_modular.c \
    -o test_modular

./test_modular
