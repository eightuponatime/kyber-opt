#!/usr/bin/env sh

gcc -g -O0 -Wall -Wextra -mavx2 simd_modular.c -o simd_modular
gdb ./simd_modular -x commands.txt
