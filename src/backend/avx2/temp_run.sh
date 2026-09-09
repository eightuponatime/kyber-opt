#!/usr/bin/env sh

gcc -g -mavx2 -O0 simd_modular.c -o simd_modular
gdb ./simd_modular -x temp_commands.txt
