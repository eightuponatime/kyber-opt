#!/bin/bash

# Wall for recommended set of warnings
# Wextra for some additional warnings about POTENTIAL errors
gcc -Wall -Wextra -std=c11 \
    tests/test_modular.c \
    src/backend/scalar/modular.c \
    -o test_modular

./test_modular
