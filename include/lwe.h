#pragma once
#include <seal/seal.h>
/*typedef struct {
    size_t n;
    uint64_t modulus;
    std::vector<uint64_t> cipher;
} LWE;*/
struct LWE {
    LWE() {}
    LWE(size_t m, uint64_t mod)
    {
        n = m;
        modulus = mod;
        cipher.reserve(n + 1);
    }
    size_t n{0};
    uint64_t modulus{0};
    std::vector<uint64_t> cipher;
};