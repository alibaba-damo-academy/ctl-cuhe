// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

//#include "lwe/lwe.h"
#include "seal.inc"
#include <random>

using namespace std;

namespace seal
{
class KeyGeneratorM
{
  public:
    KeyGeneratorM(const SEALContext &context);
    SEAL_NODISCARD const SecretKey &secret_key() const
    {
        return secret_key_;
    }
    SEAL_NODISCARD const SecretKey &secret_key_intt() const
    {
        return secret_key_not_ntt;
    }

  private:
    void generate_sk(bool is_initialized = false);
    SEALContext context_;
    SecretKey secret_key_;
    SecretKey secret_key_not_ntt;
};

namespace util
{
void sample_poly_binary(
    shared_ptr<UniformRandomGenerator> prng, const EncryptionParameters &parms, uint64_t *destination);
void tfhe_sample_poly_normal(
    shared_ptr<UniformRandomGenerator> prng, const EncryptionParameters &parms, uint64_t *destination);
void tfhe_encrypt_zero_symmetric(
    const SecretKey &secret_key, const SEALContext &context, parms_id_type parms_id, bool is_ntt_form,
    bool save_seed, Ciphertext &destination);
} // namespace util
} // namespace seal