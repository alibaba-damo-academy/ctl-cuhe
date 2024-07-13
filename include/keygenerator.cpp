// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "keygenerator.h"
//#include "lwe/lweencryptor.h"
#include "params.h"
#include "seal.inc"
#include <algorithm>

using namespace std;
using namespace seal::util;

namespace seal
{
KeyGeneratorM::KeyGeneratorM(const SEALContext &context) : context_(context)
{
    // Verify parameters
    if (!context_.parameters_set()) {
        throw invalid_argument("encryption parameters are not set correctly");
    }
    // Generate the secret and public key
    generate_sk();
}

void KeyGeneratorM::generate_sk(bool is_initialized)
{
    // Extract encryption parameters.
    auto &context_data = *context_.key_context_data();
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    size_t coeff_count = parms.poly_modulus_degree();
    size_t coeff_modulus_size = coeff_modulus.size();

        secret_key_ = SecretKey();
        secret_key_.data().resize(mul_safe(coeff_count, coeff_modulus_size));
        /*for (size_t m = 0; m < coeff_modulus_size; m++) {
            secret_key_.data().data()[0 + m * targetP::n_] = 1;
            for (size_t i = 1; i < coeff_count; i++)
                secret_key_.data().data()[i + m * targetP::n_] = 0;
        }*/
        // Generate secret key
        RNSIter secret_key(secret_key_.data().data(), coeff_count);
        sample_poly_binary(parms.random_generator()->create(), parms, secret_key);

        secret_key_not_ntt = SecretKey(secret_key_);
        // Transform the secret s into NTT representation.
        auto ntt_tables = context_data.small_ntt_tables();
        ntt_negacyclic_harvey(secret_key, coeff_modulus_size, ntt_tables);
        /*for (int i = 0; i < coeff_count; i++)
            cout << secret_key_.data().data()[i] << "\t";
        cout << endl;*/
        // Set the parms_id for secret key
        secret_key_.parms_id() = context_data.parms_id();

}

namespace util
{
void sample_poly_binary(
    shared_ptr<UniformRandomGenerator> prng, const EncryptionParameters &parms, uint64_t *destination)
{
    auto coeff_modulus = parms.coeff_modulus();
    size_t coeff_modulus_size = coeff_modulus.size();
    size_t coeff_count = parms.poly_modulus_degree();

    RandomToStandardAdapter engine(prng);
    uniform_int_distribution<uint64_t> dist(0, 1);

    SEAL_ITERATE(iter(destination), coeff_count, [&](auto &I) {
                uint64_t rand = dist(engine);
                // uint64_t flag = static_cast<uint64_t>(-static_cast<int64_t>(rand == 0));
                SEAL_ITERATE(
                    iter(StrideIter<uint64_t *>(&I, coeff_count), coeff_modulus), coeff_modulus_size,
                    [&](auto J) { *get<0>(J) = rand; }); });
}
void tfhe_sample_poly_normal(
    shared_ptr<UniformRandomGenerator> prng, const EncryptionParameters &parms, uint64_t *destination)
{
    auto coeff_modulus = parms.coeff_modulus();
    size_t coeff_modulus_size = coeff_modulus.size();
    size_t coeff_count = parms.poly_modulus_degree();

    RandomToStandardAdapter engine(prng);
    ClippedNormalDistribution dist(
        0, targetP::stddev * std::pow(2.0, targetP::digits), targetP::maxdev * std::pow(2.0, targetP::digits));
    SEAL_ITERATE(iter(destination), coeff_count, [&](auto &I) {
        int64_t noise = static_cast<int64_t>(dist(engine));
        uint64_t flag = static_cast<uint64_t>(-static_cast<int64_t>(noise < 0));
        SEAL_ITERATE(
            iter(StrideIter<uint64_t *>(&I, coeff_count), coeff_modulus), coeff_modulus_size,
            [&](auto J) { *get<0>(J) = static_cast<uint64_t>(noise) + (flag & get<1>(J).value()); });
    });
}
void tfhe_encrypt_zero_symmetric(
    const SecretKey &secret_key, const SEALContext &context, parms_id_type parms_id, bool is_ntt_form,
    bool save_seed, Ciphertext &destination)
{

    // We use a fresh memory pool with `clear_on_destruction' enabled.
    MemoryPoolHandle pool = MemoryManager::GetPool(mm_prof_opt::mm_force_new, true);

    auto &context_data = *context.get_context_data(parms_id);
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    size_t coeff_modulus_size = coeff_modulus.size();
    size_t coeff_count = parms.poly_modulus_degree();
    auto ntt_tables = context_data.small_ntt_tables();
    size_t encrypted_size = 2;

    // If a polynomial is too small to store UniformRandomGeneratorInfo,
    // it is best to just disable save_seed. Note that the size needed is
    // the size of UniformRandomGeneratorInfo plus one (uint64_t) because
    // of an indicator word that indicates a seeded ciphertext.
    size_t poly_uint64_count = mul_safe(coeff_count, coeff_modulus_size);
    size_t prng_info_byte_count =
        static_cast<size_t>(UniformRandomGeneratorInfo::SaveSize(compr_mode_type::none));
    size_t prng_info_uint64_count =
        divide_round_up(prng_info_byte_count, static_cast<size_t>(bytes_per_uint64));
    if (save_seed && poly_uint64_count < prng_info_uint64_count + 1) {
        save_seed = false;
    }

    destination.resize(context, parms_id, encrypted_size);
    destination.is_ntt_form() = is_ntt_form;
    destination.scale() = 1.0;

    // Create an instance of a random number generator. We use this for sampling
    // a seed for a second PRNG used for sampling u (the seed can be public
    // information. This PRNG is also used for sampling the noise/error below.
    auto bootstrap_prng = parms.random_generator()->create();

    // Sample a public seed for generating uniform randomness
    prng_seed_type public_prng_seed;
    bootstrap_prng->generate(prng_seed_byte_count, reinterpret_cast<seal_byte *>(public_prng_seed.data()));

    // Set up a new default PRNG for expanding u from the seed sampled above
    auto ciphertext_prng = UniformRandomGeneratorFactory::DefaultFactory()->create(public_prng_seed);

    // Generate ciphertext: (c[0], c[1]) = ([-(as+e)]_q, a)
    uint64_t *c0 = destination.data();
    uint64_t *c1 = destination.data(1);

    // Sample a uniformly at random
    if (is_ntt_form || !save_seed) {
        // Sample the NTT form directly
        sample_poly_uniform(ciphertext_prng, parms, c1);
    } else if (save_seed) {
        // Sample non-NTT form and store the seed
        sample_poly_uniform(ciphertext_prng, parms, c1);
        for (size_t i = 0; i < coeff_modulus_size; i++) {
            // Transform the c1 into NTT representation
            ntt_negacyclic_harvey(c1 + i * coeff_count, ntt_tables[i]);
        }
    }

    // Sample e <-- chi
    auto noise(allocate_poly(coeff_count, coeff_modulus_size, pool));
    tfhe_sample_poly_normal(bootstrap_prng, parms, noise.get());
    // SEAL_NOISE_SAMPLER(bootstrap_prng, parms, noise.get());
    for (size_t i = 0; i < 100; i++)
        cout << noise.get()[i] << "\t";
    cout << endl;
    exit(0);
    // Calculate -(a*s + e) (mod q) and store in c[0]
    for (size_t i = 0; i < coeff_modulus_size; i++) {
        dyadic_product_coeffmod(
            secret_key.data().data() + i * coeff_count, c1 + i * coeff_count, coeff_count, coeff_modulus[i],
            c0 + i * coeff_count);
        if (is_ntt_form) {
            // Transform the noise e into NTT representation
            ntt_negacyclic_harvey(noise.get() + i * coeff_count, ntt_tables[i]);
        } else {
            inverse_ntt_negacyclic_harvey(c0 + i * coeff_count, ntt_tables[i]);
        }
        add_poly_coeffmod(
            noise.get() + i * coeff_count, c0 + i * coeff_count, coeff_count, coeff_modulus[i],
            c0 + i * coeff_count);
    }

    if (!is_ntt_form && !save_seed) {
        for (size_t i = 0; i < coeff_modulus_size; i++) {
            // Transform the c1 into non-NTT representation
            inverse_ntt_negacyclic_harvey(c1 + i * coeff_count, ntt_tables[i]);
        }
    }

    if (save_seed) {
        UniformRandomGeneratorInfo prng_info = ciphertext_prng->info();

        // Write prng_info to destination.data(1) after an indicator word
        c1[0] = static_cast<uint64_t>(0xFFFFFFFFFFFFFFFFULL);
        prng_info.save(reinterpret_cast<seal_byte *>(c1 + 1), prng_info_byte_count, compr_mode_type::none);
    }
}
} // namespace util

} // namespace seal
