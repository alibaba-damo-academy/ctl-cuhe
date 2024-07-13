#include "glwe.h"
#include "omp.h"
#include "seal.inc"
#include "timer.h"
using namespace std;
namespace seal
{
EncryptionParameters ConvertContext(SEALContext &context)
{
    EncryptionParameters bfvparams(scheme_type::bfv);
    auto &params = context.key_context_data().get()->parms();
    bfvparams.set_poly_modulus_degree(params.poly_modulus_degree() / targetP::k_);
    bfvparams.set_coeff_modulus(params.coeff_modulus());
    bfvparams.set_plain_modulus(params.plain_modulus());
    return bfvparams;
}
util::Pointer<uint64_t> ConvertSKArr(SEALContext &context, SecretKey &secret_key)
{
    auto &context_data = *context.key_context_data();
    auto &coeff_modulus = context_data.parms().coeff_modulus();
    auto coeff_modulus_size = coeff_modulus.size();
    auto poly_modulus_degree = context_data.parms().poly_modulus_degree();
    auto ntt_tables = context_data.small_ntt_tables();
    auto secret_key_array(util::allocate_poly_array(targetP::k_, poly_modulus_degree, coeff_modulus_size, MemoryManager::GetPool()));
    // cout << "secret_key" << endl;
    // for (size_t i = 0; i < poly_modulus_degree * targetP::k_ * coeff_modulus_size; i++) {
    //     cout << secret_key.data()[i] << "\t";
    //     if (i % (poly_modulus_degree * targetP::k_) == poly_modulus_degree * targetP::k_ - 1) cout << endl;
    // }
    // cout << endl;
    for (size_t k = 0; k < targetP::k_; k++) {
        for (size_t i = 0; i < coeff_modulus_size; i++)
            for (size_t j = 0; j < poly_modulus_degree; j++)
                secret_key_array[k * (poly_modulus_degree * coeff_modulus_size) + i * poly_modulus_degree + j] =
                    secret_key.data()[i * (poly_modulus_degree * targetP::k_) + k * poly_modulus_degree + j];
    }
    // for (size_t m = 0; m < coeff_modulus_size; m++) {
    //     cout << "secret_key_array" << endl;
    //     for (size_t k = 0; k < targetP::k_; k++)
    //         for (size_t i = 0; i < poly_modulus_degree; i++)
    //             cout << secret_key_array[k * coeff_modulus_size * poly_modulus_degree + m * poly_modulus_degree + i] << "\t";
    //     cout << endl;
    // }
    for (size_t k = 0; k < targetP::k_; k++) {
        util::ntt_negacyclic_harvey(util::RNSIter(secret_key_array.get() + k * (poly_modulus_degree * coeff_modulus_size), poly_modulus_degree), coeff_modulus_size, ntt_tables);
    }
    return secret_key_array;
}
void GLWECipher::NTT(GLWE &cipher)
{
    if (!cipher.is_ntt_form())
        util::ntt_negacyclic_harvey(cipher, cipher.size(), context_.first_context_data()->small_ntt_tables());
    cipher.is_ntt_form() = true;
}
void GLWECipher::INTT(GLWE &cipher)
{
    if (cipher.is_ntt_form())
        util::inverse_ntt_negacyclic_harvey(cipher, cipher.size(), context_.first_context_data()->small_ntt_tables());
    cipher.is_ntt_form() = false;
}
void GLWECipher::encrypt_zero(GLWE &cipher)
{
    util::encrypt_zero_symmetric(secret_key_array_, secret_key_array_size, context_, context_.first_parms_id(), true, cipher);
}
void GLWECipher::encrypt(Plaintext &plain, GLWE &cipher)
{
    encrypt_zero(cipher);
    INTT(cipher);
    util::multiply_add_plain_with_scaling_variant(plain, *context_.first_context_data(), *util::iter(cipher));
    NTT(cipher);
}
inline uint64_t GadgetRed(uint64_t plain, size_t idx, const seal::Modulus &modulus)
{
    size_t Qlen = targetP::digits - (idx + 1) * targetP::Bgbit;
    size_t Wlen = 63;
    uint64_t res = util::barrett_reduce_64(plain << (Qlen % Wlen), modulus);
    for (size_t i = 0; i < Qlen / Wlen; i++) {
        res = util::multiply_uint_mod(res, (1ULL << Wlen), modulus);
    }
    return res;
}
void GLWECipher::gadget_gen(void)
{
#ifdef TRANSPOSE_ON
    size_t rlwe_count = (targetP::k_ + 1);
    auto &modulus = parms.coeff_modulus();
    for (size_t i = 0; i < rlwe_count; i++) {
        GadgetMat[i].resize(context_, (targetP::k_ + 1) * targetP::l_);
    }
    for (size_t k = 0; k <= targetP::k_; k++) {
        for (size_t i = 0; i < targetP::l_; i++) {
            for (size_t j = 0; j < poly_modulus_degree; j++) {
                for (size_t m = 0; m < coeff_modulus_size; m++) {
                    uint64_t red = GadgetRed(1, i, modulus[m]);
                    GadgetMat[k].data(i + k * targetP::l_)[m * poly_modulus_degree + j] =
                        util::add_uint_mod(GadgetMat[k].data(i + k * targetP::l_)[m * poly_modulus_degree + j], red, modulus[m]);
                }
            }
        }
    }
#else
    size_t rlwe_count = (targetP::k_ + 1) * targetP::l_;
    auto &modulus = parms.coeff_modulus();
#ifdef ARRAY_ALLOCATE
    for (size_t i = 0; i < rlwe_count; i++) {
        GadgetMat[i].resize(context_, (targetP::k_ + 1));
    }
#else
    GadgetMat.resize(GGSW_size);
    for (size_t i = 0; i < rlwe_count; i++) {
        GadgetMat[i].resize(context_, (targetP::k_ + 1));
    }
#endif
    for (size_t k = 0; k <= targetP::k_; k++) {
        for (size_t i = 0; i < targetP::l_; i++) {
            for (size_t j = 0; j < poly_modulus_degree; j++) {
                for (size_t m = 0; m < coeff_modulus_size; m++) {
                    uint64_t red = GadgetRed(1, i, modulus[m]);
                    GadgetMat[i + k * targetP::l_].data(k)[m * poly_modulus_degree + j] = red;
                }
            }
        }
    }
#endif
}
void GLWECipher::encrypt(uint64_t plain, GGSW &cipher)
{
#ifdef TRANSPOSE_ON
    size_t rlwe_count = (targetP::k_ + 1);
    auto &modulus = parms.coeff_modulus();
    for (size_t i = 0; i < rlwe_count; i++) {
        cipher[i].resize(context_, (targetP::k_ + 1) * targetP::l_);
    }
    GLWE tmp(context_);
    for (size_t i = 0; i < (targetP::k_ + 1) * targetP::l_; i++) {
        encrypt_zero(tmp);
        for (size_t k = 0; k <= targetP::k_; k++)
            util::set_poly(cipher[k].data(i), poly_modulus_degree, coeff_modulus_size, tmp.data(k));
    }
    for (size_t k = 0; k <= targetP::k_; k++) {
        for (size_t i = 0; i < targetP::l_; i++) {
            for (size_t j = 0; j < poly_modulus_degree; j++) {
                for (size_t m = 0; m < coeff_modulus_size; m++) {
                    uint64_t red = GadgetRed(plain, i, modulus[m]);
                    cipher[k].data(i + k * targetP::l_)[m * poly_modulus_degree + j] =
                        util::add_uint_mod(cipher[k].data(i + k * targetP::l_)[m * poly_modulus_degree + j], red, modulus[m]);
                }
            }
        }
    }
#else
#ifdef ARRAY_ALLOCATE
#else
    cipher.resize(GGSW_size);
#endif
    size_t rlwe_count = (targetP::k_ + 1) * targetP::l_;
    auto &modulus = parms.coeff_modulus();
    for (size_t i = 0; i < rlwe_count; i++) {
        cipher[i].resize(context_, (targetP::k_ + 1));
    }
    for (size_t i = 0; i < (targetP::k_ + 1) * targetP::l_; i++)
        encrypt_zero(cipher[i]);
    for (size_t k = 0; k <= targetP::k_; k++) {
        for (size_t i = 0; i < targetP::l_; i++) {
            for (size_t j = 0; j < poly_modulus_degree; j++) {
                for (size_t m = 0; m < coeff_modulus_size; m++) {
                    uint64_t red = GadgetRed(plain, i, modulus[m]);
                    cipher[i + k * targetP::l_].data(k)[m * poly_modulus_degree + j] =
                        util::add_uint_mod(cipher[i + k * targetP::l_].data(k)[m * poly_modulus_degree + j], red, modulus[m]);
                }
            }
        }
    }
#endif
}
void GLWECipher::encrypt_testvec(GLWE &cipher)
{
    cipher.resize(context_, targetP::k_ + 1);
    Plaintext testvec(poly_modulus_degree);
    for (size_t i = 0; 2 * i < poly_modulus_degree; i++) {
        testvec.data()[i] = 0;
        testvec.data()[i + (poly_modulus_degree >> 1)] = 1;
    }
    util::multiply_add_plain_with_scaling_variant(testvec, *context_.first_context_data(), *util::iter(cipher));
    cipher.is_ntt_form() = false;
}
// void GLWECipher::decrypt(Plaintext &plain, GLWE &cipher)
// {
//     if (cipher.is_ntt_form()) {
//         INTT(cipher);
//         decryptor_.decrypt(cipher, plain);
//         NTT(cipher);
//     } else
//         decryptor_.decrypt(cipher, plain);
// }
// int GLWECipher::NoiseBudget(GLWE &cipher)
// {
//     if (cipher.is_ntt_form()) {
//         INTT(cipher);
//         int budget = decryptor_.invariant_noise_budget(cipher);
//         NTT(cipher);
//         return budget;
//     } else
//         return decryptor_.invariant_noise_budget(cipher);
// }
void GLWECipher::GenGBSK(SecretKey &sk, GBSK &bsk)
{
    if (sk.data().is_ntt_form()) {
        cout << "SK is in NTT form!" << endl;
        exit(0);
    }
    size_t keysize = domainP::n_;
#ifdef ARRAY_ALLOCATE
#else
    bsk.resize(keysize);
#endif
    for (size_t i = 0; i < keysize; i++)
        encrypt(sk.data()[i], bsk[i]);
}
void GLWECipher::GenGBSK_BKU(SecretKey &sk, GBSK_BKU &bsk_bku)
{
#ifdef ARRAY_ALLOCATE
#else
    bsk_bku.resize(domainP::ksize_);
#endif
    for (size_t i = 0; i < (domainP::n_ / domainP::m_); i++) {
        for (size_t j = 1; j <= domainP::mmask_; j++) {
            size_t pt = 1;
            for (size_t k = 0; k < domainP::m_; k++) {
                if ((j >> (domainP::m_ - 1 - k)) & 1)
                    pt *= sk.data()[i * domainP::m_ + k];
                else
                    pt *= (1 - sk.data()[i * domainP::m_ + k]);
            }
            encrypt(pt, bsk_bku[i * (domainP::mmask_) + j - 1]);
        }
    }
    for (size_t i = 0; i < domainP::n_ % domainP::m_; i++)
        encrypt(sk.data()[i + static_cast<size_t>(domainP::n_ / domainP::m_) * domainP::m_], bsk_bku[static_cast<size_t>(domainP::n_ / domainP::m_) * (domainP::mmask_) + i]);
}
void GLWECipher::RotateGGSW(GGSW &rotkey, GGSW &key, const size_t rotidx)
{
    auto &coeff_modulus = parms.coeff_modulus();
    size_t idx = rotidx % poly_modulus_degree;
    bool sgn = rotidx >= poly_modulus_degree;
    GLWE rotpoly;
    rotpoly.resize(context_, 2);
    for (size_t i = 0; i < coeff_modulus_size; i++) {
        rotpoly.data(0)[poly_modulus_degree * i + idx] = (sgn ? coeff_modulus[i].value() - 1 : 1);
        if (idx == 0)
            rotpoly.data(0)[poly_modulus_degree * i] -= 1;
        else
            rotpoly.data(0)[poly_modulus_degree * i] = coeff_modulus[i].value() - 1;
    }
    util::ntt_negacyclic_harvey(util::RNSIter(rotpoly.data(0), poly_modulus_degree), coeff_modulus_size, context_.first_context_data()->small_ntt_tables());
    for (size_t i = 0; i < GGSW_size; i++)
        for (size_t j = 0; j < Ciphertext_size; j++)
            util::dyadic_product_coeffmod(util::RNSIter(rotpoly.data(0), poly_modulus_degree), util::RNSIter(key[i].data(j), poly_modulus_degree), coeff_modulus_size, parms.coeff_modulus(), util::RNSIter(rotkey[i].data(j), poly_modulus_degree));
}

void GLWECipher::RotateGBSK(GGSW &rotkey, GBSK &bsk, LWE &lwe, const size_t idx)
{

    for (size_t i = 0; i < GGSW_size; i++)
        rotkey[i].resize(context_, Ciphertext_size);
    RotateGGSW(rotkey, bsk[idx], lwe.cipher[idx]);
    for (size_t i = 0; i < GGSW_size; i++)
        util::add_poly_coeffmod(rotkey[i], GadgetMat[i], Ciphertext_size, parms.coeff_modulus(), rotkey[i]);
}
void GLWECipher::RotateGBSK_BKU(GGSW &rotkey, GBSK_BKU &bsk_bku, LWE &lwe, const size_t idx)
{
    GGSW tmp;
#ifdef ARRAY_ALLOCATE
#else
    tmp.resize(GGSW_size);
#endif
    for (size_t i = 0; i < GGSW_size; i++) {
        rotkey[i].resize(context_, Ciphertext_size);
        tmp[i].resize(context_, Ciphertext_size);
    }
    if (idx < domainP::n_ / domainP::m_) {
        for (size_t i = 1; i <= domainP::mmask_; i++) {
            size_t rotidx = 0;
            for (size_t k = 0; k < domainP::m_; k++)
                if ((i >> (domainP::m_ - 1 - k)) & 1)
                    rotidx += lwe.cipher[idx * domainP::m_ + k];
            RotateGGSW(tmp, bsk_bku[idx * (domainP::mmask_) + i - 1], rotidx % (2 * poly_modulus_degree));
            for (size_t j = 0; j < GGSW_size; j++)
                util::add_poly_coeffmod(rotkey[j], tmp[j], Ciphertext_size, parms.coeff_modulus(), rotkey[j]);
        }
        for (size_t j = 0; j < GGSW_size; j++)
            util::add_poly_coeffmod(rotkey[j], GadgetMat[j], Ciphertext_size, parms.coeff_modulus(), rotkey[j]);
    } else {
        size_t bkuidx = domainP::n_ / domainP::m_;
        RotateGGSW(tmp, bsk_bku[bkuidx * domainP::mmask_ + idx - bkuidx], lwe.cipher[bkuidx * domainP::m_ + idx - bkuidx]);
        for (size_t j = 0; j < GGSW_size; j++)
            util::add_poly_coeffmod(rotkey[j], tmp[j], Ciphertext_size, parms.coeff_modulus(), rotkey[j]);
        for (size_t j = 0; j < GGSW_size; j++)
            util::add_poly_coeffmod(rotkey[j], GadgetMat[j], Ciphertext_size, parms.coeff_modulus(), rotkey[j]);
    }
#ifdef ARRAY_ALLOCATE
#else
    tmp.clear();
#endif
}
void GLWECipher::BlindRotate_internal(LWE &lwe, GBSK_BKU &bskntt, GLWE &testvector)
{
    auto &coeff_modulus = parms.coeff_modulus();
    auto ntt_tables = context_.first_context_data()->small_ntt_tables();
    GLWE rotated_vector(context_), decntt(context_), product(context_), zerocipher(context_);
    decntt.resize(context_, (targetP::k_ + 1) * targetP::l_);
    product.resize(context_, (targetP::k_ + 1) * targetP::l_);
    rotated_vector.resize(context_, targetP::k_ + 1);
    zerocipher.resize(context_, targetP::k_ + 1);

    util::negacyclic_multiply_poly_mono_coeffmod(
        testvector, targetP::k_ + 1, 1, lwe.cipher[lwe.n], coeff_modulus, rotated_vector, pool_);
    testvector = rotated_vector;
    // double br_{0.};
    // MSecTimer br(&br_, "BlindRotate:");
    Plaintext plain(poly_modulus_degree);
    for (size_t i = 0; i < domainP::bkusize_; i++) {
        GGSW rotkey;
#ifdef ARRAY_ALLOCATE
#else
        rotkey.resize(GGSW_size);
#endif
        for (size_t i = 0; i < GGSW_size; i++)
            rotkey[i].resize(context_, Ciphertext_size);
        RotateGBSK_BKU(rotkey, bskntt, lwe, i);
        rotated_vector = testvector;
        testvector = zerocipher;
        // util::ntt_negacyclic_harvey_lazy(testvector, targetP::k_ + 1, ntt_tables);

        vector<vector<uint64_t>>
            crtdec[targetP::k_ + 1];
        for (size_t k = 0; k < targetP::k_ + 1; k++)
            rns_.CRTDecPoly(rotated_vector.data(k), crtdec[k]);
        for (size_t l = 0; l < targetP::l_; l++)
            for (size_t k = 0; k <= targetP::k_; k++)
                for (size_t m = 0; m < coeff_modulus_size; m++)
                    util::set_poly(crtdec[k][l].data(), poly_modulus_degree, 1, decntt.data(k * targetP::l_ + l) + m * poly_modulus_degree);
        util::ntt_negacyclic_harvey_lazy(decntt, (targetP::k_ + 1) * targetP::l_, ntt_tables);
#ifdef TRANSPOSE_ON
        for (size_t k = 0; k <= targetP::k_; k++) {
            util::dyadic_product_coeffmod(decntt, util::PolyIter(rotkey[k]), (targetP::k_ + 1) * targetP::l_, coeff_modulus, product);
            for (size_t j = 0; j < (targetP::k_ + 1) * targetP::l_; j++)
                util::add_poly_coeffmod(util::RNSIter(testvector.data(k), poly_modulus_degree), util::ConstRNSIter(product.data(j), poly_modulus_degree), coeff_modulus_size, coeff_modulus, util::RNSIter(testvector.data(k), poly_modulus_degree));
        }
#else
        for (size_t k = 0; k < (targetP::k_ + 1) * targetP::l_; k++) {
            for (size_t j = 0; j < targetP::k_ + 1; j++) {
                util::dyadic_product_coeffmod(util::RNSIter(decntt.data(k), poly_modulus_degree), util::RNSIter(rotkey[k].data(j), poly_modulus_degree), coeff_modulus_size, coeff_modulus, util::RNSIter(product.data(k), poly_modulus_degree));
                util::add_poly_coeffmod(util::RNSIter(testvector.data(j), poly_modulus_degree), util::RNSIter(product.data(k), poly_modulus_degree), coeff_modulus_size, coeff_modulus, util::RNSIter(testvector.data(j), poly_modulus_degree));
            }
        }
#endif
#ifdef ARRAY_ALLOCATE
#else
        rotkey.clear();
#endif
        util::inverse_ntt_negacyclic_harvey_lazy(testvector, targetP::k_ + 1, ntt_tables);
    }
    // br.stop();
}
void GLWECipher::SampleExtractIndex(vector<LWE> &lwe, GLWE &rlwe, const size_t index)
{
    lwe.resize(coeff_modulus_size);
    for (size_t i = 0; i < coeff_modulus_size; i++) {
        lwe[i].n = poly_modulus_degree * targetP::k_;
        lwe[i].cipher.resize(lwe[i].n + 1);
        lwe[i].modulus = parms.coeff_modulus()[i].value();
        for (size_t k = 0; k < targetP::k_; k++) {
            lwe[i].cipher[k * poly_modulus_degree] = rlwe.data(k + 1)[i * poly_modulus_degree] % lwe[i].modulus;
            for (size_t j = 1; j < poly_modulus_degree; j++)
                lwe[i].cipher[j + k * poly_modulus_degree] = lwe[i].modulus - rlwe.data(k + 1)[(i + 1) * poly_modulus_degree - j] % lwe[i].modulus;
        }
        lwe[i].cipher[lwe[i].n] = rlwe.data(0)[i * poly_modulus_degree] % lwe[i].modulus;
    }
}
void GLWECipher::GateBootstrappingLWE2LWENTT(vector<LWE> &res, LWE &lwe, GBSK_BKU &bskntt, GLWE &testvector)
{
    LWEScaling(lwe);
    GLWE acc(context_);
    // acc.resize(2);
    BlindRotate_internal(lwe, bskntt, testvector);
    SampleExtractIndex(res, testvector, 0);
}
void GLWECipher::GateBootstrappingLWE2LWENTTwithoutTestVec(vector<LWE> &res, LWE &lwe, GBSK_BKU &bskntt)
{
    LWEScaling(lwe);
    GLWE testvec(context_);
    testvec.resize(context_, targetP::k_ + 1);
    encrypt_testvec(testvec);
    // GLWE acc(context_);
    // acc.resize(2);
    //  BlindRotate(acc, lwe, bskntt, testvec);

    // double br_{0.};
    {
        // MSecTimer br(&br_, "BlindRotate:");
        BlindRotate_internal(lwe, bskntt, testvec);
        SampleExtractIndex(res, testvec, 0);
    }

    // cout << "BlindRotate Done!" << endl;
    //  cout << "Budget: " << NoiseBudget(testvec) << endl;
    // Plaintext plain(poly_modulus_degree);
    // decrypt(plain, testvec);
    // cout << plain.data()[0] << endl;
    // exit(0);
}
void GLWECipher::LWEScaling(LWE &res)
{
    for (size_t i = 0; i <= res.n; i++) {
        res.cipher[i] = static_cast<uint64_t>(2 * poly_modulus_degree - static_cast<__uint128_t>(res.cipher[i]) * poly_modulus_degree * 2 / res.modulus);
    }
}
vector<vector<LWE>> GLWECipher::bootstrap(vector<LWE> lwe_in, SecretKey lwe_sk)
{
    vector<vector<LWE>> lwe_out;
    GBSK_BKU bsk;
    GenGBSK_BKU(lwe_sk, bsk);
    lwe_out.resize(lwe_in.size());

    for (size_t i = 0; i < lwe_in.size(); i++)
        GateBootstrappingLWE2LWENTTwithoutTestVec(lwe_out[i], lwe_in[i], bsk);
    return lwe_out;
}
} // namespace seal
