#include "tfhe.h"
#include "omp.h"
#include "seal.inc"
#include "timer.h"
using namespace std;
namespace seal
{
void TFHECryptor::NTT(RLWECipher &cipher)
{
    if (!cipher.is_ntt_form())
        evaluator_->transform_to_ntt_inplace(cipher);
}
void TFHECryptor::INTT(RLWECipher &cipher)
{
    if (cipher.is_ntt_form())
        evaluator_->transform_from_ntt_inplace(cipher);
}
void TFHECryptor::encrypt_zero(RLWECipher &cipher)
{
    encryptor_->encrypt_zero_symmetric(cipher);
    NTT(cipher);
}
void TFHECryptor::encrypt(Plaintext &plain, RLWECipher &cipher)
{
    encryptor_->encrypt_symmetric(plain, cipher);
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
void TFHECryptor::encrypt(uint64_t plain, RGSWCipher &cipher)
{
    size_t rlwe_count = 2 * targetP::l_;
    auto &modulus = parms->coeff_modulus();
    cipher.reserve(rlwe_count);
    for (size_t i = 0; i < rlwe_count; i++) {
        RLWECipher tmp(*context_);
        encrypt_zero(tmp);
        cipher.emplace_back(move(tmp));
    }
    for (size_t k = 0; k < 2; k++) {
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
}
void TFHECryptor::encrypt_testvec(RLWECipher &cipher)
{
    cipher.resize(2);
    Plaintext testvec(poly_modulus_degree);
    for (size_t i = 0; 2 * i < poly_modulus_degree; i++) {
        testvec.data()[i] = 0;
        testvec.data()[i + (poly_modulus_degree >> 1)] = 1;
    }
    util::multiply_add_plain_with_scaling_variant(testvec, *context_->first_context_data(), *util::iter(cipher));
}
void TFHECryptor::decrypt(Plaintext &plain, RLWECipher &cipher)
{
    if (cipher.is_ntt_form()) {
        INTT(cipher);
        decryptor_->decrypt(cipher, plain);
        NTT(cipher);
    } else
        decryptor_->decrypt(cipher, plain);
}
/*void TFHECryptor::ExternalProduct_internal(util::RNSIter res_iter0, util::RNSIter res_iter1, util::RNSIter decntt_iter, util::RNSIter prod_iter0, util::RNSIter prod_iter1, RLWECipher &src, RGSWCipher operand)
{

    auto &coeff_modulus = parms->coeff_modulus();
    vector<vector<uint64_t>> crtdec[2];
    for (size_t i = 0; i < 2; i++)
        rns_->CRTDecPoly(src.data(i), crtdec[i]);
    for (size_t k = 0; k < 2; k++)
        for (size_t i = 0; i < targetP::l_; i++) {
            for (size_t j = 0; j < poly_modulus_degree; j++)
                for (size_t l = 0; l < coeff_modulus_size; l++)
                    decntt.data(1)[j + l * poly_modulus_degree] = crtdec[k][i][j];
            util::RNSIter data_iter0(operand[i + k * targetP::l_].data(0), poly_modulus_degree);
            util::RNSIter data_iter1(operand[i + k * targetP::l_].data(1), poly_modulus_degree);
            decntt.is_ntt_form() = false;
            product.is_ntt_form() = true;
            util::ntt_negacyclic_harvey_lazy(decntt_iter, coeff_modulus_size, context_->first_context_data()->small_ntt_tables());
            util::dyadic_product_accumulate(decntt_iter, data_iter0, coeff_modulus_size, res_iter0, prod_iter0);
            util::dyadic_product_accumulate(decntt_iter, data_iter1, coeff_modulus_size, res_iter1, prod_iter1);
        }
    util::dyadic_coeffmod(res_iter0, prod_iter0, coeff_modulus_size, coeff_modulus, res_iter0);
    util::dyadic_coeffmod(res_iter1, prod_iter1, coeff_modulus_size, coeff_modulus, res_iter1);
}*/
void TFHECryptor::ExternalProduct(RLWECipher &dst, RLWECipher &src, RGSWCipher operand)
{
    // INTT(src);
    auto &coeff_modulus = parms->coeff_modulus();
    vector<vector<uint64_t>> crtdec[2];
    for (size_t i = 0; i < 2; i++)
        rns_->CRTDecPoly(src.data(i), crtdec[i]);
    RLWECipher decntt(*context_), product(*context_);
    decntt.resize(2);
    product.resize(2);
    // dst.is_ntt_form() = true;
    util::RNSIter res_iter0(dst.data(0), poly_modulus_degree);
    util::RNSIter res_iter1(dst.data(1), poly_modulus_degree);
    util::RNSIter decntt_iter(decntt.data(1), poly_modulus_degree);
    util::RNSIter prod_iter0(product.data(0), poly_modulus_degree);
    util::RNSIter prod_iter1(product.data(1), poly_modulus_degree);
    for (size_t k = 0; k < 2; k++)
        for (size_t i = 0; i < targetP::l_; i++) {
            for (size_t j = 0; j < poly_modulus_degree; j++)
                for (size_t l = 0; l < coeff_modulus_size; l++)
                    decntt.data(1)[j + l * poly_modulus_degree] = crtdec[k][i][j];
            util::RNSIter data_iter0(operand[i + k * targetP::l_].data(0), poly_modulus_degree);
            util::RNSIter data_iter1(operand[i + k * targetP::l_].data(1), poly_modulus_degree);
            util::ntt_negacyclic_harvey_lazy(decntt_iter, coeff_modulus_size, context_->first_context_data()->small_ntt_tables());
            // util::dyadic_product_accumulate(decntt_iter, data_iter0, coeff_modulus_size, res_iter0, prod_iter0);
            // util::dyadic_product_accumulate(decntt_iter, data_iter1, coeff_modulus_size, res_iter1, prod_iter1);
            util::dyadic_product_coeffmod(decntt_iter, data_iter0, coeff_modulus_size, coeff_modulus, prod_iter0);
            util::dyadic_product_coeffmod(decntt_iter, data_iter1, coeff_modulus_size, coeff_modulus, prod_iter1);
            util::add_poly_coeffmod(res_iter0, prod_iter0, coeff_modulus_size, coeff_modulus, res_iter0);
            util::add_poly_coeffmod(res_iter1, prod_iter1, coeff_modulus_size, coeff_modulus, res_iter1);

            /*for (int m = 0; m < 2; m++)
            {
                util::RNSIter res_iter(dst.data(m), poly_modulus_degree);
                util::RNSIter decntt_iter(decntt.data(1), poly_modulus_degree);
                util::RNSIter prod_iter(product.data(1), poly_modulus_degree);
                util::RNSIter data_iter(operand[i + k * targetP::l_].data(m), poly_modulus_degree);
                util::dyadic_product_coeffmod(decntt_iter, data_iter, coeff_modulus_size, coeff_modulus, prod_iter);
                util::add_poly_coeffmod(res_iter, prod_iter, coeff_modulus_size, coeff_modulus, res_iter);
            }*/
        }
    // util::dyadic_coeffmod(res_iter0, prod_iter0, coeff_modulus_size, coeff_modulus, res_iter0);
    // util::dyadic_coeffmod(res_iter1, prod_iter1, coeff_modulus_size, coeff_modulus, res_iter1);
}
int TFHECryptor::NoiseBudget(RLWECipher &cipher)
{
    if (cipher.is_ntt_form()) {
        INTT(cipher);
        int budget = decryptor_->invariant_noise_budget(cipher);
        NTT(cipher);
        return budget;
    } else
        return decryptor_->invariant_noise_budget(cipher);
}
void TFHECryptor::GenBSK(SecretKey &sk, BSK &bsk)
{
    if (sk.data().is_ntt_form()) {
        cout << "SK is in NTT form!" << endl;
        exit(0);
    }
    size_t keysize = domainP::ksize_;
    bsk.reserve(keysize);
    for (size_t i = 0; i < domainP::n_ / domainP::m_; i++) {
        for (size_t j = 0; j < (1 << domainP::m_) - 1; j++) {
            RGSWCipher rgsw;
            size_t pt = 1;
            for (size_t k = 0; k < domainP::m_; k++) {
                if ((j >> (domainP::m_ - 1 - k)) & 1)
                    pt *= (1 - sk.data()[i * domainP::m_ + k]);
                else
                    pt *= sk.data()[i * domainP::m_ + k];
            }
            encrypt(pt, rgsw);
            bsk.emplace_back(move(rgsw));
        }
    }
    for (size_t i = (domainP::n_ / domainP::m_) * domainP::m_; i < domainP::n_; i++) {
        RGSWCipher rgsw;
        encrypt(sk.data()[i], rgsw);
        bsk.emplace_back(move(rgsw));
    }
}
void TFHECryptor::ConvertBSKtoBKU(BSK &dst, BSK &src, const LWE &scaled_lwe)
{
    size_t keysize = domainP::bkusize_;
    dst.reserve(keysize);
    Plaintext plain(targetP::n_);
    for (size_t i = 0; i < domainP::n_ / domainP::m_; i++) {
        RGSWCipher bku;
        bku.reserve((targetP::k_ + 1) * targetP::l_);
        for (size_t j = 0; j < (targetP::k_ + 1) * targetP::l_; j++) {
            RLWECipher ct(*context_);
            ct.resize(targetP::k_ + 1);
            bku.emplace_back(move(ct));
        }
        for (size_t j = 0; j < (1 << domainP::m_) - 1; j++) {
            RGSWCipher rgsw;
            encrypt(0, rgsw);
            size_t sft = 0;
            for (size_t u = 0; u < domainP::m_; u++)
                if ((j >> (domainP::m_ - 1 - u)) & 1) sft += scaled_lwe.cipher[i * domainP::m_ + j];
            sft %= (targetP::n_ * 2);
            plain.data()[sft] = 1;
            for (size_t k = 0; k < (targetP::k_ + 1) * targetP::l_; k++) {
                evaluator_->multiply_plain(src[i * domainP::m_ + j][k], plain, rgsw[k]);
                evaluator_->sub_inplace(rgsw[k], src[i * domainP::m_ + j][k]);
                evaluator_->add_inplace(bku[k], rgsw[k]);
            }
            plain.data()[sft] = 0;
        }
        dst.emplace_back(move(bku));
    }
    for (size_t i = 0; i < domainP::n_ % domainP::m_; i++) {
        RGSWCipher rgsw;
        encrypt(0, rgsw);
        size_t sft = scaled_lwe.cipher[i + (domainP::n_ / domainP::m_) * domainP::m_];
        plain.data()[sft] = 1;
        for (size_t k = 0; k < (targetP::k_ + 1) * targetP::l_; k++) {
            evaluator_->multiply_plain(src[i + (domainP::n_ / domainP::m_) * ((1 << domainP::m_) - 1)][k], plain, rgsw[k]);
            evaluator_->sub_inplace(rgsw[k], src[i + (domainP::n_ / domainP::m_) * ((1 << domainP::m_) - 1)][k]);
        }
        plain.data()[sft] = 0;
        dst.emplace_back(rgsw);
    }
}
void TFHECryptor::CMUXNTTwithPolynomialMulByXaiMinusOne(RLWECipher &acc, RGSWCipher &cs, const uint64_t a)
{
    auto &coeff_modulus = parms->coeff_modulus();

    RLWECipher temp(*context_);
    RLWECipher res(*context_);
    temp.resize(2);
    res.resize(2);
    temp.is_ntt_form() = false;
    res.is_ntt_form() = true;
    INTT(acc);
    PolynomialMulByXaiMinusOne(temp, acc, a);
    ExternalProduct(res, temp, cs);
    NTT(acc);
    for (size_t k = 0; k < 2; k++) {
        util::RNSIter acc_iter(acc.data(k), poly_modulus_degree);
        util::ConstRNSIter res_iter(res.data(k), poly_modulus_degree);
        util::add_poly_coeffmod(acc_iter, res_iter, coeff_modulus_size, coeff_modulus, acc_iter);
    }
}
/*void TFHECryptor::CMUXNTTwithPolynomialMulByXaiMinusOne_internal(util::RNSIter res_iter0, util::RNSIter res_iter1, util::RNSIter decntt_iter, util::RNSIter prod_iter0, util::RNSIter prod_iter1, util::RNSIter acc_iter0, util::RNSIter acc_iter1, util::RNSIter sft_iter0, util::RNSIter sft_iter1, RLWECipher &sft, RGSWCipher &cs, const uint64_t a)
{
    auto &coeff_modulus = parms->coeff_modulus();
    PolynomialMulByXaiMinusOne_internal(sft_iter0, sft_iter1, acc_iter0, acc_iter1, a);
    ExternalProduct_internel(res_iter0, res_iter1, decntt_iter, prod_iter0, prod_iter1, sft, cs);
    util::add_poly_coeffmod(acc_iter0, res_iter0, coeff_modulus_size, coeff_modulus, acc_iter0);
    util::add_poly_coeffmod(acc_iter1, res_iter1, coeff_modulus_size, coeff_modulus, acc_iter1);
}*/
void TFHECryptor::PolynomialMulByXai(RLWECipher &dst, RLWECipher &src, const uint64_t a)
{
    auto &coeff_modulus = parms->coeff_modulus();
    for (size_t k = 0; k < 2; k++) {
        util::ConstRNSIter src_iter(src.data(k), poly_modulus_degree);
        util::RNSIter dst_iter(dst.data(k), poly_modulus_degree);
        util::negacyclic_shift_poly_coeffmod(src_iter, coeff_modulus.size(), a, coeff_modulus, dst_iter);
    }
}
/*void TFHECryptor::PolynomialMulByXai_internal(util::RNSIter dst_iter0, util::RNSIter dst_iter1, util::RNSIter src_iter0, util::RNSIter src_iter1, const uint64_t a)
{
    auto &coeff_modulus = parms->coeff_modulus();
    RLWECipher rotation(*context_);
    rotation.resize(2);
    rotation.is_ntt_form() = false;
    for (size_t k = 0; k < 2; k++)
        for (size_t i = 0; i < coeff_modulus_size; i++)
            rotation.data(k)[a + i * poly_modulus_degree] = 1;
    NTT(rotation);
    util::dyadic_product_coeffmod(src_iter0, util::RNSIter(rotation.data(0), poly_modulus_degree), coeff_modulus_size, coeff_modulus, dst_iter0);
    util::dyadic_product_coeffmod(src_iter1, util::RNSIter(rotation.data(1), poly_modulus_degree), coeff_modulus_size, coeff_modulus, dst_iter1);
}*/
void TFHECryptor::PolynomialMulByXaiMinusOne(RLWECipher &dst, RLWECipher &src, const uint64_t a)
{
    auto &coeff_modulus = parms->coeff_modulus();
    // INTT(dst);
    for (size_t k = 0; k < 2; k++) {
        util::ConstRNSIter src_iter(src.data(k), poly_modulus_degree);
        util::RNSIter dst_iter(dst.data(k), poly_modulus_degree);
        util::negacyclic_shift_poly_coeffmod(src_iter, coeff_modulus_size, a, coeff_modulus, dst_iter);
        util::sub_poly_coeffmod(dst_iter, src_iter, coeff_modulus_size, coeff_modulus, dst_iter);
    }
}
/*void TFHECryptor::PolynomialMulByXaiMinusOne_internal(util::RNSIter dst_iter0, util::RNSIter dst_iter1, util::RNSIter src_iter0, util::RNSIter src_iter1, const uint64_t a)
{
    auto &coeff_modulus = parms->coeff_modulus();
    RLWECipher rotation(*context_);
    rotation.resize(2);
    rotation.is_ntt_form() = false;
    for (size_t k = 0; k < 2; k++)
        for (size_t i = 0; i < coeff_modulus_size; i++)
            rotation.data(k)[a + i * poly_modulus_degree] = 1;
    NTT(rotation);
    util::dyadic_rotate(src_iter0, util::RNSIter(rotation.data(0), poly_modulus_degree), coeff_modulus_size, coeff_modulus, dst_iter0);
    util::dyadic_rotate(src_iter1, util::RNSIter(rotation.data(1), poly_modulus_degree), coeff_modulus_size, coeff_modulus, dst_iter1);
}*/

void TFHECryptor::BlindRotate(RLWECipher &res, const LWE &lwe, BSK &bskntt, RLWECipher &testvector)
{
    PolynomialMulByXai(res, testvector, poly_modulus_degree * 2 - lwe.cipher[lwe.n]);
    for (size_t i = 0; i < lwe.n; i++) {
        uint64_t a = lwe.cipher[i];
        if (a == 0)
            continue;
        CMUXNTTwithPolynomialMulByXaiMinusOne(res, bskntt[i], a);
    }
}
void TFHECryptor::BlindRotate_internal(RLWECipher &res, const LWE &lwe, BSK &bskntt, RLWECipher &testvector)
{
    auto &coeff_modulus = parms->coeff_modulus();
    util::RNSIter testvector_iter[2] = {util::RNSIter(testvector.data(0), poly_modulus_degree), util::RNSIter(testvector.data(1), poly_modulus_degree)};
    RLWECipher rotated_vector(*context_), decntt(*context_), product(*context_);
    decntt.resize(2);
    product.resize(2);
    rotated_vector.resize(2);
    util::RNSIter rotated_vector_iter[2] = {util::RNSIter(rotated_vector.data(0), poly_modulus_degree), util::RNSIter(rotated_vector.data(1), poly_modulus_degree)};
    util::RNSIter decntt_iter(decntt.data(0), poly_modulus_degree);
    util::RNSIter product_iter[2] = {util::RNSIter(product.data(0), poly_modulus_degree), util::RNSIter(product.data(1), poly_modulus_degree)};

    /*
     PolynomialMulByXai(res, testvector, poly_modulus_degree * 2 - lwe.cipher[lwe.n]);
     In External Product, ACC as input in INTT form, hence mulbyxai/mulbyxaiminusone are done in INTT form
     mulbyxai is done by negacyclic_shift_poly_coeffmod

     @param[in] testvector
     @param[out] testvector
    */
    for (size_t k = 0; k < 2; k++)
        util::negacyclic_shift_poly_coeffmod(testvector_iter[k], coeff_modulus_size, 2 * poly_modulus_degree - lwe.cipher[lwe.n], coeff_modulus, testvector_iter[k]);

    /*
     for loop of CMUXNTTwithPolynomialMulByXaiMinusOne
     Start with testvector in INTT, hence at the end of each rotation, testvector should be converted to INTT form
    */
    for (size_t i = 0; i < lwe.n; i++) {
        if (lwe.cipher[i] == 0) continue;
        /*
         CMUXNTTwithPolynomialMulByXaiMinusOne(testvector, bskntt[i], lwe.cipher[i]);
         PolynomialMulByXaiMinusOne is first performed to testvector(INTT), output in rotated_vector(INTT)
         testvector converted to NTT form
         ExternalProduct is then performed to rotated_vector(INTT) & bskntt[i](NTT), output added to testvector(NTT)
         Lazy reduction is performed
         ntt_lazy reduce the result in [0,4q), it is OK for accumulation in daydic_product_accumulate, which maintains the result in two uint64s

         @param[in] bskntt[i]
         @param[in/out] testvector
        */

        // PolynomialMulByXaiMinusOne
        for (size_t k = 0; k < 2; k++) {
            util::negacyclic_shift_poly_coeffmod(testvector_iter[k], coeff_modulus_size, lwe.cipher[i], coeff_modulus, rotated_vector_iter[k]);
            util::sub_poly_coeffmod(rotated_vector_iter[k], testvector_iter[k], coeff_modulus_size, coeff_modulus, rotated_vector_iter[k]);
        }
        // util::rotate_minus_one_lazy(testvector_iter[k], coeff_modulus_size, lwe.cipher[i], coeff_modulus, rotated_vector_iter[k]);

        // NTT of testvector
        NTT(testvector);

        /*
         ExternalProduct(testvector, rotated_vector, bskntt[i])
         Here External Product is actually external product and accumulation of testvector

         @param[in] bskntt[i](NTT)/rotated_vector(INTT)
         @param[in/out] testvector(NTT)
        */

        vector<vector<uint64_t>> crtdec[2];
        for (size_t k = 0; k < 2; k++)
            rns_->CRTDecPoly(rotated_vector.data(k), crtdec[k]);
        for (size_t k = 0; k < 2; k++)
            for (size_t m = 0; m < targetP::l_; m++) {
                for (size_t j = 0; j < poly_modulus_degree; j++)
                    for (size_t l = 0; l < coeff_modulus_size; l++)
                        decntt.data(0)[j + l * poly_modulus_degree] = crtdec[k][m][j];
                util::RNSIter data_iter[2] = {util::RNSIter(bskntt[i][m + k * targetP::l_].data(0), poly_modulus_degree), util::RNSIter(bskntt[i][m + k * targetP::l_].data(1), poly_modulus_degree)};
                util::ntt_negacyclic_harvey_lazy(decntt_iter, coeff_modulus_size, context_->first_context_data()->small_ntt_tables());
                for (size_t kk = 0; kk < 2; kk++) {
                    util::dyadic_product_coeffmod(decntt_iter, data_iter[kk], coeff_modulus_size, coeff_modulus, product_iter[kk]);
                    util::add_poly_coeffmod(testvector_iter[kk], product_iter[kk], coeff_modulus_size, coeff_modulus, testvector_iter[kk]);
                }
                // util::dyadic_product_accumulate(decntt_iter, data_iter[kk], coeff_modulus_size, testvector_iter[kk], product_iter[kk]);
            }
        // for (size_t k = 0; k < 2; k++)
        //     util::dyadic_coeffmod(testvector_iter[k], product_iter[k], coeff_modulus_size, coeff_modulus, testvector_iter[k]);
        INTT(testvector);
    }
}
/**
 * @brief
 *
 * @param res
 * @param lwe
 * @param bskntt
 * @param testvector
 */
void TFHECryptor::BlindRotate_internal_withBKU(RLWECipher &res, const LWE &lwe, BSK &bskntt, RLWECipher &testvector)
{
    auto &coeff_modulus = parms->coeff_modulus();
    util::RNSIter testvector_iter[2] = {util::RNSIter(testvector.data(0), poly_modulus_degree), util::RNSIter(testvector.data(1), poly_modulus_degree)};
    RLWECipher rotated_vector(*context_), decntt(*context_), product(*context_);
    decntt.resize(2);
    product.resize(2);
    rotated_vector.resize(2);
    util::RNSIter rotated_vector_iter[2] = {util::RNSIter(rotated_vector.data(0), poly_modulus_degree), util::RNSIter(rotated_vector.data(1), poly_modulus_degree)};
    util::RNSIter decntt_iter(decntt.data(0), poly_modulus_degree);
    util::RNSIter product_iter[2] = {util::RNSIter(product.data(0), poly_modulus_degree), util::RNSIter(product.data(1), poly_modulus_degree)};

    BSK bku;
    ConvertBSKtoBKU(bku, bskntt, lwe);
    /*
     PolynomialMulByXai(res, testvector, poly_modulus_degree * 2 - lwe.cipher[lwe.n]);
     In External Product, ACC as input in INTT form, hence mulbyxai/mulbyxaiminusone are done in INTT form
     mulbyxai is done by negacyclic_shift_poly_coeffmod

     @param[in] testvector
     @param[out] testvector
    */
    for (size_t k = 0; k < 2; k++)
        util::negacyclic_shift_poly_coeffmod(testvector_iter[k], coeff_modulus_size, 2 * poly_modulus_degree - lwe.cipher[lwe.n], coeff_modulus, testvector_iter[k]);

    /*
     for loop of CMUXNTTwithPolynomialMulByXaiMinusOne
     Start with testvector in INTT, hence at the end of each rotation, testvector should be converted to INTT form
    */
    for (size_t i = 0; i < (lwe.n / domainP::m_) + (lwe.n % domainP::m_); i++) {
        /*
         CMUXNTTwithPolynomialMulByXaiMinusOne(testvector, bskntt[i], lwe.cipher[i]);
         PolynomialMulByXaiMinusOne is first performed to testvector(INTT), output in rotated_vector(INTT)
         testvector converted to NTT form
         ExternalProduct is then performed to rotated_vector(INTT) & bskntt[i](NTT), output added to testvector(NTT)
         Lazy reduction is performed
         ntt_lazy reduce the result in [0,4q), it is OK for accumulation in daydic_product_accumulate, which maintains the result in two uint64s

         @param[in] bskntt[i]
         @param[in/out] testvector
        */

        // PolynomialMulByXaiMinusOne
        rotated_vector = testvector;
        // util::rotate_minus_one_lazy(testvector_iter[k], coeff_modulus_size, lwe.cipher[i], coeff_modulus, rotated_vector_iter[k]);

        // NTT of testvector
        NTT(testvector);

        /*
         ExternalProduct(testvector, rotated_vector, bskntt[i])
         Here External Product is actually external product and accumulation of testvector

         @param[in] bskntt[i](NTT)/rotated_vector(INTT)
         @param[in/out] testvector(NTT)
        */

        vector<vector<uint64_t>> crtdec[2];
        for (size_t k = 0; k < 2; k++)
            rns_->CRTDecPoly(rotated_vector.data(k), crtdec[k]);
        for (size_t k = 0; k < 2; k++)
            for (size_t m = 0; m < targetP::l_; m++) {
                for (size_t j = 0; j < poly_modulus_degree; j++)
                    for (size_t l = 0; l < coeff_modulus_size; l++)
                        decntt.data(0)[j + l * poly_modulus_degree] = crtdec[k][m][j];
                util::RNSIter data_iter[2] = {util::RNSIter(bskntt[i][m + k * targetP::l_].data(0), poly_modulus_degree), util::RNSIter(bskntt[i][m + k * targetP::l_].data(1), poly_modulus_degree)};
                util::ntt_negacyclic_harvey_lazy(decntt_iter, coeff_modulus_size, context_->first_context_data()->small_ntt_tables());
                for (size_t kk = 0; kk < 2; kk++) {
                    util::dyadic_product_coeffmod(decntt_iter, data_iter[kk], coeff_modulus_size, coeff_modulus, product_iter[kk]);
                    util::add_poly_coeffmod(testvector_iter[kk], product_iter[kk], coeff_modulus_size, coeff_modulus, testvector_iter[kk]);
                }
                // util::dyadic_product_accumulate(decntt_iter, data_iter[kk], coeff_modulus_size, testvector_iter[kk], product_iter[kk]);
            }
        // for (size_t k = 0; k < 2; k++)
        //     util::dyadic_coeffmod(testvector_iter[k], product_iter[k], coeff_modulus_size, coeff_modulus, testvector_iter[k]);
        INTT(testvector);
    }
}
void TFHECryptor::SampleExtractIndex(vector<LWE> &lwe, RLWECipher &rlwe, const size_t index)
{
    lwe.resize(coeff_modulus_size);
    for (size_t i = 0; i < coeff_modulus_size; i++) {
        lwe[i].n = poly_modulus_degree;
        lwe[i].cipher.resize(poly_modulus_degree + 1);
        lwe[i].cipher[0] = rlwe.data(1)[0];
        lwe[i].modulus = parms->coeff_modulus()[i].value();
        for (size_t j = 1; j < poly_modulus_degree; j++)
            lwe[i].cipher[j] = lwe[i].modulus - rlwe.data(1)[(i + 1) * poly_modulus_degree - j];
        lwe[i].cipher[poly_modulus_degree] = rlwe.data(0)[i * poly_modulus_degree];
    }
}
void TFHECryptor::GateBootstrappingLWE2LWENTT(vector<LWE> &res, LWE &lwe, BSK &bskntt, RLWECipher &testvector)
{
    LWEScaling(lwe);
    RLWECipher acc(*context_);
    acc.resize(2);
    BlindRotate(acc, lwe, bskntt, testvector);
    SampleExtractIndex(res, acc, 0);
}
void TFHECryptor::GateBootstrappingLWE2LWENTTwithoutTestVec(vector<LWE> &res, LWE &lwe, BSK &bskntt)
{
    LWEScaling(lwe);
    RLWECipher testvec(*context_);
    testvec.resize(2);
    encrypt_testvec(testvec);
    RLWECipher acc(*context_);
    acc.resize(2);
    acc.is_ntt_form() = false;
    testvec.is_ntt_form() = false;
    // BlindRotate(acc, lwe, bskntt, testvec);
    // double br_{0.};
    // MSecTimer br(&br_, "BlindRotate:");
    BlindRotate_internal(acc, lwe, bskntt, testvec);
    // br.stop();
    cout << "Budget: " << NoiseBudget(testvec) << endl;
    Plaintext plain(poly_modulus_degree);
    decrypt(plain, testvec);
    cout << plain.data()[0] << endl;
    SampleExtractIndex(res, testvec, 0);
}
void TFHECryptor::LWEScaling(LWE &res)
{
    for (size_t i = 0; i <= res.n; i++) {
        res.cipher[i] = static_cast<uint64_t>(static_cast<__uint128_t>(res.cipher[i]) * poly_modulus_degree * 2 / res.modulus);
    }
}
vector<vector<LWE>> TFHECryptor::bootstrap(vector<LWE> lwe_in, SecretKey lwe_sk)
{
    vector<vector<LWE>> lwe_out;
    BSK bsk;
    GenBSK(lwe_sk, bsk);
    lwe_out.resize(lwe_in.size());
#pragma omp parallel for
    for (size_t i = 0; i < lwe_in.size(); i++)
        GateBootstrappingLWE2LWENTTwithoutTestVec(lwe_out[i], lwe_in[i], bsk);
    return lwe_out;
}
} // namespace seal
