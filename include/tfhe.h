#pragma once
#include "lwe.h"
// #include "db/db.h"
#include "params.h"
#include "rns.h"
#include "seal.inc"
#include <vector>
namespace seal
{
#ifdef TRANSPOSE_ON
static constexpr size_t Ciphertext_size = 2 * targetP::l_;
static constexpr size_t RGSW_size = 2;
#else
static constexpr size_t Ciphertext_size = 2;
static constexpr size_t RGSW_size = 2 * targetP::l_;
#endif
using RLWECipher = Ciphertext;
using RGSWCipher = std::vector<RLWECipher>;
using BSK = std::vector<RGSWCipher>;
class TFHECryptor
{
  public:
    TFHECryptor(void) = default;
    TFHECryptor(SEALContext &context, SecretKey &secret_key)
    {
        context_ = std::make_shared<SEALContext>(context);
        evaluator_ = std::make_shared<Evaluator>(*context_);
        secret_key_ = std::make_shared<SecretKey>(secret_key);
        encryptor_ = std::make_shared<Encryptor>(*context_, *secret_key_);
        decryptor_ = std::make_shared<Decryptor>(*context_, *secret_key_);
        rns_ = std::make_shared<TFHERNS>(*context_);
        parms = std::make_shared<EncryptionParameters>(context_->first_context_data().get()->parms());

        poly_modulus_degree = parms->poly_modulus_degree();
        coeff_modulus_size = parms->coeff_modulus().size();
    }
    const SEALContext &get_context(void)
    {
        return *context_;
    }
    void NTT(RLWECipher &cipher);
    void INTT(RLWECipher &cipher);
    void encrypt_zero(RLWECipher &cipher);
    void encrypt(Plaintext &plain, RLWECipher &cipher);
    void encrypt(uint64_t plain, RGSWCipher &cipher);
    void encrypt_testvec(RLWECipher &cipher);
    void decrypt(Plaintext &plain, RLWECipher &cipher);
    void ExternalProduct(RLWECipher &dst, RLWECipher &src, RGSWCipher operand);
    void ExternalProduct_internal(util::RNSIter res0_iter, util::RNSIter res1_iter, util::RNSIter decntt_iter, util::RNSIter prod0_iter, util::RNSIter prod1_iter, RLWECipher &src, RGSWCipher operand);
    int NoiseBudget(RLWECipher &cipher);
    void BlindRotate(LWE &lwe, RLWECipher &testvec, BSK &bsk);
    void SampleExtract(vector<LWE> &lwe, RLWECipher &RLWECipher);
    void Bootstrap(LWE lwe_in, LWE lwe_out, BSK &bsk);
    vector<vector<LWE>> bootstrap(vector<LWE> lwe_in, SecretKey lwe_sk);
    void GenBSK(SecretKey &sk, BSK &bsk);
    void ConvertBSKtoBKU(BSK &dst, BSK &src, const LWE &scaled_lwe);
    void CMUXNTTwithPolynomialMulByXaiMinusOne(RLWECipher &acc, RGSWCipher &cs, const uint64_t a);
    void CMUXNTTwithPolynomialMulByXaiMinusOne_internal(util::RNSIter res_iter0, util::RNSIter res_iter1, util::RNSIter decntt_iter, util::RNSIter prod_iter0, util::RNSIter prod_iter1, util::RNSIter acc_iter0, util::RNSIter acc_iter1, util::RNSIter sft_iter0, util::RNSIter sft_iter1, RLWECipher &sft, RGSWCipher &cs, const uint64_t a);
    void PolynomialMulByXai(RLWECipher &dst, RLWECipher &src, const uint64_t a);
    void PolynomialMulByXaiMinusOne(RLWECipher &dst, RLWECipher &src, const uint64_t a);
    void PolynomialMulByXaiMinusOne_internal(util::RNSIter dst_iter0, util::RNSIter dst_iter1, util::RNSIter src_iter0, util::RNSIter src_iter1, const uint64_t a);
    void BlindRotate(RLWECipher &res, const LWE &lwe, BSK &bskntt, RLWECipher &testvector);
    void BlindRotate_internal(RLWECipher &res, const LWE &lwe, BSK &bskntt, RLWECipher &testvector);
    void BlindRotate_internal_withBKU(RLWECipher &res, const LWE &lwe, BSK &bskntt, RLWECipher &testvector);
    void SampleExtractIndex(vector<LWE> &lwe, RLWECipher &rlwe, const size_t index);
    void GateBootstrappingLWE2LWENTT(vector<LWE> &res, LWE &lwe, BSK &bskntt, RLWECipher &testvector);
    void GateBootstrappingLWE2LWENTTwithoutTestVec(vector<LWE> &res, LWE &lwe, BSK &bskntt);
    void LWEScaling(LWE &res);

  private:
    std::shared_ptr<Evaluator> evaluator_ = nullptr;
    std::shared_ptr<SEALContext> context_ = nullptr;
    std::shared_ptr<SecretKey> secret_key_ = nullptr;
    std::shared_ptr<Encryptor> encryptor_ = nullptr;
    std::shared_ptr<Decryptor> decryptor_ = nullptr;
    std::shared_ptr<TFHERNS> rns_ = nullptr;
    size_t poly_modulus_degree;
    size_t coeff_modulus_size;
    std::shared_ptr<EncryptionParameters> parms = nullptr;
};
} // namespace seal