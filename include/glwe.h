#pragma once
#include "lwe.h"
// #include "db/db.h"
#include "params.h"
#include "rns.h"
#include "seal.inc"
#include <array>
// #define TRANSPOSE_ON
#define BKU_ON
namespace seal
{
using GLWE = Ciphertext;
#ifdef TRANSPOSE_ON
static constexpr size_t Ciphertext_size = (targetP::k_ + 1) * targetP::l_;
static constexpr size_t GGSW_size = (targetP::k_ + 1);
#else
static constexpr size_t Ciphertext_size = (targetP::k_ + 1);
static constexpr size_t GGSW_size = (targetP::k_ + 1) * targetP::l_;
#endif
#ifdef ARRAY_ALLOCATE
using GGSW = std::array<GLWE, GGSW_size>;
using GBSK = std::array<GGSW, domainP::n_>;
using GBSK_BKU = std::array<GGSW, domainP::ksize_>;
using GBSK_BKB = std::array<GGSW, domainP::bkusize_>;
#else
using GGSW = std::vector<GLWE>;
using GBSK_BKU = std::vector<GGSW>;
using GBSK = std::vector<GGSW>;
#endif
EncryptionParameters ConvertContext(SEALContext &context);
util::Pointer<uint64_t> ConvertSKArr(SEALContext &context, SecretKey &secret_key);
SecretKey ConvertSK(SecretKey &secret_key);
class GLWECipher
{
  public:
    GLWECipher(void) = default;
    GLWECipher(SEALContext &context, SecretKey &secret_key) : context_(ConvertContext(context)), secret_key_array_(ConvertSKArr(context_, secret_key)), secret_key_(secret_key),
                                                              rns_(context_), parms(context_.first_context_data().get()->parms()),
                                                              decryptor_(context_, secret_key, targetP::k_)
    {
        poly_modulus_degree = parms.poly_modulus_degree();
        coeff_modulus_size = parms.coeff_modulus().size();
        secret_key_array_size = targetP::k_;
        gadget_gen();
    }
    SEALContext &get_context(void)
    {
        return context_;
    }
    void NTT(GLWE &cipher);
    void INTT(GLWE &cipher);
    void gadget_gen(void);
    void encrypt_zero(GLWE &cipher);
    void encrypt(Plaintext &plain, GLWE &cipher);
    void encrypt(uint64_t plain, GGSW &cipher);
    void encrypt_testvec(GLWE &cipher);
    void BlindRotate(LWE &lwe, GLWE &testvec, GBSK &bsk);
    void SampleExtract(vector<LWE> &lwe, GLWE &GLWE);
    vector<vector<LWE>> bootstrap(vector<LWE> lwe_in, SecretKey lwe_sk);
    void GenGBSK(SecretKey &sk, GBSK &bsk);
    void GenGBSK_BKU(SecretKey &sk, GBSK_BKU &bsk_bku);
    void RotateGGSW(GGSW &rotkey, GGSW &key, const size_t rotidx);
    void RotateGBSK(GGSW &rotkey, GBSK &bsk, LWE &lwe, const size_t idx);
    void RotateGBSK_BKU(GGSW &rotkey, GBSK_BKU &bsk_bku, LWE &lwe, const size_t idx);
    void BlindRotate_internal(LWE &lwe, GBSK_BKU &bskntt, GLWE &testvector);
    void SampleExtractIndex(vector<LWE> &lwe, GLWE &rlwe, const size_t index);
    void GateBootstrappingLWE2LWENTT(vector<LWE> &res, LWE &lwe, GBSK_BKU &bskntt, GLWE &testvector);
    void GateBootstrappingLWE2LWENTTwithoutTestVec(vector<LWE> &res, LWE &lwe, GBSK_BKU &bskntt);
    void LWEScaling(LWE &res);
    void TEST_RotateGBSK_BKU(LWE &lwe);

  private:
    SEALContext context_;
    util::Pointer<std::uint64_t> secret_key_array_;
    SecretKey secret_key_;
    Decryptor decryptor_;
    TFHERNS rns_;
    size_t poly_modulus_degree;
    size_t coeff_modulus_size;
    EncryptionParameters parms;
    size_t secret_key_array_size;
    GGSW GadgetMat;
    MemoryPoolHandle pool_ = MemoryManager::GetPool();
};
} // namespace seal