#pragma once

#include <seal/ciphertext.h>
#include <seal/kswitchkeys.h>
#include <thread>
#include <bitset>
#include <assert.h>
using namespace seal;
using namespace std;

using BSK = vector<vector<Ciphertext>>;

// For better decoupling, duplicate the name of LWE to LWE_gb and pass arguments after type casting
struct LWE_gb {
  LWE_gb() {}
  LWE_gb(size_t m, uint64_t mod) {
    n = m;
    modulus = mod;
    cipher.reserve(n + 1);
  }
  size_t n{0};
  uint64_t modulus{0};
  std::vector<uint64_t> cipher;
};

// select gpus to use
constexpr int MAX_THREAD_SIZE = 1024; // thread size limit of CUDA
constexpr int MAX_RNS_MODULUS = 4; // need to set in advance as cuda does not support array with dynamic size
constexpr int MAX_RNS_QWORD = 8; // need to set in advance as cuda does not support array with dynamic size
constexpr int MAX_M = 4; // need to set in advance as cuda does not support array with dynamic size

struct GB_CUDA_Params {
    // uint32_t Bgbit;
    uint32_t Lgsw;
    // uint32_t digits;
    uint32_t MLWE_k;
    uint32_t BKU_m;
    uint32_t coeff_modulus_size;
    uint32_t poly_modulus_degree;
    uint32_t poly_mlwe_degree;
    uint32_t lwe_n;
    vector<uint32_t> gpu_id;
};

EncryptionParameters ConvertContext(SEALContext &context, uint32_t k_);

class GB_CUDA
{
    public:
        GB_CUDA(SEALContext &context, uint32_t n_lwe_, uint32_t lwe_n_, uint32_t digits_, uint32_t Bgbit_, uint32_t BKU_m_, uint32_t MLWE_k_, uint32_t Lgsw_, int n_bootstrap, vector<uint32_t> &gpu_id) : context_(context), context_mlwe(ConvertContext(context, MLWE_k_)), n_lwe(n_lwe_), lwe_n(lwe_n_), digits(digits_), Bgbit(Bgbit_), BKU_m(BKU_m_), MLWE_k(MLWE_k_), Lgsw(Lgsw_), N_BootStrap(n_bootstrap), GPU_ID(gpu_id), N_GPU(gpu_id.size())
        {
            auto &context_data = *context_mlwe.first_context_data().get();
            auto &parms = context_data.parms();
            coeff_modulus_size = parms.coeff_modulus().size();
            rns_coeff_modulus_size = coeff_modulus_size + 1;
            poly_mlwe_degree = parms.poly_modulus_degree();
            poly_modulus_degree = (poly_mlwe_degree * MLWE_k_);

            gbcudaparam.coeff_modulus_size = coeff_modulus_size;
            gbcudaparam.poly_modulus_degree = poly_modulus_degree;
            gbcudaparam.poly_mlwe_degree = poly_mlwe_degree;
            gbcudaparam.lwe_n = lwe_n_;
            gbcudaparam.BKU_m = BKU_m_;
            gbcudaparam.MLWE_k = MLWE_k_;
            gbcudaparam.Lgsw = Lgsw_;
            gbcudaparam.gpu_id = GPU_ID;

            printf("GB_CUDA init!\n");
        }
        ~GB_CUDA()
        {
            Deinit_Pack();
            Deinit_Bootstrap();
            printf("GB_CUDA deinit!\n");
        }
        void Init_Bootstrapping(Ciphertext &testvector, BSK &bsk_bku);
        void Init_Pack(const KSwitchKeys &ksk_in);
        void Bootstrap_batch(vector<vector<LWE_gb>> &lwe_out, vector<LWE_gb> &lwe_in);
        void Bootstrap(LWE_gb* lwe_in, vector<LWE_gb>* lwe_out, size_t gpu);
        void PackLWEs(vector<vector<LWE_gb>> &lwes_zp, uint32_t n, Ciphertext& rlwe_ct);
        void verify_lwe(vector<LWE_gb> &lwe_1, vector<LWE_gb> &lwe_2, const SEALContext& context, int verbose);

    private:
        void Init_Bootstrapping_per_gpu(uint32_t gpu, Ciphertext &testvector, BSK &bsk_bku);
        void Deinit_Bootstrap();
        void Deinit_Pack();
        void Bootstrap_batch_per_gpu(LWE_gb* lwe_in, vector<LWE_gb>* lwe_out, size_t begin, size_t length, size_t gpu);
        void rns_constant_init(vector<uint64_t> &Qrst, vector<uint64_t> &Qinv);
        void gadget_gen(vector<Ciphertext> &GadgetMat);
        void extract_constants(vector<vector<uint64_t>> &twiddle_fac_in, vector<uint64_t> &q_arr_in, vector<uint64_t> &mu_2q_arr_in, vector<uint64_t> &mu_4q_arr_in, vector<uint64_t> &modswitch_arr_in, vector<uint64_t> &N_inv_q_arr_in, vector<uint32_t> &qbit_arr_in);
        
        // number of parallel processed Bootstrapping 
        int N_BootStrap;
        uint32_t rns_coeff_modulus_size;
        uint32_t poly_modulus_degree;
        uint32_t poly_mlwe_degree;
        uint32_t coeff_modulus_size;
        uint32_t lwe_n; // lwe.n
        uint32_t n_lwe; // number of input lwes
        uint32_t digits;
        uint32_t Bgbit;
        uint32_t BKU_m;
        uint32_t MLWE_k;
        uint32_t Lgsw;
        SEALContext context_;
        SEALContext context_mlwe;
        uint32_t N_GPU;
        vector<uint32_t> GPU_ID;
        GB_CUDA_Params gbcudaparam;
};