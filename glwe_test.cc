#include "glwe.h"
#include "keygenerator.h"
#include "tfhe.h"
#include "timer.h"
#include <array>
#include <cstdlib>
#include <sys/resource.h>

// header file of GB_CUDA lib
#include "gb_cuda/gb_cuda.h"

using namespace std;
using namespace seal;
void print(uint64_t *data, size_t poly_degree, size_t modulus_size, size_t sample_per_row)
{
    for (size_t m = 0; m < modulus_size; m++) {
        for (size_t i = 0; i < poly_degree; i += (poly_degree / sample_per_row))
            cout << data[m * poly_degree + i] << "\t";
        cout << endl;
    }
}
int main(void)
{
    // const rlim_t stacksize = 32 * 1024 * 1024;
    // struct rlimit rl;
    // rl.rlim_cur = stacksize;
    //  setrlimit(RLIMIT_STACK, &rl);
    EncryptionParameters parms_d(scheme_type::bfv);
    size_t nbit_plain = 3;
    size_t poly_modulus_degree_d = targetP::n_; // targetP::n_;
    parms_d.set_poly_modulus_degree(poly_modulus_degree_d);
    parms_d.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree_d, {60, 40, 40, 60}));
    parms_d.set_plain_modulus(Modulus((uint64_t)1 << nbit_plain));
    SEALContext context_d(parms_d);
    KeyGeneratorM keygen_d(context_d);
    // KeyGeneratorM keygen_d(context_d);
    SecretKey secret_key_d = keygen_d.secret_key();
    SecretKey secret_key_d_intt = keygen_d.secret_key_intt();
    // cout << secret_key_d.data().data()[0] << '\t';
    // cout << secret_key_d.data().data()[0] << endl;
    // cout << secret_key_d.data().dyn_array().size() << "\t" << secret_key_d_intt.data().dyn_array().size() << endl;
    // TFHECryptor tfhecryptor(context_d, secret_key_d);
    // cout << 1 << endl;
    // cout << 2 << endl;
    //  util::set_poly(secret_key_d_intt.data().data(), poly_modulus_degree_t, 4, secret_key_array.get());
    // cout << 1 << endl;
    // print(secret_key_d_intt.data().data(), poly_modulus_degree_d / targetP::k_, 1, poly_modulus_degree_d / targetP::k_);
    GLWECipher glwecipher(context_d, secret_key_d_intt);
    // print(secret_key_d_intt.data().data(), poly_modulus_degree_d / targetP::k_, 1, 32);
    GLWE glwe(glwecipher.get_context());
    glwecipher.encrypt_testvec(glwe);
    Plaintext plain(poly_modulus_degree_d / targetP::k_);
    glwecipher.decrypt(plain, glwe);
    cout << "Testvec:\n";
    print(plain.data(), poly_modulus_degree_d / targetP::k_, 1, 32);
    glwecipher.encrypt(plain, glwe);
    glwecipher.decrypt(plain, glwe);
    cout << "Encrypt:\n";
    print(plain.data(), poly_modulus_degree_d / targetP::k_, 1, 32);

    GGSW ggsw;
    glwecipher.encrypt(1, ggsw);
    cout << "GGSW" << endl;
    for (size_t i = 0; i < GGSW_size; i++) {
        cout << i << ":\t" << glwecipher.NoiseBudget(ggsw[i]) << endl;
    }
    GBSK_BKU bsk;
    double genbsk_{0.};
    MSecTimer genbsk(&genbsk_, "Generate BSK");
    glwecipher.GenGBSK_BKU(secret_key_d_intt, bsk);
    genbsk.stop();
    srand(0);

    // test bootstrapping for one LWE
    printf("test bootstrapping for one LWE!\n");
    {
        // init GB_CUDA lib
        // select needed gpus
        uint32_t gpu_arr[] = {2};
        vector<uint32_t> gpu_id(begin(gpu_arr), end(gpu_arr));
        GB_CUDA gb_cuda(
        context_d, 
        8192, // number of LWEs to be packed
        domainP::n_, // lwe.n
        targetP::digits, // digits
        targetP::Bgbit, // Bgbit
        domainP::m_, // BKU_m
        targetP::k_, // MLWE_k,
        targetP::l_, // Lgsw
        1, // number of bootstrappings that run simultaneously
        gpu_id // enabled gpus
        );

        // need to tell GB_CUDA lib the init state of testvector
        GLWE init_testvec(glwecipher.get_context());
        init_testvec.resize(glwecipher.get_context(), targetP::k_ + 1);
        glwecipher.encrypt_testvec(init_testvec);

        // init bootstrapping
        gb_cuda.Init_Bootstrapping(init_testvec, bsk);

        // create a LWE
        double timer_{0.};
        vector<LWE> res, res_temp;
        LWE lwe(domainP::n_, RAND_MAX), lwe_temp(domainP::n_, RAND_MAX);
        int pt = rand() % 2;
        lwe.cipher[domainP::n_] = (rand() & 16383) + (lwe.modulus / 4 * 3) * pt;
        for (size_t i = 0; i < domainP::n_; i++) {
            lwe.cipher[i] = rand() % RAND_MAX;
            lwe.cipher[domainP::n_] += lwe.cipher[i] * secret_key_d_intt.data()[i];
            if (lwe.cipher[domainP::n_] > lwe.modulus) lwe.cipher[domainP::n_] -= lwe.modulus;
        }

        // duplicate LWE
        for (size_t i=0; i<domainP::n_ + 1; i++)
            lwe_temp.cipher[i] = lwe.cipher[i];

        // Bootstrapping by GB_CUDA lib
        {
            MSecTimer timer(&timer_, "gb_cuda.Bootstrap:");
            gb_cuda.Bootstrap((LWE_gb *)&lwe, (vector<LWE_gb> *)&res, 0);
            timer.stop();
        }
        // Bootstrapping by glwe cpu implementation
        {
            MSecTimer timer(&timer_, "glwe_cpu.Bootstrap:");
            glwecipher.GateBootstrappingLWE2LWENTTwithoutTestVec(res_temp, lwe_temp, bsk);
            timer.stop();
        }

        // verify the results
        gb_cuda.verify_lwe(*(vector<LWE_gb> *)&res, *(vector<LWE_gb> *)&res_temp, context_d, 2);

        // cout << "Correct result: " << pt << endl;
    }

    // test bootstrapping for MANY LWE
    printf("test bootstrapping for MANY LWE!\n");
    {
        // init GB_CUDA lib
        // select needed gpus
        uint32_t gpu_arr[] = {2, 0, 3, 1};
        vector<uint32_t> gpu_id(begin(gpu_arr), end(gpu_arr));
        GB_CUDA gb_cuda(
        context_d, 
        8192, // number of LWEs to be packed
        domainP::n_, // lwe.n
        targetP::digits, // digits
        targetP::Bgbit, // Bgbit
        domainP::m_, // BKU_m
        targetP::k_, // MLWE_k,
        targetP::l_, // Lgsw
        16, // number of bootstrappings that run simultaneously
        gpu_id // enabled gpus
        );

        // need to tell GB_CUDA lib the init state of testvector
        GLWE init_testvec(glwecipher.get_context());
        init_testvec.resize(glwecipher.get_context(), targetP::k_ + 1);
        glwecipher.encrypt_testvec(init_testvec);

        // init bootstrapping
        gb_cuda.Init_Bootstrapping(init_testvec, bsk);

        // create many LWEs
        int num_lwes = 1024;
        double timer_{0.};
        vector<vector<LWE>> res_1;
        vector<LWE> res_2;
        vector<LWE> lwe_1(num_lwes);

        for (size_t ii=0; ii<num_lwes; ii++)
        {
            lwe_1[ii].n = domainP::n_;
            lwe_1[ii].modulus = RAND_MAX;
            lwe_1[ii].cipher.resize(domainP::n_ + 1);
            int pt = rand() % 2;
            lwe_1[ii].cipher[domainP::n_] = (rand() & 16383) + (RAND_MAX / 4 * 3) * pt;
            for (size_t i = 0; i < domainP::n_; i++) {
                lwe_1[ii].cipher[i] = rand() % RAND_MAX;
                lwe_1[ii].cipher[domainP::n_] += lwe_1[ii].cipher[i] * secret_key_d_intt.data()[i];
                if (lwe_1[ii].cipher[domainP::n_] > RAND_MAX) lwe_1[ii].cipher[domainP::n_] -= RAND_MAX;
            }
        }

        // Bootstrapping by GB_CUDA lib
        {
            MSecTimer timer(&timer_, "gb_cuda.Bootstrap:");
            gb_cuda.Bootstrap_batch(*(vector<vector<LWE_gb>> *)&res_1, *(vector<LWE_gb> *)&lwe_1);
            timer.stop();
        }
        // randomly select an LWE, do Bootstrapping by glwe cpu implementation
        uint32_t rand_idx = rand()%num_lwes;
        {
            MSecTimer timer(&timer_, "glwe_cpu.Bootstrap:");
            glwecipher.GateBootstrappingLWE2LWENTTwithoutTestVec(res_2, lwe_1[rand_idx], bsk);
            timer.stop();
        }

        // verify the results
        gb_cuda.verify_lwe(*(vector<LWE_gb> *)&res_1[rand_idx], *(vector<LWE_gb> *)&res_2, context_d, 2);

        // cout << "Correct result: " << pt << endl;
    }
    return 0;
}