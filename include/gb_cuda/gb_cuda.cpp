#include "gb_cuda.h"
#include "kernels.cuh"

EncryptionParameters ConvertContext(SEALContext &context, uint32_t k_)
{
    EncryptionParameters bfvparams(scheme_type::bfv);
    auto &params = context.key_context_data().get()->parms();
    bfvparams.set_poly_modulus_degree(params.poly_modulus_degree() / k_);
    bfvparams.set_coeff_modulus(params.coeff_modulus());
    bfvparams.set_plain_modulus(params.plain_modulus());
    return bfvparams;
}

void GB_CUDA::Bootstrap(LWE_gb* lwe_in, vector<LWE_gb>* lwe_out, size_t gpu)
{
    Bootstrap_CUDA(&lwe_in[0], &lwe_out[0], 1, gbcudaparam, gpu);
}

void GB_CUDA::Bootstrap_batch_per_gpu(LWE_gb* lwe_in, vector<LWE_gb>* lwe_out, size_t begin, size_t length, size_t gpu)
{
    for (size_t i = begin; i < begin + length; i+=N_BootStrap)
    {
        Bootstrap_CUDA(&lwe_in[i], &lwe_out[i], N_BootStrap, gbcudaparam, gpu);
    }
}
void GB_CUDA::Bootstrap_batch(vector<vector<LWE_gb>> &lwe_out, vector<LWE_gb> &lwe_in)
{
    uint32_t length = lwe_in.size();
    uint32_t concurrent = (length / N_BootStrap) / N_GPU;
    uint32_t rest = (length / N_BootStrap) % N_GPU;
    printf("concurrent = %d, rest = %d\n", concurrent, rest);

    lwe_out.resize(length);

    vector<thread> threads;

    for (int g = 0; g < N_GPU; g++) 
    {
        threads.push_back(thread(&GB_CUDA::Bootstrap_batch_per_gpu, this, &lwe_in[0], &lwe_out[0], g * concurrent * N_BootStrap, concurrent * N_BootStrap, g));
    }

    for (auto &th : threads) 
    {
        th.join();
    }

    if (rest != 0)
    {
        threads.clear();
        
        for (int g = 0; g < rest; g++) 
        {
            threads.push_back(thread(&GB_CUDA::Bootstrap_batch_per_gpu, this, &lwe_in[0], &lwe_out[0], N_GPU * concurrent * N_BootStrap + g * N_BootStrap, N_BootStrap, g));
        }

        for (auto &th : threads) 
        {
            th.join();
        }
    }
}

void GB_CUDA::PackLWEs(vector<vector<LWE_gb>> &lwes_zp, uint32_t n, Ciphertext& rlwe_ct) 
{
    const auto& parms = context_.first_context_data()->parms();
    const size_t N = parms.poly_modulus_degree();
    if (!(n > 0 && n <= N && N % n == 0))
    {
        printf("[PackLWEsForward]: invalid lwe size\n");
        return;
    }

    ReduceTwo_CUDA(lwes_zp, rlwe_ct, n, gbcudaparam);

    const size_t log2N = static_cast<size_t>(log2(N));
    const size_t log2Nn = static_cast<size_t>(log2(N / n));
    printf("log2N=%zu, log2(N/n)=%zu, n=%zu\n", log2N, log2Nn, n);
    for (size_t k = 1; k <= log2Nn; ++k) {
      Ciphertext tmp{rlwe_ct};
      Automorph_CUDA(tmp, (1UL << (log2N - k + 1)) + 1, n, gbcudaparam);
      printf("galois %zu\n", (1UL << (log2N - k + 1)) + 1);
      add_CUDA(rlwe_ct, tmp, rlwe_ct, gbcudaparam);
    }
}

void GB_CUDA::Init_Bootstrapping(Ciphertext &testvector, BSK &bsk_bku)
{
    printf("%d gpus are avaliable!\n", N_GPU);

    twiddle_factor.resize(N_GPU);
    inv_twiddle_factor.resize(N_GPU);
    d_decntt_arr.resize(N_GPU);
    d_testvector_arr.resize(N_GPU);
    d_initvector_arr.resize(N_GPU);
    d_lwe_cipher.resize(N_GPU);
    d_res_lwe.resize(N_GPU);
    d_crtdec.resize(N_GPU);
    d_bsk_bku.resize(N_GPU);
    d_rotkey.resize(N_GPU);
    d_gadetmat.resize(N_GPU);
    d_rotpoly.resize(N_GPU);
    d_rot_bku_idx.resize(N_GPU);

    streams.resize(2 * N_GPU);

    for (uint32_t i = 0; i < N_GPU; i++) 
    {
        Init_Bootstrapping_per_gpu(i, testvector, bsk_bku);
    }
}

void GB_CUDA::Init_Bootstrapping_per_gpu(uint32_t gpu, Ciphertext &testvector, BSK &bsk_bku)
{
    cudaSetDevice(GPU_ID[gpu]);
    cudaStreamCreateWithFlags(&streams[gpu * 2], cudaStreamNonBlocking);
    cudaDeviceSynchronize();
    cudaStreamCreateWithFlags(&streams[gpu * 2 + 1], cudaStreamNonBlocking);

    // extract constants
    // constants for context
    // init and assert
    uint32_t RNS_QWORD = digits / Bgbit + 1;
    if (RNS_QWORD > (MAX_RNS_QWORD))
    {
        printf("FATAL: need to specify a larger MAX_RNS_QWORD!\n");
        exit(-1);
    }
    if (rns_coeff_modulus_size > MAX_RNS_MODULUS)
    {
        printf("FATAL: need to specify a larger MAX_RNS_MODULUS!\n");
        exit(-1);
    }
    if (BKU_m > MAX_M)
    {
        printf("FATAL: need to specify a larger MAX_M!\n");
        exit(-1);
    }
    uint32_t ROTKEY_SIZE = (MLWE_k+1) * Lgsw * (MLWE_k+1) * poly_mlwe_degree * coeff_modulus_size;
    uint32_t DECNTT_SIZE = (poly_mlwe_degree * (MLWE_k+1) * coeff_modulus_size * Lgsw);
    uint32_t TESTVEC_SIZE = (poly_mlwe_degree * (MLWE_k+1) * coeff_modulus_size);
    uint32_t BKUSIZE = (lwe_n / BKU_m) + (lwe_n % BKU_m);
    uint32_t KSIZE = (lwe_n / BKU_m) * ((1 << BKU_m) - 1) + (lwe_n % BKU_m);
    uint32_t M_MASK = (1 << BKU_m) - 1;
    // pass values to GPU
    cudaMemcpyToSymbol(&D_RNS_QWORD, &RNS_QWORD, sizeof(uint32_t));
    cudaMemcpyToSymbol(&D_ROTKEY_SIZE, &ROTKEY_SIZE, sizeof(uint32_t));
    cudaMemcpyToSymbol(&D_DECNTT_SIZE, &DECNTT_SIZE, sizeof(uint32_t));
    cudaMemcpyToSymbol(&D_TESTVEC_SIZE, &TESTVEC_SIZE, sizeof(uint32_t));
    cudaMemcpyToSymbol(&D_N_LWE, &n_lwe, sizeof(uint32_t));
    cudaMemcpyToSymbol(&D_BKUSIZE, &BKUSIZE, sizeof(uint32_t));
    cudaMemcpyToSymbol(&D_M_MASK, &M_MASK, sizeof(uint32_t));
    cudaMemcpyToSymbol(&D_RNS_COEFF_MODULUS, &rns_coeff_modulus_size, sizeof(uint32_t));
    cudaMemcpyToSymbol(&D_COEFF_MODULUS, &coeff_modulus_size, sizeof(uint32_t));
    cudaMemcpyToSymbol(&D_POLY_DEGREE, &poly_modulus_degree, sizeof(uint32_t));
    cudaMemcpyToSymbol(&D_POLY_MLWE_DEGREE, &poly_mlwe_degree, sizeof(uint32_t));
    cudaMemcpyToSymbol(&D_LWE_N, &lwe_n, sizeof(uint32_t));
    cudaMemcpyToSymbol(&D_BGBIT, &Bgbit, sizeof(uint32_t));
    cudaMemcpyToSymbol(&D_BKU_M, &BKU_m, sizeof(uint32_t));
    cudaMemcpyToSymbol(&D_MLWE_K, &MLWE_k, sizeof(uint32_t));
    cudaMemcpyToSymbol(&D_LGSW, &Lgsw, sizeof(uint32_t));
    
    // buffers to save constant values
    vector<vector<uint64_t>> twiddle_fac_in;
    vector<uint64_t> q_arr_in, mu_2q_arr_in, mu_4q_arr_in, modswitch_arr_in, N_inv_q_arr_in;
    vector<uint32_t> qbit_arr_in;
    extract_constants(twiddle_fac_in, q_arr_in, mu_2q_arr_in, mu_4q_arr_in, modswitch_arr_in, N_inv_q_arr_in, qbit_arr_in);
    // constants for barrett reduction
    cudaMemcpyToSymbol(q_arr, q_arr_in.data(), rns_coeff_modulus_size * sizeof(uint64_t));
    cudaMemcpyToSymbol(mu_4q_arr, mu_4q_arr_in.data(), rns_coeff_modulus_size * sizeof(uint64_t));
    cudaMemcpyToSymbol(mu_2q_arr, mu_2q_arr_in.data(), rns_coeff_modulus_size * sizeof(uint64_t));
    cudaMemcpyToSymbol(qbit_arr, qbit_arr_in.data(), rns_coeff_modulus_size * sizeof(uint32_t));
    // constants for ntt/intt
    cudaMalloc ((void**)&twiddle_factor[gpu], poly_mlwe_degree * rns_coeff_modulus_size * sizeof (uint64_t));
    cudaMemcpyAsync ((void*)twiddle_factor[gpu], twiddle_fac_in[0].data(), poly_mlwe_degree * rns_coeff_modulus_size * sizeof (uint64_t), cudaMemcpyHostToDevice, streams[gpu * 2]);
    cudaMalloc ((void**)&inv_twiddle_factor[gpu], poly_mlwe_degree * rns_coeff_modulus_size * sizeof (uint64_t));
    cudaMemcpyAsync ((void*)inv_twiddle_factor[gpu], twiddle_fac_in[1].data(), poly_mlwe_degree * rns_coeff_modulus_size * sizeof (uint64_t), cudaMemcpyHostToDevice, streams[gpu * 2]);
    // constants for packing
    cudaMemcpyToSymbol(N_inv_q_arr, N_inv_q_arr_in.data(), coeff_modulus_size * sizeof(uint64_t));
    cudaMemcpyToSymbol(modswitch_arr, modswitch_arr_in.data(), coeff_modulus_size * sizeof(uint64_t));


    // bsk_bku
    cudaMalloc ((void**)&d_bsk_bku[gpu], (KSIZE * (MLWE_k+1) * Lgsw * (MLWE_k+1) * coeff_modulus_size * poly_mlwe_degree) * sizeof (uint64_t));
    for (size_t ks = 0; ks < KSIZE; ks++)
    {
        for (size_t gg = 0; gg<(MLWE_k + 1) * Lgsw; gg++)
        {
            for (size_t ii = 0; ii < (MLWE_k + 1); ii++)
            {
                uint32_t index = ((ks * (MLWE_k+1) * Lgsw + gg) * (MLWE_k+1) + ii);
                cudaMemcpyAsync ((void*)d_bsk_bku[gpu] + index * coeff_modulus_size * poly_mlwe_degree * sizeof (uint64_t), bsk_bku[ks][gg].data(ii), coeff_modulus_size * poly_mlwe_degree * sizeof (uint64_t), cudaMemcpyHostToDevice, streams[gpu * 2]);
            }
        }
    }
    // rotpoly for bku
    cudaMalloc ((void**)&d_rotpoly[gpu], (2 * poly_mlwe_degree * poly_mlwe_degree * coeff_modulus_size) * sizeof (uint64_t));
    cudaMemset(d_rotpoly[gpu], 0, (2 * poly_mlwe_degree * poly_mlwe_degree * coeff_modulus_size) * sizeof (uint64_t));
    set_rotpoly(gpu, d_rotpoly[gpu], twiddle_factor[gpu], gbcudaparam);
    // GadgetMat for bku
    vector<Ciphertext> GadgetMat;
    gadget_gen(GadgetMat);
    cudaMalloc ((void**)&d_gadetmat[gpu], ((MLWE_k+1) * Lgsw * (MLWE_k+1) * poly_mlwe_degree * coeff_modulus_size) * sizeof (uint64_t));
    for (size_t j = 0; j < (MLWE_k + 1) * Lgsw; j++)
    {
        for (size_t ii=0; ii<(MLWE_k + 1); ii++)
        {
            cudaMemcpyAsync ((void*)d_gadetmat[gpu] + (j * (MLWE_k + 1) + ii) * coeff_modulus_size * poly_mlwe_degree * sizeof (uint64_t), GadgetMat[j].data(ii), poly_mlwe_degree * coeff_modulus_size * sizeof (uint64_t), cudaMemcpyHostToDevice, streams[gpu * 2]);
        }
    }


    // rns constants
    vector<uint64_t> Qrst;
    vector<uint64_t> Qinv;
    rns_constant_init(Qrst, Qinv);
    cudaMemcpyToSymbol(rns_q_inv, Qinv.data(), coeff_modulus_size * sizeof(uint64_t));
    cudaMemcpyToSymbol(rns_q_rst, Qrst.data(), coeff_modulus_size * RNS_QWORD * sizeof(uint64_t));

    cudaMalloc ((void**)&d_initvector_arr[gpu], TESTVEC_SIZE * sizeof (uint64_t));
    for (int i=0; i<(MLWE_k+1); i++)
    {
        cudaMemcpyAsync ((void*)d_initvector_arr[gpu] + i * coeff_modulus_size * poly_mlwe_degree * sizeof (uint64_t), &testvector.data(i)[0], poly_mlwe_degree * coeff_modulus_size * sizeof (uint64_t), cudaMemcpyHostToDevice, streams[gpu * 2]);
    }

    cudaMalloc ((void**)&d_rot_bku_idx[gpu], BKUSIZE * N_BootStrap * M_MASK * sizeof (uint64_t));
    cudaMalloc ((void**)&d_decntt_arr[gpu], N_BootStrap * DECNTT_SIZE * sizeof (uint64_t));
    cudaMalloc ((void**)&d_testvector_arr[gpu], N_BootStrap * TESTVEC_SIZE * sizeof (uint64_t));
    cudaMalloc ((void**)&d_lwe_cipher[gpu], N_BootStrap * (lwe_n + 1) * sizeof (uint64_t));
    cudaMalloc ((void**)&d_res_lwe[gpu], N_BootStrap * coeff_modulus_size * (poly_modulus_degree + 1) * sizeof (uint64_t));
    cudaMalloc ((void**)&d_crtdec[gpu], N_BootStrap * poly_mlwe_degree * (MLWE_k+1) * RNS_QWORD * sizeof (uint64_t));
    cudaMalloc ((void**)&d_rotkey[gpu], N_BootStrap * ROTKEY_SIZE * sizeof (uint64_t));

    cudaDeviceSynchronize();
}

void GB_CUDA::Init_Pack(const KSwitchKeys &ksk_in)
{
    cudaSetDevice(GPU_ID[0]);

    // extract constants
    vector<vector<uint64_t>> twiddle_fac_in;
    vector<uint64_t> q_arr_in, mu_2q_arr_in, mu_4q_arr_in, modswitch_arr_in, N_inv_q_arr_in;
    vector<uint32_t> qbit_arr_in;
    extract_constants(twiddle_fac_in, q_arr_in, mu_2q_arr_in, mu_4q_arr_in, modswitch_arr_in, N_inv_q_arr_in, qbit_arr_in);
    // constants for barrett reduction
    cudaMemcpyToSymbol(q_arr, q_arr_in.data(), rns_coeff_modulus_size * sizeof(uint64_t));
    cudaMemcpyToSymbol(mu_4q_arr, mu_4q_arr_in.data(), rns_coeff_modulus_size * sizeof(uint64_t));
    cudaMemcpyToSymbol(mu_2q_arr, mu_2q_arr_in.data(), rns_coeff_modulus_size * sizeof(uint64_t));
    cudaMemcpyToSymbol(qbit_arr, qbit_arr_in.data(), rns_coeff_modulus_size * sizeof(uint32_t));
    // constants for ntt/intt
    cudaFree(twiddle_factor[0]);
    cudaMalloc ((void**)&twiddle_factor[0], poly_mlwe_degree * rns_coeff_modulus_size * sizeof (uint64_t));
    cudaMemcpyAsync ((void*)twiddle_factor[0], twiddle_fac_in[0].data(), poly_mlwe_degree * rns_coeff_modulus_size * sizeof (uint64_t), cudaMemcpyHostToDevice, streams[0]);
    cudaFree(inv_twiddle_factor[0]);
    cudaMalloc ((void**)&inv_twiddle_factor[0], poly_mlwe_degree * rns_coeff_modulus_size * sizeof (uint64_t));
    cudaMemcpyAsync ((void*)inv_twiddle_factor[0], twiddle_fac_in[1].data(), poly_mlwe_degree * rns_coeff_modulus_size * sizeof (uint64_t), cudaMemcpyHostToDevice, streams[0]);
    // constants for packing
    cudaMemcpyToSymbol(N_inv_q_arr, N_inv_q_arr_in.data(), coeff_modulus_size * sizeof(uint64_t));
    cudaMemcpyToSymbol(modswitch_arr, modswitch_arr_in.data(), coeff_modulus_size * sizeof(uint64_t));

    cudaMalloc ((void**)&d_ksk, (uint32_t)(log2(poly_modulus_degree)) * coeff_modulus_size * 2 * rns_coeff_modulus_size * poly_modulus_degree * sizeof (uint64_t));
    cudaMalloc ((void**)&d_rlwe_buf_1, (n_lwe / 2) * 2 * coeff_modulus_size * poly_modulus_degree * sizeof (uint64_t));
    cudaMalloc ((void**)&d_rlwe_buf_2, (n_lwe / 2) * 2 * coeff_modulus_size * poly_modulus_degree * sizeof (uint64_t));
    cudaMalloc ((void**)&d_rlwe_buf_3, (n_lwe / 2) * 2 * coeff_modulus_size * poly_modulus_degree * sizeof (uint64_t));
    cudaMalloc ((void**)&d_rlwes, n_lwe * 2 * coeff_modulus_size * poly_modulus_degree * sizeof (uint64_t));
    cudaMalloc ((void**)&d_pack_buffer, (n_lwe / 2) * coeff_modulus_size * rns_coeff_modulus_size * poly_modulus_degree * sizeof (uint64_t));
    cudaMalloc ((void**)&d_prod, (n_lwe / 2) * 2 * rns_coeff_modulus_size * poly_modulus_degree * sizeof (uint64_t));
    cudaMalloc ((void**)&d_lwes, n_lwe * coeff_modulus_size * (poly_modulus_degree + 1) * sizeof (uint64_t));

    for (size_t S=0; S<(uint32_t)(log2(poly_modulus_degree)); S++)
    {
        for (size_t J=0; J<coeff_modulus_size; J++)
        {
            for (size_t K=0; K<2; K++)
            {
                cudaMemcpyAsync ((void*)(&(d_ksk[S * coeff_modulus_size * 2 * rns_coeff_modulus_size * poly_modulus_degree + J * 2 * rns_coeff_modulus_size * poly_modulus_degree + K * rns_coeff_modulus_size * poly_modulus_degree])), &(ksk_in.data()[1 << S][J].data().data(K)[0]), rns_coeff_modulus_size * poly_modulus_degree * sizeof (uint64_t), cudaMemcpyHostToDevice, streams[S % 2]);
            }
        }
    }

    cudaDeviceSynchronize();
}

void GB_CUDA::Deinit_Bootstrap()
{
    for (uint32_t gpu=0; gpu<N_GPU; gpu++)
    {
        cudaSetDevice(GPU_ID[gpu]);
        
        cudaFree(twiddle_factor[gpu]);
        cudaFree(inv_twiddle_factor[gpu]);
        cudaFree(d_bsk_bku[gpu]);
        cudaFree(d_initvector_arr[gpu]);
        cudaFree(d_testvector_arr[gpu]);
        cudaFree(d_decntt_arr[gpu]);
        cudaFree(d_lwe_cipher[gpu]);
        cudaFree(d_res_lwe[gpu]);
        cudaFree(d_crtdec[gpu]);
        cudaFree(d_rotkey[gpu]);
        cudaFree(d_gadetmat[gpu]);
        cudaFree(d_rotpoly[gpu]);
        cudaFree(d_rot_bku_idx[gpu]);

        cudaStreamDestroy(streams[gpu * 2]);
        cudaStreamDestroy(streams[gpu * 2 + 1]);
    }
}

void GB_CUDA::Deinit_Pack()
{
    cudaSetDevice(GPU_ID[0]);

    cudaFree(d_rlwe_buf_1);
    cudaFree(d_rlwe_buf_2);
    cudaFree(d_rlwe_buf_3);
    cudaFree(d_rlwes);
    cudaFree(d_pack_buffer);
    cudaFree(d_prod);
    cudaFree(d_lwes);
}

// rns utils
inline void basered(const uint64_t *vec_in, uint64_t *vec_out, size_t in_cnt, size_t Qword, const size_t sft, uint32_t digits_, uint32_t Bgbit_)
{
    const uint32_t MAX_BITSET = 256;
    if (digits_ + Bgbit_ + 4 > MAX_BITSET)
    {
        printf("FATAL: need to specify a larger MAX_BITSET!\n");
        exit(-1);
    }

    bitset<MAX_BITSET> buff(0);
    bitset<MAX_BITSET> mask((1 << Bgbit_) - 1);
    for (size_t i = 0; i < in_cnt; i++) {
        buff <<= 64;
        buff |= vec_in[in_cnt - 1 - i];
    }
    buff <<= sft;
    for (size_t i = 0; i < Qword; i++) {
        if ((i + 1) * Bgbit_ > digits_)
            vec_out[i] = static_cast<uint64_t>(((buff << ((i + 1) * Bgbit_ - digits_)) & mask).to_ulong());
        else
            vec_out[i] = static_cast<uint64_t>(((buff >> (digits_ - (i + 1) * Bgbit_)) & mask).to_ulong());
    }
}
void GB_CUDA::rns_constant_init(vector<uint64_t> &Qrst, vector<uint64_t> &Qinv)
{
    auto &context_data = *context_mlwe.first_context_data().get();
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    auto Mcnt = coeff_modulus.size();
    auto Qlen = digits;
    size_t Wlen = Bgbit;
    size_t Qword = Qlen / Wlen + 1;
    size_t coeff_modulus_size = Mcnt;
    auto punctured_prod_array = context_data.rns_tool()->base_q()->punctured_prod_array();
    auto inv_punctured_prod = context_data.rns_tool()->base_q()->inv_punctured_prod_mod_base_array();
    auto base_prod = context_data.rns_tool()->base_q()->base_prod();
    size_t Qsize = context_data.rns_tool()->base_q()->size();
    Qrst.resize(Mcnt * Qword);
    Qinv.resize(Mcnt);
    for (size_t i = 0; i < Mcnt; i++)
        Qinv[i] = inv_punctured_prod[i].operand;
    for (size_t i = 0; i < Mcnt; i++) {
        basered(punctured_prod_array + Qsize * i, &Qrst[i * Qword], Qsize, Qword, 0, Qlen, Wlen);
    }
}

// bsk_bku utils
inline uint64_t GadgetRed(uint64_t plain, size_t idx, const Modulus &modulus, uint32_t digits_, uint32_t Bgbit_)
{
    size_t Qlen = digits_ - (idx + 1) * Bgbit_;
    size_t Wlen = 63;
    uint64_t res = util::barrett_reduce_64(plain << (Qlen % Wlen), modulus);
    for (size_t i = 0; i < Qlen / Wlen; i++) {
        res = util::multiply_uint_mod(res, (1ULL << Wlen), modulus);
    }
    return res;
}
void GB_CUDA::gadget_gen(vector<Ciphertext> &GadgetMat)
{
    size_t rlwe_count = (MLWE_k + 1) * Lgsw;
    auto &parms = context_mlwe.first_context_data().get()->parms();
    auto &modulus = parms.coeff_modulus();
    auto coeff_modulus_size = modulus.size();

    GadgetMat.resize((MLWE_k + 1) * Lgsw);
    for (size_t i = 0; i < rlwe_count; i++) {
        GadgetMat[i].resize(context_mlwe, (MLWE_k + 1));
    }
    for (size_t k = 0; k <= MLWE_k; k++) {
        for (size_t i = 0; i < Lgsw; i++) {
            for (size_t j = 0; j < poly_mlwe_degree; j++) {
                for (size_t m = 0; m < coeff_modulus_size; m++) {
                    uint64_t red = GadgetRed(1, i, modulus[m], digits, Bgbit);
                    GadgetMat[i + k * Lgsw].data(k)[m * poly_mlwe_degree + j] = red;
                }
            }
        }
    }
}

// extract needed constants from seal context
void GB_CUDA::extract_constants(vector<vector<uint64_t>> &twiddle_fac_in, vector<uint64_t> &q_arr_in, vector<uint64_t> &mu_2q_arr_in, vector<uint64_t> &mu_4q_arr_in, vector<uint64_t> &modswitch_arr_in, vector<uint64_t> &N_inv_q_arr_in, vector<uint32_t> &qbit_arr_in)
{
    auto &key_context_data = *context_mlwe.key_context_data();
    auto key_ntt_tables = iter(key_context_data.small_ntt_tables());
    auto modswitch_factors = key_context_data.rns_tool()->inv_q_last_mod_q();
    uint32_t Nbit = log2(poly_mlwe_degree);
    twiddle_fac_in.resize(2);
    for (size_t i=0; i<twiddle_fac_in.size(); i++)
       twiddle_fac_in[i].resize(rns_coeff_modulus_size * poly_mlwe_degree);

    q_arr_in.resize(rns_coeff_modulus_size);
    mu_2q_arr_in.resize(rns_coeff_modulus_size);
    mu_4q_arr_in.resize(rns_coeff_modulus_size);
    qbit_arr_in.resize(rns_coeff_modulus_size);
    modswitch_arr_in.resize(coeff_modulus_size);
    N_inv_q_arr_in.resize(coeff_modulus_size);

    for (size_t j=0; j<rns_coeff_modulus_size; j++)
    {
        const util::NTTTables &ntt_tables = key_ntt_tables[j];

        for (size_t i=0; i<poly_mlwe_degree; i++)
        {
            // SEAL's twiddle factors for NTT can be directly used
            twiddle_fac_in[0][j * poly_mlwe_degree + i] = (ntt_tables.get_from_root_powers() + i)->operand;

            // SEAL's twiddle factors for INTT need to be specifically transformed 
            // in order to feed our INTT design (which is embeded with pre and post scaling)
            uint64_t temp=i-1;
            uint64_t res = 0;

            for (int ii = 0; ii < Nbit; ii++)
            {
                res <<= 1;
                res = (temp & 1) | res;
                temp >>= 1;
            }

            temp = res + 1;
            res = 0;

            for (int ii = 0; ii < Nbit; ii++)
            {
                res <<= 1;
                res = (temp & 1) | res;
                temp >>= 1;
            }

            twiddle_fac_in[1][j * poly_mlwe_degree + res] = (ntt_tables.get_from_inv_root_powers() + i)->operand;
        }

        q_arr_in[j] = ntt_tables.modulus().value();
        qbit_arr_in[j] = ceil(log2(q_arr_in[j]));
        uint128_t mu1 = (uint128_t)1 << (qbit_arr_in[j] * 2);
        mu_2q_arr_in[j] = (uint64_t)(mu1 / q_arr_in[j]);
        mu1 = (uint128_t)1 << (qbit_arr_in[j] * 2 + 3);
        mu_4q_arr_in[j] = (uint64_t)(mu1 / q_arr_in[j]);

        if (j < coeff_modulus_size)
        {
            modswitch_arr_in[j] = modswitch_factors[j].operand;
            N_inv_q_arr_in[j] = ntt_tables.inv_degree_modulo().operand;
        }
    }
}

void GB_CUDA::verify_lwe(vector<LWE_gb> &lwe_1, vector<LWE_gb> &lwe_2, const SEALContext& context, int verbose)
{
    int error_cnt = 0, error_temp = 0, totoal_cnt = 0, error_index = -1;

    for (size_t jj = 0; jj < coeff_modulus_size; jj++) {
        uint64_t q = context.first_context_data()->small_ntt_tables()[jj].modulus().value();
        // if (lwe_1[jj].n != lwe_1[jj].n || lwe_1[jj].modulus != lwe_2[jj].modulus)
        if (lwe_1[jj].n != lwe_1[jj].n)
        {
            printf("not match at all! stop verification!\n");
            return;
        }

        for (size_t nn=0; nn<(lwe_1[jj].n+1); nn++)
        {
            totoal_cnt++;
            if ((lwe_1[jj].cipher[nn] % q) != (lwe_2[jj].cipher[nn] % q))
            {
                error_cnt++;
                if (verbose & 1 == 1)
                printf("q=%llu, lwe[%d].cipher[%d]    (lwe_1)%llu ?= (lwe_2)%llu\n", q, jj, nn, lwe_1[jj].cipher[nn], lwe_2[jj].cipher[nn]);
            }
        }
    }
    printf("    error rate: %d/%d\n", error_cnt, totoal_cnt);

    if ((verbose & 2) > 0)
    {
        uint32_t jj = rand()%coeff_modulus_size;
        uint32_t nn = rand()%(lwe_1[jj].n+1);

        printf("    Random Sample: lwe[%d].cipher[%d]; (lwe_1)%llu ?= (lwe_2)%llu\n", jj, nn, lwe_1[jj].cipher[nn], lwe_2[jj].cipher[nn]);
    }
}