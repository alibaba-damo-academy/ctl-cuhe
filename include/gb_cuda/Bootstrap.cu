#include "kernels.cuh"
#include "uint128_cuda.h"
#include "NTT.h"

__global__ void rotate_poly(uint64_t *initvector_arr, uint64_t *testvector_arr, uint64_t *lwe_cipher)
{
    extern __shared__ uint64_t shared_mem[];

    for (uint32_t ii=blockIdx.y; ii<(D_MLWE_K+1) * D_COEFF_MODULUS; ii+=gridDim.y)
    {

        for (size_t nn=threadIdx.x; nn<D_POLY_MLWE_DEGREE; nn+=blockDim.x)
        {
            uint64_t index_raw = lwe_cipher[blockIdx.x * (D_LWE_N + 1) + D_LWE_N] + nn;
            uint64_t index = index_raw & (uint64_t)(D_POLY_MLWE_DEGREE - 1);
            uint64_t poly = initvector_arr[ii * D_POLY_MLWE_DEGREE + nn];
            uint64_t *target_result = &shared_mem[index];
            if (!(index_raw & (uint64_t)(D_POLY_MLWE_DEGREE)) || !poly)
            {
                *target_result = poly;
            }
            else
            {
                *target_result = q_arr[ii % D_COEFF_MODULUS] - poly;
            }
        }

        __syncthreads();

        for (size_t i=threadIdx.x; i<D_POLY_MLWE_DEGREE; i+=blockDim.x)
        {
            testvector_arr[blockIdx.x * D_TESTVEC_SIZE + ii * D_POLY_MLWE_DEGREE + i] = shared_mem[i];
        }

        __syncthreads();
    }
}

__global__ void CRTDecPoly(uint64_t *testvector_arr, uint64_t *crtdec, uint64_t *decntt)
{
    for (uint32_t ii=blockIdx.y; ii<(D_MLWE_K+1); ii+=gridDim.y)
    {
        for (size_t nn=threadIdx.x; nn<D_POLY_MLWE_DEGREE; nn+=blockDim.x)
        {
            uint64_t *crtdec_out = crtdec + blockIdx.x * D_POLY_MLWE_DEGREE * (D_MLWE_K+1) * D_RNS_QWORD + ii * D_RNS_QWORD * D_POLY_MLWE_DEGREE;
            for (size_t qq = 0; qq < D_RNS_QWORD; qq++) 
            {
                crtdec_out[qq * D_POLY_MLWE_DEGREE + nn] = 0;
            }

            for (size_t jj = 0; jj < D_COEFF_MODULUS; jj++)
            {
                uint64_t value = testvector_arr[blockIdx.x * D_TESTVEC_SIZE + ii * D_COEFF_MODULUS * D_POLY_MLWE_DEGREE + jj * D_POLY_MLWE_DEGREE + nn];
                uint128_t_cu temp;
                mul64(value, rns_q_inv[jj], temp);
                singleBarrett_2q(temp, q_arr[jj], mu_2q_arr[jj], qbit_arr[jj]);
                value = temp.low;

                uint128_t_cu prod = 0;
                for (size_t qq = 0; qq < D_RNS_QWORD; qq++) 
                {
                    uint64_t val = rns_q_rst[jj * D_RNS_QWORD + D_RNS_QWORD - 1 - qq];
                    uint128_t_cu temp;
                    mul64(value, val, temp);
                    uint64_t *res = crtdec_out + (D_RNS_QWORD - 1 - qq) * D_POLY_MLWE_DEGREE + nn; // used n_modulus times
                    temp  = temp + *res;
                    prod = prod + temp;
                    *res = prod.low & ((1 << D_BGBIT) - 1);
                    uint128_t_cu::shiftr(prod, D_BGBIT);
                }
            }
            // set decntt from crtdec
            for (size_t ll = 0; ll < D_LGSW; ll++)
            {
                decntt[blockIdx.x * D_DECNTT_SIZE + ii * D_LGSW * D_POLY_MLWE_DEGREE * D_COEFF_MODULUS + ll * D_POLY_MLWE_DEGREE * D_COEFF_MODULUS + (D_COEFF_MODULUS - 1) * D_POLY_MLWE_DEGREE + nn] = crtdec_out[ll * D_POLY_MLWE_DEGREE + nn];
            }
        }
    }
}

__global__ void NTT_decntt(uint64_t *decntt, const uint64_t *twiddle_fac)
{
    extern __shared__ uint64_t shared_mem[];

    for (size_t ii=blockIdx.y; ii<(D_MLWE_K+1) * D_LGSW; ii+=gridDim.y)
    {
        for (size_t jj=blockIdx.z; jj<D_COEFF_MODULUS; jj+=gridDim.z)
        {
            for (size_t i=threadIdx.x; i<D_POLY_MLWE_DEGREE; i+=blockDim.x)
            {
                shared_mem[i] = decntt[blockIdx.x * D_DECNTT_SIZE + ii * D_COEFF_MODULUS * D_POLY_MLWE_DEGREE + (D_COEFF_MODULUS - 1) * D_POLY_MLWE_DEGREE + i];
            }

            __syncthreads();

            NTT_inline(shared_mem, D_POLY_MLWE_DEGREE, twiddle_fac + jj * D_POLY_MLWE_DEGREE, q_arr[jj], qbit_arr[jj], mu_2q_arr[jj]);

            __syncthreads();

            for (size_t i=threadIdx.x; i<D_POLY_MLWE_DEGREE; i+=blockDim.x)
            {
                decntt[blockIdx.x * D_DECNTT_SIZE + (ii * D_COEFF_MODULUS + jj) * D_POLY_MLWE_DEGREE + i] = shared_mem[i];
            }
        }
    }
}

__global__ void Rotate_BKU(uint64_t *rotkey, uint64_t *rotpoly, uint64_t *GadgetMat, uint64_t *bsk_bku, uint64_t *rot_idx, uint32_t bkun, int n_bootstrap)
{
    if (bkun < D_LWE_N / D_BKU_M) 
    {
        for (uint32_t gg=blockIdx.y; gg<(D_MLWE_K+1) * D_LGSW * (D_MLWE_K+1) * D_COEFF_MODULUS; gg+=gridDim.y)
        {
            for (size_t nn=threadIdx.x; nn<D_POLY_MLWE_DEGREE; nn+=blockDim.x)
            {
                uint64_t *rotpoly_arr = rotpoly + (gg % D_COEFF_MODULUS) * D_POLY_MLWE_DEGREE;
                uint64_t *GadgetMat_arr = GadgetMat + gg * D_POLY_MLWE_DEGREE;
                uint64_t *bku_arr = bsk_bku + gg * D_POLY_MLWE_DEGREE;
                register uint64_t q = q_arr[gg % D_COEFF_MODULUS];
                register uint64_t mu = mu_4q_arr[gg % D_COEFF_MODULUS];
                register uint32_t qbit = qbit_arr[gg % D_COEFF_MODULUS];

                register uint64_t bku_value_im[(1 << MAX_M) - 1]; // allocate an redundant array, actually we need bku_value_im[D_M_MASK]...
                for (size_t im = 0; im < D_M_MASK; im++) 
                {
                    bku_value_im[im] = bku_arr[((bkun * (D_M_MASK) + im) * (D_MLWE_K+1) * D_LGSW * (D_MLWE_K+1)) * D_POLY_MLWE_DEGREE * D_COEFF_MODULUS + nn];
                }

                for (uint32_t BS=blockIdx.x; BS<n_bootstrap; BS+=gridDim.x)
                {
                    uint128_t_cu sum = GadgetMat_arr[nn];
                    for (size_t im = 0; im < D_M_MASK; im++) 
                    {
                        // need more efforts when m > 4 after lazy optimization
                        uint128_t_cu temp;
                        mul64(rotpoly_arr[rot_idx[BS * D_BKUSIZE * D_M_MASK + bkun * D_M_MASK + im] + nn], bku_value_im[im], temp); // 120 bit

                        sum = sum + temp;

                        // lazy optimization, require m<=4
                        // singleBarrett_2q(sum, q_arr[gg % D_COEFF_MODULUS], mu_2q_arr[gg % D_COEFF_MODULUS], qbit_arr[gg % D_COEFF_MODULUS]);
                    }
                    // barrett reduction
                    singleBarrett_4q_lazy(sum, q, mu, qbit);

                    rotkey[BS * D_ROTKEY_SIZE + gg * D_POLY_MLWE_DEGREE + nn] = sum.low;
                }
            }
        }
    }
    else
    {
        for (uint32_t gg=blockIdx.y; gg<(D_MLWE_K+1) * D_LGSW * (D_MLWE_K+1) * D_COEFF_MODULUS; gg+=gridDim.y)
        {
            for (size_t nn=threadIdx.x; nn<D_POLY_MLWE_DEGREE; nn+=blockDim.x)
            {
                uint64_t *rotpoly_arr = rotpoly + (gg % D_COEFF_MODULUS) * D_POLY_MLWE_DEGREE;
                uint64_t *GadgetMat_arr = GadgetMat + gg * D_POLY_MLWE_DEGREE;
                uint64_t *bku_arr = bsk_bku + gg * D_POLY_MLWE_DEGREE;
                register uint64_t q = q_arr[gg % D_COEFF_MODULUS];
                register uint64_t mu = mu_4q_arr[gg % D_COEFF_MODULUS];
                register uint32_t qbit = qbit_arr[gg % D_COEFF_MODULUS];

                register uint64_t bku_value_im = bku_arr[(((D_LWE_N / D_BKU_M) * (D_M_MASK) + bkun - (D_LWE_N / D_BKU_M)) * (D_MLWE_K+1) * D_LGSW * (D_MLWE_K+1)) * D_POLY_MLWE_DEGREE * D_COEFF_MODULUS + nn];

                for (uint32_t BS=blockIdx.x; BS<n_bootstrap; BS+=gridDim.x)
                {
                    uint128_t_cu temp;
                    mul64(rotpoly_arr[rot_idx[BS * D_BKUSIZE * D_M_MASK + bkun * D_M_MASK] + nn], bku_value_im, temp); // 120 bit

                    temp = temp + GadgetMat_arr[nn];

                    // barrett reduction
                    singleBarrett_4q_lazy(temp, q, mu, qbit);

                    rotkey[BS * D_ROTKEY_SIZE + gg * D_POLY_MLWE_DEGREE + nn] = temp.low;
                }
            }
        }
    }
}

__global__ void Extract_Rot_Idx(uint64_t *lwe_cipher, uint64_t *rot_idx_arr)
{
    uint64_t *lwe = lwe_cipher + blockIdx.x * (D_LWE_N + 1) + blockIdx.y * D_BKU_M;

    uint64_t *rot_idx = rot_idx_arr + blockIdx.x * D_BKUSIZE * D_M_MASK + blockIdx.y * D_M_MASK;

    if (blockIdx.y < D_LWE_N / D_BKU_M)
    {
        for (uint32_t im=threadIdx.x+1; im<=D_M_MASK; im+=blockDim.x)
        {
            uint64_t rotidx = 0;
            for (size_t k = 0; k < D_BKU_M; k++)
            {
                if ((im >> (D_BKU_M - 1 - k)) & 1)
                {
                    rotidx += lwe[k];
                }
            }

            rotidx %= 2 * D_POLY_MLWE_DEGREE;

            // select rotpoly
            uint32_t rotidx_N = rotidx % D_POLY_MLWE_DEGREE;
            uint32_t sgn = rotidx >= D_POLY_MLWE_DEGREE ? 1 : 0;

            rot_idx[im-1] = (sgn * D_POLY_MLWE_DEGREE + rotidx_N) * D_COEFF_MODULUS * D_POLY_MLWE_DEGREE;
        }
    }
    else
    {
        uint64_t *lwe_ = lwe_cipher + blockIdx.x * (D_LWE_N + 1) + (D_LWE_N / D_BKU_M) * D_BKU_M;

        for (uint32_t im=threadIdx.x; im<1; im+=blockDim.x)
        {
            uint64_t rotidx = lwe_[blockIdx.y - (D_LWE_N / D_BKU_M)];

            // select rotpoly
            uint32_t rotidx_N = rotidx % D_POLY_MLWE_DEGREE;
            uint32_t sgn = rotidx >= D_POLY_MLWE_DEGREE ? 1 : 0;

            rot_idx[im] = (sgn * D_POLY_MLWE_DEGREE + rotidx_N) * D_COEFF_MODULUS * D_POLY_MLWE_DEGREE;
        }
    }

}

__global__ void Mul_Acc_INTT(uint64_t *rotkey_arr, uint64_t *decntt, uint64_t *testvector_arr, const uint64_t *inv_twiddle_fac)
{
    extern __shared__ uint64_t shared_mem[];

    for (size_t ii=blockIdx.y; ii<(D_MLWE_K+1); ii+=gridDim.y)
    {
        for (size_t jj=blockIdx.z; jj<D_COEFF_MODULUS; jj+=gridDim.z)
        {
            for (size_t nn=threadIdx.x; nn<D_POLY_MLWE_DEGREE; nn+=blockDim.x)
            {
                uint128_t_cu sum;
                register uint64_t q = q_arr[jj];
                for (size_t gg = 0; gg < (D_MLWE_K+1) * D_LGSW; gg++)
                {
                    uint128_t_cu temp;
                    mul64(decntt[blockIdx.x * D_DECNTT_SIZE + jj * D_POLY_MLWE_DEGREE + gg * D_POLY_MLWE_DEGREE * D_COEFF_MODULUS + nn], rotkey_arr[blockIdx.x * D_ROTKEY_SIZE + ii * D_COEFF_MODULUS * D_POLY_MLWE_DEGREE + jj * D_POLY_MLWE_DEGREE + gg * (D_MLWE_K+1) * D_POLY_MLWE_DEGREE * D_COEFF_MODULUS + nn], temp);
                    sum = sum + temp;
                    singleBarrett_4q_lazy(sum, q, mu_4q_arr[jj], qbit_arr[jj]);
                }

                if (sum.low >= 3 * q)
                    sum.low -= 3 * q;
                else if (sum.low >= 2 * q)
                    sum.low -= 2 * q;
                else if (sum.low >= q)
                    sum.low -= q;
                shared_mem[nn] = sum.low;
            }

            __syncthreads();

            INTT_inline(shared_mem, D_POLY_MLWE_DEGREE, inv_twiddle_fac + jj * D_POLY_MLWE_DEGREE, q_arr[jj], qbit_arr[jj], mu_2q_arr[jj]);

            __syncthreads();

            for (size_t i=threadIdx.x; i<D_POLY_MLWE_DEGREE; i+=blockDim.x)
            {
                testvector_arr[blockIdx.x * D_TESTVEC_SIZE + ii * D_COEFF_MODULUS * D_POLY_MLWE_DEGREE + jj * D_POLY_MLWE_DEGREE + i] = shared_mem[i];
            }
        }
    }
}

__global__ void SampleExtractIndex(uint64_t *res_lwe, uint64_t *testvector_arr)
{
    uint32_t res_lwe_size = (D_POLY_DEGREE + 1);
    for (size_t jj = blockIdx.z; jj < D_COEFF_MODULUS; jj+=gridDim.z) 
    {
        for (size_t ii = blockIdx.y; ii < D_MLWE_K; ii+=gridDim.y) 
        {
            for (size_t nn = threadIdx.x; nn < D_POLY_MLWE_DEGREE; nn+=blockDim.x)
            {
                if (nn == 0)
                {
                    res_lwe[blockIdx.x * D_COEFF_MODULUS * res_lwe_size + jj * res_lwe_size + ii * D_POLY_MLWE_DEGREE] = testvector_arr[blockIdx.x * D_TESTVEC_SIZE + (ii + 1) * D_POLY_MLWE_DEGREE * D_COEFF_MODULUS + jj * D_POLY_MLWE_DEGREE ] % q_arr[jj];
                }
                else
                {
                    res_lwe[blockIdx.x * D_COEFF_MODULUS * res_lwe_size + jj * res_lwe_size + ii * D_POLY_MLWE_DEGREE + nn] = q_arr[jj] - testvector_arr[blockIdx.x * D_TESTVEC_SIZE + (ii + 1) * D_POLY_MLWE_DEGREE * D_COEFF_MODULUS + (jj + 1) * D_POLY_MLWE_DEGREE - nn] % q_arr[jj];
                }
            }
        }
        res_lwe[blockIdx.x * D_COEFF_MODULUS * res_lwe_size + jj * res_lwe_size + D_POLY_MLWE_DEGREE * D_MLWE_K] = testvector_arr[blockIdx.x * D_TESTVEC_SIZE + jj * D_POLY_MLWE_DEGREE] % q_arr[jj];
    }
}

__host__ void Bootstrap_CUDA(LWE_gb* lwe_in, std::vector<LWE_gb>* lwe_out, int n_bootstrap, GB_CUDA_Params& gbcudaparam, uint32_t gpu)
{
    cudaSetDevice(gbcudaparam.gpu_id[gpu]);
    int THREAD_SIZE_NTT = (gbcudaparam.poly_mlwe_degree > 2 * MAX_THREAD_SIZE ? MAX_THREAD_SIZE : (gbcudaparam.poly_mlwe_degree >> 1));
    int THREAD_SIZE_N = (gbcudaparam.poly_mlwe_degree > MAX_THREAD_SIZE ? MAX_THREAD_SIZE : gbcudaparam.poly_mlwe_degree);
    uint32_t BKUSIZE = (gbcudaparam.lwe_n / gbcudaparam.BKU_m) + (gbcudaparam.lwe_n % gbcudaparam.BKU_m);
    uint32_t M_MASK = (1 << gbcudaparam.BKU_m) - 1;

    // read input lwes with lwe scaling
    for (size_t i = 0; i < n_bootstrap; ++i)
    {
        LWE_gb lwe_in_scaled(lwe_in[i].n, lwe_in[i].modulus);
        for (size_t nn = 0; nn <= lwe_in[i].n; nn++) {
            lwe_in_scaled.cipher[nn] = (uint64_t)(2 * gbcudaparam.poly_mlwe_degree - (uint128_t)(lwe_in[i].cipher[nn]) * gbcudaparam.poly_mlwe_degree * 2 / lwe_in[i].modulus);
        }
        cudaMemcpyAsync ((void*)d_lwe_cipher[gpu] + i * (gbcudaparam.lwe_n + 1) * sizeof (uint64_t), &(lwe_in_scaled.cipher[0]), (gbcudaparam.lwe_n + 1) * sizeof (uint64_t), cudaMemcpyHostToDevice, streams[gpu * 2 + i & 1]);
    }

    cudaDeviceSynchronize();

    // rotate init value of testvector by lwe[n]
    cudaFuncSetAttribute(rotate_poly, cudaFuncAttributeMaxDynamicSharedMemorySize, gbcudaparam.poly_mlwe_degree * sizeof(uint64_t));
    rotate_poly<<<dim3(n_bootstrap, (gbcudaparam.MLWE_k+1) * gbcudaparam.coeff_modulus_size), THREAD_SIZE_NTT, gbcudaparam.poly_mlwe_degree * sizeof(uint64_t), streams[2 * gpu]>>>(
        d_initvector_arr[gpu],
        d_testvector_arr[gpu],
        d_lwe_cipher[gpu]);

    // precompute indices of rotpoly
    // faster as it release some computational burden of Rotate_BKU()
    Extract_Rot_Idx<<<dim3(n_bootstrap, BKUSIZE), M_MASK, 0, streams[2 * gpu]>>>(
        d_lwe_cipher[gpu],
        d_rot_bku_idx[gpu]);

    for (uint32_t bkun = 0; bkun < BKUSIZE; bkun++) {
        
        cudaStreamSynchronize(streams[2 * gpu]);

        // BSK unrolling with bsk sharing
        // output d_rotkey
        Rotate_BKU<<<dim3(1, (gbcudaparam.MLWE_k+1) * gbcudaparam.Lgsw * (gbcudaparam.MLWE_k+1) * gbcudaparam.coeff_modulus_size), THREAD_SIZE_N, 0, streams[2 * gpu + 1]>>>(
            d_rotkey[gpu],
            d_rotpoly[gpu], d_gadetmat[gpu], d_bsk_bku[gpu],
            d_rot_bku_idx[gpu],
            bkun,
            n_bootstrap);

        // rns
        CRTDecPoly<<<dim3(n_bootstrap, (gbcudaparam.MLWE_k+1)), THREAD_SIZE_N, 0, streams[2 * gpu]>>>(
            d_testvector_arr[gpu],
            d_crtdec[gpu], 
            d_decntt_arr[gpu]);

        // NTT the output of CRTDecPoly
        cudaFuncSetAttribute(NTT_decntt, cudaFuncAttributeMaxDynamicSharedMemorySize, gbcudaparam.poly_mlwe_degree * sizeof(uint64_t));
        NTT_decntt<<<dim3(n_bootstrap, (gbcudaparam.MLWE_k+1) * gbcudaparam.Lgsw, gbcudaparam.coeff_modulus_size), THREAD_SIZE_NTT, gbcudaparam.poly_mlwe_degree * sizeof(uint64_t), streams[2 * gpu]>>>(
            d_decntt_arr[gpu],
            twiddle_factor[gpu]);

        cudaStreamSynchronize(streams[2 * gpu + 1]);
        
        // Accumulate the multiplication of decntt and rotkey, save to testvector after INTT
        cudaFuncSetAttribute(Mul_Acc_INTT, cudaFuncAttributeMaxDynamicSharedMemorySize, gbcudaparam.poly_mlwe_degree * sizeof(uint64_t));
        Mul_Acc_INTT<<<dim3(n_bootstrap, gbcudaparam.MLWE_k+1, gbcudaparam.coeff_modulus_size), THREAD_SIZE_NTT, gbcudaparam.poly_mlwe_degree * sizeof(uint64_t), streams[2 * gpu]>>>(
            d_rotkey[gpu],
            d_decntt_arr[gpu],
            d_testvector_arr[gpu],
            inv_twiddle_factor[gpu]);
    }

    // extract lwe sample from testvector
    SampleExtractIndex<<<dim3(n_bootstrap, gbcudaparam.MLWE_k, gbcudaparam.coeff_modulus_size), THREAD_SIZE_N, 0, streams[2 * gpu]>>>(
        d_res_lwe[gpu],
        d_testvector_arr[gpu]
    );

    cudaDeviceSynchronize();


    // write back to lwe_out
    for (size_t i = 0; i < n_bootstrap; ++i)
    {
        lwe_out[i].resize(gbcudaparam.coeff_modulus_size);
        for (size_t jj = 0; jj < gbcudaparam.coeff_modulus_size; jj++) {
            lwe_out[i][jj].n = gbcudaparam.poly_modulus_degree;
            lwe_out[i][jj].cipher.resize(gbcudaparam.poly_modulus_degree + 1);

            cudaMemcpyAsync (&(lwe_out[i][jj].cipher[0]), (void*)d_res_lwe[gpu] + i * gbcudaparam.coeff_modulus_size * (gbcudaparam.poly_modulus_degree + 1) * sizeof (uint64_t) + jj * (gbcudaparam.poly_modulus_degree + 1) * sizeof (uint64_t), (gbcudaparam.poly_modulus_degree + 1) * sizeof (uint64_t), cudaMemcpyDeviceToHost, streams[2 * gpu + jj & 1]);
        }
    }

    cudaDeviceSynchronize();
}