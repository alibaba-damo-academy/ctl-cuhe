#include "kernels.cuh"
#include "uint128_cuda.h"
#include "NTT.h"

__global__ void rotate_poly(uint64_t *rlwe_1, uint64_t *rlwe_2, uint32_t rotate, uint32_t stride)
{
    for (uint32_t idx = blockIdx.x; idx < D_N_LWE / stride; idx += gridDim.x)
    {
        for (uint32_t ii=blockIdx.y; ii<2 * D_COEFF_MODULUS; ii+=gridDim.y)
        {
            uint64_t *rlwe_in = rlwe_1 + ii * D_POLY_DEGREE + idx * stride * 2 * D_COEFF_MODULUS * D_POLY_DEGREE;
            uint64_t *rlwe_out = rlwe_2 + ii * D_POLY_DEGREE + idx * 2 * D_COEFF_MODULUS * D_POLY_DEGREE;
            uint64_t q = q_arr[ii % D_COEFF_MODULUS];

            for (size_t nn=threadIdx.x; nn<D_POLY_DEGREE; nn+=blockDim.x)
            {
                uint64_t index_raw = rotate + nn;
                uint64_t index = index_raw & (uint64_t)(D_POLY_DEGREE - 1);
                uint64_t poly = rlwe_in[nn];
                uint64_t *target_result = &rlwe_out[index];
                if (!(index_raw & (uint64_t)(D_POLY_DEGREE)) || !poly)
                {
                    *target_result = poly;
                }
                else
                {
                    *target_result = q - poly;
                }
            }
        }
    }
}

// not the standard substract (interfered by stride...)
__global__ void sub(uint64_t *rlwe_1, uint64_t *rlwe_2, uint64_t *rlwe_out, uint32_t stride)
{
for (uint32_t idx = blockIdx.x; idx < D_N_LWE / stride; idx += gridDim.x)
{
    for (uint32_t ii=blockIdx.y; ii<2 * D_COEFF_MODULUS; ii+=gridDim.y)
    {
        uint64_t *rlwe_in_1 = rlwe_1 + ii * D_POLY_DEGREE + idx * stride * 2 * D_COEFF_MODULUS * D_POLY_DEGREE;
        uint64_t *rlwe_in_2 = rlwe_2 + ii * D_POLY_DEGREE + idx * 2 * D_COEFF_MODULUS * D_POLY_DEGREE;
        uint64_t *rlwe_out_= rlwe_out + ii * D_POLY_DEGREE + idx * 2 * D_COEFF_MODULUS * D_POLY_DEGREE;
        uint64_t q = q_arr[ii % D_COEFF_MODULUS];

        for (size_t nn=threadIdx.x; nn<D_POLY_DEGREE; nn+=blockDim.x)
        {
            int64_t borrow = (unsigned char)(rlwe_in_1[nn] < rlwe_in_2[nn]);
            rlwe_out_[nn] = (rlwe_in_1[nn] - rlwe_in_2[nn]) + (q & (uint64_t)(-borrow));
        }
    }
}
}

__global__ void add(uint64_t *rlwe_1, uint64_t *rlwe_2, uint64_t *rlwe_out)
{
    for (uint32_t ii=blockIdx.y; ii<2 * D_COEFF_MODULUS; ii+=gridDim.y)
    {
        uint64_t *rlwe_in_1 = rlwe_1 + ii * D_POLY_DEGREE;
        uint64_t *rlwe_in_2 = rlwe_2 + ii * D_POLY_DEGREE;
        uint64_t *rlwe_out_ = rlwe_out + ii * D_POLY_DEGREE;
        uint64_t q = q_arr[ii % D_COEFF_MODULUS];

        for (size_t nn=threadIdx.x; nn<D_POLY_DEGREE; nn+=blockDim.x)
        {
            rlwe_out_[nn] = rlwe_in_1[nn] + rlwe_in_2[nn];
            if (rlwe_out_[nn] > q)
                rlwe_out_[nn] -= q;
        }
    }

    __syncthreads();
}

__global__ void apply_galois(uint64_t *rlwe_1, uint64_t *rlwe_2, uint32_t galois_elt, uint32_t stride)
{
for (uint32_t idx = blockIdx.x; idx < D_N_LWE / stride; idx += gridDim.x)
{
    for (uint32_t ii=blockIdx.y; ii<2; ii+=gridDim.y)
    {
        for (uint32_t jj=blockIdx.z; jj<D_COEFF_MODULUS; jj+=gridDim.z )
        {
            uint64_t *rlwe_in = rlwe_1 + ii * D_COEFF_MODULUS * D_POLY_DEGREE + jj * D_POLY_DEGREE + idx * 2 * D_COEFF_MODULUS * D_POLY_DEGREE;
            uint64_t *rlwe_out = rlwe_2 + ii * D_COEFF_MODULUS * D_POLY_DEGREE + jj * D_POLY_DEGREE + idx * 2 * D_COEFF_MODULUS * D_POLY_DEGREE;
            uint64_t q = q_arr[jj];

            for (size_t nn=threadIdx.x; nn<D_POLY_DEGREE; nn+=blockDim.x)
            {
                uint64_t index_raw = nn * galois_elt;
                uint64_t result_value = rlwe_in[nn];
                if ((index_raw / D_POLY_DEGREE) & 1)
                {
                    int64_t non_zero = (unsigned char)(result_value != 0);
                    result_value = (q - result_value) & (uint64_t)(-non_zero);
                }
                rlwe_out[index_raw % D_POLY_DEGREE] = result_value;
            }
        }
    }

    __syncthreads();

    for (uint32_t ii=blockIdx.y; ii<2; ii+=gridDim.y)
    {
        for (uint32_t jj=blockIdx.z; jj<D_COEFF_MODULUS; jj+=gridDim.z )
        {
            uint64_t *rlwe_in = rlwe_2 + ii * D_COEFF_MODULUS * D_POLY_DEGREE + jj * D_POLY_DEGREE + idx * 2 * D_COEFF_MODULUS * D_POLY_DEGREE;
            uint64_t *rlwe_out = rlwe_1 + ii * D_COEFF_MODULUS * D_POLY_DEGREE + jj * D_POLY_DEGREE + idx * 2 * D_COEFF_MODULUS * D_POLY_DEGREE;

            for (size_t nn=threadIdx.x; nn<D_POLY_DEGREE; nn+=blockDim.x)
            {
                if (ii == 0)
                {
                    rlwe_out[nn] = rlwe_in[nn];
                }
                else if (ii == 1)
                {
                    rlwe_out[nn] = 0;
                }
            }
        }
    }
}
}

__global__ void rns_conversion(uint64_t *rlwe_in, uint64_t *buffer_out, uint32_t stride)
{
for (uint32_t idx = blockIdx.x; idx < D_N_LWE / stride; idx += gridDim.x)
{
    for (size_t J=blockIdx.y; J<D_COEFF_MODULUS; J+=gridDim.y)
    {
        for (size_t I=blockIdx.z; I<D_RNS_COEFF_MODULUS; I+=gridDim.z)
        {
            uint64_t *in = rlwe_in + J * D_POLY_DEGREE + idx * 2 * D_COEFF_MODULUS * D_POLY_DEGREE;
            uint64_t *out = buffer_out + I * D_POLY_DEGREE + J * D_POLY_DEGREE * D_RNS_COEFF_MODULUS + idx * D_COEFF_MODULUS * D_RNS_COEFF_MODULUS * D_POLY_DEGREE;

            if (q_arr[J] > q_arr[I])
            {
                for (size_t nn=threadIdx.x; nn<D_POLY_DEGREE; nn+=blockDim.x)
                {
                    out[nn] = in[nn] % q_arr[I];
                }
            }
            else
            {
                for (size_t nn=threadIdx.x; nn<D_POLY_DEGREE; nn+=blockDim.x)
                {
                    out[nn] = in[nn];
                }
            }
        }
    }
}
}

__global__ void Acc_Mul(uint64_t *buffer, uint64_t *prod, uint64_t *ksk, uint32_t key_index, uint32_t stride)
{
    for (uint32_t idx = blockIdx.x; idx < D_N_LWE / stride; idx += gridDim.x)
    {
        for (size_t I=blockIdx.y; I<D_RNS_COEFF_MODULUS; I+=gridDim.y)
        {
            for (size_t K=blockIdx.z; K<2; K+=gridDim.z)
            {
                for (size_t L=threadIdx.x; L<D_POLY_DEGREE; L+=blockDim.x)
                {
                    uint64_t *prod_val = prod + I * D_POLY_DEGREE + K * D_POLY_DEGREE * D_RNS_COEFF_MODULUS + L + idx * 2 * D_RNS_COEFF_MODULUS * D_POLY_DEGREE;

                    *prod_val = 0;

                    for (size_t J=0; J<D_COEFF_MODULUS; J++)
                    {
                        uint128_t_cu qword = ksk[key_index * D_COEFF_MODULUS * 2 * D_RNS_COEFF_MODULUS * D_POLY_DEGREE + J * 2 * D_RNS_COEFF_MODULUS * D_POLY_DEGREE + K * D_RNS_COEFF_MODULUS * D_POLY_DEGREE + I * D_POLY_DEGREE + L];

                        mul64(qword.low, buffer[(J * D_RNS_COEFF_MODULUS + I) * D_POLY_DEGREE + L + idx * D_COEFF_MODULUS * D_RNS_COEFF_MODULUS * D_POLY_DEGREE], qword); // 120 bit

                        qword = qword + *prod_val;

                        singleBarrett_4q(qword, q_arr[I], mu_4q_arr[I], qbit_arr[I]);

                        *prod_val = qword.low;
                    }
                }
            }
        }
    }
}

__global__ void Add_Mul_ModSwitch(uint64_t *prod, uint64_t *rlwe_add_1, uint64_t *rlwe_out, uint32_t stride)
{
    for (uint32_t idx = blockIdx.x; idx < D_N_LWE / stride; idx += gridDim.x) 
    {
        for (size_t I=blockIdx.y; I<2; I+=gridDim.y)
        {
            for (size_t J=blockIdx.z; J<D_COEFF_MODULUS; J+=gridDim.z)
            {

                // Add (p-1)/2 to change from flooring to rounding.
                uint64_t qk = q_arr[(D_RNS_COEFF_MODULUS - 1)];
                uint64_t qk_half = qk >> 1;

                // (ct mod 4qk) mod qi
                uint64_t qi = q_arr[J];
                uint64_t fix = qi - (qk_half % qi);

                for (size_t L=threadIdx.x; L<D_POLY_DEGREE; L+=blockDim.x)
                {
                    uint64_t temp;
                    uint64_t prod_val = prod[I * D_POLY_DEGREE * D_RNS_COEFF_MODULUS + (D_RNS_COEFF_MODULUS - 1) * D_POLY_DEGREE + L + idx * 2 * D_RNS_COEFF_MODULUS * D_POLY_DEGREE];

                    if (qk > qi)
                    {
                        temp = (prod_val + qk_half) % qk % qi;
                    }
                    else
                    {
                        temp = (prod_val + qk_half) % qk;
                    }

                    temp += fix;

                    uint128_t_cu qword = prod[I * D_POLY_DEGREE * D_RNS_COEFF_MODULUS + J * D_POLY_DEGREE + L + idx * 2 * D_RNS_COEFF_MODULUS * D_POLY_DEGREE];

                    qword.low = qword.low + (qi << 1) - temp; 

                    mul64(qword.low, modswitch_arr[J], qword);

                    qword = qword + rlwe_add_1[I * D_COEFF_MODULUS * D_POLY_DEGREE + J * D_POLY_DEGREE + L + idx * stride * 2 * D_COEFF_MODULUS * D_POLY_DEGREE];

                    singleBarrett_4q(qword, q_arr[J], mu_4q_arr[J], qbit_arr[J]);

                    rlwe_out[I * D_COEFF_MODULUS * D_POLY_DEGREE + J * D_POLY_DEGREE + L + idx * stride * 2 * D_COEFF_MODULUS * D_POLY_DEGREE] = qword.low;
                }
            }
        }
    }
}

__global__ void add_three_rlwes(uint64_t *rlwe_add_1, uint64_t *rlwe_add_2, uint64_t *rlwe_add_3, uint64_t *rlwe_out, uint32_t stride)
{
for (uint32_t idx = blockIdx.x; idx < D_N_LWE / stride; idx += gridDim.x) 
{
    for (size_t I=blockIdx.y; I<2; I+=gridDim.y)
    {
        for (size_t J=blockIdx.z; J<D_COEFF_MODULUS; J+=gridDim.z)
        {
            for (size_t L=threadIdx.x; L<D_POLY_DEGREE; L+=blockDim.x)
            {
                uint64_t temp = rlwe_add_1[I * D_COEFF_MODULUS * D_POLY_DEGREE + J * D_POLY_DEGREE + L + idx * stride * 2 * D_COEFF_MODULUS * D_POLY_DEGREE] + rlwe_add_2[I * D_COEFF_MODULUS * D_POLY_DEGREE + J * D_POLY_DEGREE + L + idx * 2 * D_COEFF_MODULUS * D_POLY_DEGREE];

                if (temp > q_arr[J])
                {
                    temp -= q_arr[J];
                }

                temp += rlwe_add_3[I * D_COEFF_MODULUS * D_POLY_DEGREE + J * D_POLY_DEGREE + L + idx * 2 * D_COEFF_MODULUS * D_POLY_DEGREE];

                if (temp > q_arr[J])
                {
                    temp -= q_arr[J];
                }

                rlwe_out[I * D_COEFF_MODULUS * D_POLY_DEGREE + J * D_POLY_DEGREE + L + idx * stride * 2 * D_COEFF_MODULUS * D_POLY_DEGREE] = temp;
            }
        }
    }
}
}

__host__ void add_CUDA(seal::Ciphertext& rlwe_in_1, seal::Ciphertext& rlwe_in_2, seal::Ciphertext& rlwe_out, GB_CUDA_Params& gbcudaparam)
{
    cudaSetDevice(gbcudaparam.gpu_id[0]);
    int THREAD_SIZE_NTT = (gbcudaparam.poly_modulus_degree > 2 * MAX_THREAD_SIZE ? MAX_THREAD_SIZE : (gbcudaparam.poly_modulus_degree >> 1));
    int THREAD_SIZE_N = (gbcudaparam.poly_modulus_degree > MAX_THREAD_SIZE ? MAX_THREAD_SIZE : gbcudaparam.poly_modulus_degree);

    for (int ii=0; ii<2; ii++)
    {
        cudaMemcpyAsync ((void*)d_rlwe_buf_1 + ii * D_COEFF_MODULUS * D_POLY_DEGREE * sizeof (uint64_t), &rlwe_in_1.data(ii)[0], D_COEFF_MODULUS * D_POLY_DEGREE * sizeof (uint64_t), cudaMemcpyHostToDevice, streams[ii % 2]);
        cudaMemcpyAsync ((void*)d_rlwe_buf_2 + ii * D_COEFF_MODULUS * D_POLY_DEGREE * sizeof (uint64_t), &rlwe_in_2.data(ii)[0], D_COEFF_MODULUS * D_POLY_DEGREE * sizeof (uint64_t), cudaMemcpyHostToDevice, streams[ii % 2]);
    }

    cudaDeviceSynchronize();

    add<<<dim3(1, 2 * D_COEFF_MODULUS), THREAD_SIZE_N, 0, streams[0]>>>(
        d_rlwe_buf_1,
        d_rlwe_buf_2,
        d_rlwe_buf_1
    );

    for (int ii=0; ii<2; ii++)
    {
        cudaMemcpyAsync (&rlwe_out.data(ii)[0], (void*)d_rlwe_buf_1 + ii * D_COEFF_MODULUS * D_POLY_DEGREE * sizeof (uint64_t), D_COEFF_MODULUS * D_POLY_DEGREE * sizeof (uint64_t), cudaMemcpyDeviceToHost, streams[ii % 2]);
    }

    cudaDeviceSynchronize();
}

__global__ void rns_NTT(uint64_t *buffer, const uint64_t *twiddle_fac, uint32_t stride)
{
    extern __shared__ uint64_t shared_mem[];

    for (uint32_t idx = blockIdx.x; idx < D_N_LWE / stride; idx += gridDim.x) 
    {
        for (size_t J=blockIdx.y; J<D_COEFF_MODULUS; J+=gridDim.y)
        {
            for (size_t I=blockIdx.z; I<D_RNS_COEFF_MODULUS; I+=gridDim.z)
            {
                for (size_t i=threadIdx.x; i<D_POLY_DEGREE; i+=blockDim.x)
                {
                    shared_mem[i] = buffer[i + I * D_POLY_DEGREE + J * D_RNS_COEFF_MODULUS *  D_POLY_DEGREE  + idx * D_COEFF_MODULUS * D_RNS_COEFF_MODULUS * D_POLY_DEGREE];
                }

                __syncthreads();

                NTT_inline(shared_mem, D_POLY_DEGREE, twiddle_fac + I * D_POLY_DEGREE, q_arr[I], qbit_arr[I], mu_2q_arr[I]);

                __syncthreads();

                for (size_t i=threadIdx.x; i<D_POLY_DEGREE; i+=blockDim.x)
                {
                    buffer[i + I * D_POLY_DEGREE + J * D_RNS_COEFF_MODULUS *  D_POLY_DEGREE + idx * D_COEFF_MODULUS * D_RNS_COEFF_MODULUS * D_POLY_DEGREE] = shared_mem[i];
                }
            }
        }
    }
}

__global__ void rns_INTT(uint64_t *prod, const uint64_t *inv_twiddle_fac, uint32_t stride)
{
    extern __shared__ uint64_t shared_mem[];

    for (uint32_t idx = blockIdx.x; idx < D_N_LWE / stride; idx += gridDim.x) 
    {
        for (size_t I=blockIdx.y; I<2; I+=gridDim.y)
        {
            for (size_t J=blockIdx.z; J<D_RNS_COEFF_MODULUS; J+=gridDim.z)
            {
                for (size_t i=threadIdx.x; i<D_POLY_DEGREE; i+=blockDim.x)
                {
                    shared_mem[i] = prod[i + J * D_POLY_DEGREE + I * D_RNS_COEFF_MODULUS * D_POLY_DEGREE + idx * 2 * D_RNS_COEFF_MODULUS * D_POLY_DEGREE];
                }

                __syncthreads();

                INTT_inline(shared_mem, D_POLY_DEGREE, inv_twiddle_fac + J * D_POLY_DEGREE, q_arr[J], qbit_arr[J], mu_2q_arr[J]);

                __syncthreads();

                for (size_t i=threadIdx.x; i<D_POLY_DEGREE; i+=blockDim.x)
                {
                    prod[i + J * D_POLY_DEGREE + I * D_RNS_COEFF_MODULUS * D_POLY_DEGREE + idx * 2 * D_RNS_COEFF_MODULUS * D_POLY_DEGREE] = shared_mem[i];
                }
            }
        }
    }
}

__global__ void extract_rlwe(uint64_t *rlwes, uint64_t *lwes)
{
    for (uint32_t idx = blockIdx.x; idx < D_N_LWE; idx += gridDim.x)
    {
        for (uint32_t jj = blockIdx.y; jj < D_COEFF_MODULUS; jj += gridDim.y)
        {
            size_t rev = 0, u = idx;
            for (size_t j = 1; j < D_N_LWE; j *= 2) {
            rev <<= 1;
            rev += u & 1;
            u >>= 1;
            }
            size_t idx_rev = rev;

            uint64_t *lwe_in = lwes + idx_rev * D_COEFF_MODULUS * (D_POLY_DEGREE + 1);
            uint64_t *rlwe_in = rlwes + idx * 2 * D_COEFF_MODULUS * D_POLY_DEGREE;

            // N^{-1} mod q
            uint64_t N_inv_q = N_inv_q_arr[jj];

            for (size_t nn = threadIdx.x; nn < D_POLY_DEGREE; nn += blockDim.x)
            {
                if ( nn == 0 )
                {
                    uint128_t_cu qword;

                    qword = lwe_in[jj * (D_POLY_DEGREE + 1) + nn] % q_arr[jj];
                    mul64(qword.low, N_inv_q, qword);
                    singleBarrett_4q(qword, q_arr[jj], mu_4q_arr[jj], qbit_arr[jj]);
                    rlwe_in[D_COEFF_MODULUS * D_POLY_DEGREE + jj * D_POLY_DEGREE] = qword.low;

                    // const term
                    qword = lwe_in[jj * (D_POLY_DEGREE + 1) + D_POLY_DEGREE] % q_arr[jj];
                    mul64(qword.low, N_inv_q, qword);
                    singleBarrett_4q(qword, q_arr[jj], mu_4q_arr[jj], qbit_arr[jj]);
                    rlwe_in[jj * D_POLY_DEGREE] = qword.low;
                }
                else
                {
                    uint128_t_cu qword;

                    // -(N)^{-1} mod q
                    uint64_t neg_N_inv_q = (q_arr[jj] - N_inv_q) & static_cast<std::uint64_t>(-static_cast<std::int64_t>(N_inv_q != 0));

                    qword = lwe_in[jj * (D_POLY_DEGREE + 1) + nn] % q_arr[jj];
                    mul64(qword.low, neg_N_inv_q, qword);
                    singleBarrett_4q(qword, q_arr[jj], mu_4q_arr[jj], qbit_arr[jj]);
                    rlwe_in[D_COEFF_MODULUS * D_POLY_DEGREE + jj * D_POLY_DEGREE + D_POLY_DEGREE - nn] = qword.low;

                    // const term
                    rlwe_in[jj * D_POLY_DEGREE + nn] = 0;
                }
            }
        }
    }
}

__host__ __forceinline__ void Automorph (uint64_t* rlwe_buf_in, uint64_t* rlwe_buf_1, uint64_t* buffer, uint64_t* prod, uint64_t* rlwes_out, uint64_t* ksk, uint32_t galois, uint32_t n_lwe, uint32_t stride, GB_CUDA_Params& gbcudaparam)
{
    int THREAD_SIZE_NTT = (gbcudaparam.poly_modulus_degree > 2 * MAX_THREAD_SIZE ? MAX_THREAD_SIZE : (gbcudaparam.poly_modulus_degree >> 1));
    int THREAD_SIZE_N = (gbcudaparam.poly_modulus_degree > MAX_THREAD_SIZE ? MAX_THREAD_SIZE : gbcudaparam.poly_modulus_degree);
    uint32_t rns_coeff_modulus_size = gbcudaparam.coeff_modulus_size + 1;

    apply_galois<<<dim3(n_lwe / stride, 2, gbcudaparam.coeff_modulus_size), THREAD_SIZE_N, 0, streams[0]>>>(
        rlwe_buf_in,
        rlwe_buf_1,
        galois,
        stride
    );

    rns_conversion<<<dim3(n_lwe / stride, gbcudaparam.coeff_modulus_size, rns_coeff_modulus_size), THREAD_SIZE_N, 0, streams[0]>>>(
        rlwe_buf_1 + gbcudaparam.coeff_modulus_size * gbcudaparam.poly_modulus_degree,
        buffer,
        stride
    );

    cudaFuncSetAttribute(rns_NTT, cudaFuncAttributeMaxDynamicSharedMemorySize, gbcudaparam.poly_modulus_degree * sizeof(uint64_t));
    rns_NTT<<<dim3(n_lwe / stride, gbcudaparam.coeff_modulus_size, rns_coeff_modulus_size), THREAD_SIZE_NTT, gbcudaparam.poly_modulus_degree * sizeof(uint64_t), streams[0]>>>(
        buffer,
        twiddle_factor[0],
        stride
    );

    Acc_Mul<<<dim3(n_lwe / stride, rns_coeff_modulus_size, 2), THREAD_SIZE_N, 0, streams[0]>>>(
        buffer,
        prod,
        ksk,
        (uint32_t)(std::log2(((galois - 1) >> 1))),
        stride
    );

    cudaFuncSetAttribute(rns_INTT, cudaFuncAttributeMaxDynamicSharedMemorySize, gbcudaparam.poly_modulus_degree * sizeof(uint64_t));
    rns_INTT<<<dim3(n_lwe / stride, 2, rns_coeff_modulus_size), THREAD_SIZE_NTT, gbcudaparam.poly_modulus_degree * sizeof(uint64_t), streams[0]>>>(
        prod,
        inv_twiddle_factor[0],
        stride
    );

    Add_Mul_ModSwitch<<<dim3(n_lwe / stride, 2, gbcudaparam.coeff_modulus_size), THREAD_SIZE_N, 0, streams[0]>>>(
        prod,
        rlwes_out,
        rlwes_out,
        stride
    );
}

__host__ void Automorph_CUDA(seal::Ciphertext& rlwe_out, uint64_t galois_elt, uint32_t n_lwe, GB_CUDA_Params& gbcudaparam)
{
    cudaSetDevice(gbcudaparam.gpu_id[0]);
    int THREAD_SIZE_NTT = (gbcudaparam.poly_modulus_degree > 2 * MAX_THREAD_SIZE ? MAX_THREAD_SIZE : (gbcudaparam.poly_modulus_degree >> 1));
    int THREAD_SIZE_N = (gbcudaparam.poly_modulus_degree > MAX_THREAD_SIZE ? MAX_THREAD_SIZE : gbcudaparam.poly_modulus_degree);

    uint32_t key_index = (uint32_t)(std::log2(((galois_elt - 1) >> 1)));

    for (int ii=0; ii<2; ii++)
    {
        cudaMemcpyAsync ((void*)d_rlwe_buf_1 + ii * gbcudaparam.coeff_modulus_size * gbcudaparam.poly_modulus_degree * sizeof (uint64_t), &rlwe_out.data(ii)[0], gbcudaparam.coeff_modulus_size * gbcudaparam.poly_modulus_degree * sizeof (uint64_t), cudaMemcpyHostToDevice, streams[ii % 2]);
    }

    cudaDeviceSynchronize();

    // when stride = 1, Automorph() just does a single automorphism operation for one rlwe
    Automorph (d_rlwe_buf_1, d_rlwe_buf_3, d_pack_buffer, d_prod, d_rlwes, d_ksk, galois_elt, n_lwe, n_lwe, gbcudaparam);

    cudaDeviceSynchronize();

    for (int ii=0; ii<2; ii++)
    {
        cudaMemcpyAsync (&rlwe_out.data(ii)[0], (void*)d_rlwe_buf_1 + ii * gbcudaparam.coeff_modulus_size * gbcudaparam.poly_modulus_degree * sizeof (uint64_t), gbcudaparam.coeff_modulus_size * gbcudaparam.poly_modulus_degree * sizeof (uint64_t), cudaMemcpyDeviceToHost, streams[ii % 2]);
    }

    cudaDeviceSynchronize();
}

__host__ void ReduceTwo_CUDA(vector<vector<LWE_gb>> &lwes_zp, seal::Ciphertext& rlwe_out, uint32_t n_lwe, GB_CUDA_Params& gbcudaparam)
{
    cudaSetDevice(gbcudaparam.gpu_id[0]);
    int THREAD_SIZE_NTT = (gbcudaparam.poly_modulus_degree > 2 * MAX_THREAD_SIZE ? MAX_THREAD_SIZE : (gbcudaparam.poly_modulus_degree >> 1));
    int THREAD_SIZE_N = (gbcudaparam.poly_modulus_degree > MAX_THREAD_SIZE ? MAX_THREAD_SIZE : gbcudaparam.poly_modulus_degree);

    for (size_t i = 0; i < n_lwe; ++i)
    {
        for (size_t jj = 0; jj < gbcudaparam.coeff_modulus_size; jj++) 
        {
            cudaMemcpyAsync ((void*)d_lwes + i * gbcudaparam.coeff_modulus_size * (gbcudaparam.poly_modulus_degree + 1) * sizeof (uint64_t) + jj * (gbcudaparam.poly_modulus_degree + 1) * sizeof (uint64_t), &(lwes_zp[i][jj].cipher[0]), (gbcudaparam.poly_modulus_degree + 1) * sizeof (uint64_t), cudaMemcpyHostToDevice, streams[i % 2]);
        }
    }

    cudaDeviceSynchronize();

    // extract rlwes from lwes
    // with bit reversed index
    // merged some pre-scaling
    extract_rlwe<<<dim3(n_lwe, gbcudaparam.coeff_modulus_size), THREAD_SIZE_N, 0, streams[0]>>>(
        d_rlwes,
        d_lwes
    );

    for (size_t stride = 2; stride <= n_lwe; stride *= 2) 
    {
        // d_rlwe_buf_1 = X^{N/stride} * d_rlwes[odd]
        // d_rlwe_buf_1 = X^{N/stride} * ct_odd
        rotate_poly<<<dim3(n_lwe / stride, 2 * gbcudaparam.coeff_modulus_size), THREAD_SIZE_N, 0, streams[0]>>>(
            d_rlwes + (stride / 2) * 2 * gbcudaparam.coeff_modulus_size * gbcudaparam.poly_modulus_degree, // ct_odd
            d_rlwe_buf_1,
            gbcudaparam.poly_modulus_degree / stride,
            stride
        );

        // d_rlwe_buf_2 = d_rlwes[even] - d_rlwe_buf_1
        // d_rlwe_buf_2 = ct_even - X^{N/stride} * ct_odd
        sub<<<dim3(n_lwe / stride, 2 * gbcudaparam.coeff_modulus_size), THREAD_SIZE_N, 0, streams[0]>>>(
            d_rlwes, // ct_even
            d_rlwe_buf_1,
            d_rlwe_buf_2,
            stride
        );

        // d_rlwes = Automorph(d_rlwe_buf_2, stride + 1)
        // d_rlwes = Automorph(ct_even - X^{N/stride} * ct_odd, stride + 1)
        Automorph (d_rlwe_buf_2, d_rlwe_buf_3, d_pack_buffer, d_prod, d_rlwes, d_ksk, stride + 1, n_lwe, stride, gbcudaparam);

        // d_rlwes = d_rlwe_buf_2 + d_rlwe_buf_1 + d_rlwes; NOTE: d_rlwe_buf_2 is modified in apply_galois()
        // d_rlwes = apply_galois(ct_even) + X^{N/stride} * ct_odd + Automorph(ct_even - X^{N/stride} * ct_odd, stride + 1)
        add_three_rlwes<<<dim3(n_lwe / stride, 2, gbcudaparam.coeff_modulus_size), THREAD_SIZE_N, 0, streams[0]>>>(
            d_rlwes,
            d_rlwe_buf_1,
            d_rlwe_buf_2,
            d_rlwes,
            stride
        );
    }

    cudaDeviceSynchronize();

    // assume rlwe_out.data(0) and rlwe_out.data(1) are continuous!, otherwise fails
    cudaMemcpyAsync (&rlwe_out.data(0)[0], (void*)d_rlwes, 2 * gbcudaparam.coeff_modulus_size * gbcudaparam.poly_modulus_degree * sizeof (uint64_t), cudaMemcpyDeviceToHost, streams[0]);

    cudaDeviceSynchronize();
}