#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "uint128_cuda.h"

#define SHUFFLE_SIZE 32 // warp shuffle optimization

/**
 * @brief NTT transform a_arr
 * 
 * @param a_arr input array
 * @param N size of a_arr
 * @param twiddle_fac twiddle factor 
 * @param q modulus
 * @param qbit bit length of q
 * @param mu 1 << (qbit * 2)
 */
__device__ __forceinline__ void NTT_inline(uint64_t *a_arr, uint32_t N, const uint64_t *twiddle_fac, uint64_t& q, uint32_t& qbit, uint64_t& mu)
{
    for (size_t m = 1; m < N / 2 / SHUFFLE_SIZE; m *= 2)
    {
        for (size_t i=threadIdx.x; i<N/2; i+=blockDim.x)
        {
            int gap = (N / m) / 2;
            int tf_step = i / gap;
            int target_index = tf_step * gap * 2 + i % gap;

            uint64_t tf = twiddle_fac[m + tf_step];

            uint64_t u = a_arr[target_index];

            uint128_t_cu temp;

            mul64(a_arr[target_index + gap], tf, temp);

            singleBarrett_2q(temp, q, mu, qbit);

            uint64_t v = temp.low;

            uint64_t target_result = u + v;

            target_result -= q * (target_result >= q); // target_result mod q

            a_arr[target_index] = target_result;

            u += q * (u < v); // u mod q

            a_arr[target_index + gap] = u - v;

            __syncthreads();
        }
    }

    for (size_t i=threadIdx.x; i<N/2; i+=blockDim.x)
    {
        // gap = SHUFFLE_SIZE here

        int gt = i;
        int ti = gt / SHUFFLE_SIZE * SHUFFLE_SIZE * 2 + gt % SHUFFLE_SIZE;

        uint64_t A = a_arr[ti];
        uint64_t B = a_arr[ti + SHUFFLE_SIZE];

        for (size_t m = N / 2 / SHUFFLE_SIZE; m < N / 2; m *= 2)
        {
            int gap = (N / m) / 2;
            int tf_step = i / gap;
            int target_index = tf_step * gap * 2 + i % gap;

            uint64_t tf = twiddle_fac[m + tf_step];

            uint64_t u = A;

            uint128_t_cu temp;

            mul64(B, tf, temp);

            singleBarrett_2q(temp, q, mu, qbit);

            uint64_t v = temp.low;

            uint64_t target_result = u + v;

            target_result -= q * (target_result >= q); // target_result mod q

            u += q * (u < v); // u mod q

            A = target_result;
            B = u - v;

            if ((i % gap) >= (gap >> 1))
            {
                uint64_t C = A;
                A = B;
                B = C;
            }

            B = __shfl_xor_sync(-1, B, gap >> 1);

            if ((i % gap) >= (gap >> 1))
            {
                uint64_t C = A;
                A = B;
                B = C;
            }

            // __syncthreads();
        }

        uint64_t u = A;

        uint128_t_cu temp;

        mul64(B, twiddle_fac[N / 2 + i], temp);

        singleBarrett_2q(temp, q, mu, qbit);

        uint64_t v = temp.low;

        uint64_t target_result = u + v;

        target_result -= q * (target_result >= q); // target_result mod q

        u += q * (u < v); // u mod q

        a_arr[2*i] = target_result;
        a_arr[2*i + 1] = u - v;

        // __syncthreads();
    }
}

/**
 * @brief INTT transform a_arr
 * 
 * @param a_arr input array
 * @param N size of a_arr
 * @param inv_twiddle_fac inverse twiddle factor
 * @param q modulus
 * @param qbit bit length of q
 * @param mu 1 << (qbit * 2)
 */
__device__ __forceinline__ void INTT_inline(uint64_t *a_arr, uint32_t N, const uint64_t *inv_twiddle_fac, uint64_t& q, uint32_t& qbit, uint64_t& mu)
{
    uint64_t q2 = (q + 1) >> 1;

    for (size_t i=threadIdx.x; i<N/2; i+=blockDim.x)
    {
        // gap = 1 here
        uint64_t A = a_arr[2*i];
        uint64_t B = a_arr[2*i + 1];
        
        for (size_t m = (N / 2); m >= N / 2 / 16; m /= 2)
        {
            int gap = (N / m) / 2;
            int tf_step = i / gap;
            int target_index = tf_step * gap * 2 + i % gap;

            uint64_t inv_tf = inv_twiddle_fac[m + tf_step];

            uint64_t u = A;
            uint64_t v = B;

            uint64_t target_result = u + v;
            
            target_result -= q * (target_result >= q);

            A = (target_result >> 1) + q2 * (target_result & 1);

            u += q * (u < v);

            uint128_t_cu temp;

            mul64(u - v, inv_tf, temp);

            singleBarrett_2q(temp, q, mu, qbit);

            uint64_t temp_low = temp.low;

            B = (temp_low >> 1) + q2 * (temp_low & 1);

            if ((i % (gap << 1)) >= (gap))
            {
                uint64_t C = A;
                A = B;
                B = C;
            }

            B = __shfl_xor_sync(-1, B, gap);

            if ((i % (gap << 1)) >= (gap))
            {
                uint64_t C = A;
                A = B;
                B = C;
            }

            // __syncthreads();
        }

        int tf_step = i / SHUFFLE_SIZE;
        int target_index = tf_step * SHUFFLE_SIZE * 2 + i % SHUFFLE_SIZE;

        uint64_t inv_tf = inv_twiddle_fac[(N / 2 / SHUFFLE_SIZE) + tf_step];

        uint64_t u = A;
        uint64_t v = B;

        uint64_t target_result = u + v;
        
        target_result -= q * (target_result >= q);

        a_arr[target_index] = (target_result >> 1) + q2 * (target_result & 1);

        u += q * (u < v);

        uint128_t_cu temp;

        mul64(u - v, inv_tf, temp);

        singleBarrett_2q(temp, q, mu, qbit);

        uint64_t temp_low = temp.low;

        a_arr[target_index + SHUFFLE_SIZE] = (temp_low >> 1) + q2 * (temp_low & 1);
    }

    __syncthreads();

    for (size_t m = (N / 2 / SHUFFLE_SIZE / 2); m >= 1; m /= 2)
    {
        for (size_t i=threadIdx.x; i<N/2; i+=blockDim.x)
        {
            int gap = (N / m) / 2;
            int tf_step = i / gap;
            int target_index = tf_step * gap * 2 + i % gap;

            uint64_t inv_tf = inv_twiddle_fac[m + tf_step];

            uint64_t u = a_arr[target_index];
            uint64_t v = a_arr[target_index + gap];

            uint64_t target_result = u + v;

            target_result -= q * (target_result >= q);

            a_arr[target_index] = (target_result >> 1) + q2 * (target_result & 1);

            u += q * (u < v);

            uint128_t_cu temp;

            mul64(u - v, inv_tf, temp);

            singleBarrett_2q(temp, q, mu, qbit);

            uint64_t temp_low = temp.low;

            a_arr[target_index + gap] = (temp_low >> 1) + q2 * (temp_low & 1);
        }

        __syncthreads();
    }
}