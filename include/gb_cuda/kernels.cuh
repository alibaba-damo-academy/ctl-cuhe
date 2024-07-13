#pragma once

#include <stdio.h>
#include <cstdint>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include <random>
#include <thread>
#include <bitset>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gb_cuda.h"

//constants for context
extern __device__ __constant__ uint32_t D_RNS_QWORD, D_ROTKEY_SIZE, D_DECNTT_SIZE, D_TESTVEC_SIZE, D_N_LWE, D_BKUSIZE, D_M_MASK, D_RNS_COEFF_MODULUS, D_COEFF_MODULUS, D_POLY_DEGREE, D_POLY_MLWE_DEGREE, D_LWE_N, D_BGBIT, D_BKU_M, D_MLWE_K, D_LGSW;

// modulus and constants for barrett reduction
extern __device__ __constant__ uint64_t q_arr[MAX_RNS_MODULUS];
extern __device__ __constant__ uint64_t mu_4q_arr[MAX_RNS_MODULUS];
extern __device__ __constant__ uint64_t mu_2q_arr[MAX_RNS_MODULUS];
extern __device__ __constant__ uint32_t qbit_arr[MAX_RNS_MODULUS];

// rns constants
extern __device__ __constant__ uint64_t rns_q_inv[MAX_RNS_MODULUS];
extern __device__ __constant__ uint64_t rns_q_rst[MAX_RNS_MODULUS * MAX_RNS_QWORD];

// ntt twiddle factor
extern vector<uint64_t*> twiddle_factor, inv_twiddle_factor;

// Bootstrapping buffers
extern vector<uint64_t*> d_decntt_arr, d_testvector_arr, d_initvector_arr; // buffers
extern vector<uint64_t*> d_lwe_cipher, d_res_lwe; // lwe samples
extern vector<uint64_t*> d_crtdec; // rns decompose
extern vector<uint64_t*> d_bsk_bku, d_rotkey, d_gadetmat, d_rotpoly, d_rot_bku_idx; // bku

// PackLWEs buffers
extern uint64_t *d_rlwes, *d_pack_buffer, *d_prod, *d_rlwe_buf_1, *d_rlwe_buf_2, *d_rlwe_buf_3, *d_lwes; // buffers
extern __constant__ uint64_t N_inv_q_arr[MAX_RNS_MODULUS]; // N^{-1} mod q
extern __constant__ uint64_t modswitch_arr[MAX_RNS_MODULUS]; // constants for modswitch
extern uint64_t *d_ksk;

/**
 * @brief throughout the deisgn, there are two streams for each GPU
 *        used in multistream execution in Bootstrapping
 *        used in asynchronous memeory transfer
 */
extern vector<cudaStream_t> streams;

/**
 * @brief Pre-compute polynomials for bku rotation
 *        merged operations: rotpoly += 1, NTT(rotpoly), etc.
 */
__host__ void set_rotpoly(uint32_t gpu, uint64_t *rotpoly, const uint64_t *twiddle_fac, GB_CUDA_Params& gbcudaparam);

/**
 * @brief Bootstrap lwes
 *        the number of parallel computed lwes is set by n_bootstrap
 */
__host__ void Bootstrap_CUDA(LWE_gb* lwe_in, vector<LWE_gb>* lwe_out, int n_bootstrap, GB_CUDA_Params& gbcudaparam, uint32_t gpu);

/**
 * @brief ReduceTwo lwes in Packing
 *        executed in a forward loop
 */
__host__ void ReduceTwo_CUDA(vector<vector<LWE_gb>> &lwes_zp, Ciphertext& rlwe_out, uint32_t n_lwe, GB_CUDA_Params& gbcudaparam);

/**
 * @brief rlwe_out = rlwe_in_1 + rlwe_in_2 mod q
 */
__host__ void add_CUDA(Ciphertext& rlwe_in_1, Ciphertext& rlwe_in_2, Ciphertext& rlwe_out, GB_CUDA_Params& gbcudaparam);

/**
 * @brief Automorphism
 */
__host__ void Automorph_CUDA(Ciphertext& rlwe_out, uint64_t galois_elt, uint32_t n_lwe, GB_CUDA_Params& gbcudaparam);