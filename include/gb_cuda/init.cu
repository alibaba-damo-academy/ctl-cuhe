#include "kernels.cuh"
#include "uint128_cuda.h"
#include "NTT.h"

__device__ __constant__ uint32_t D_RNS_QWORD, D_ROTKEY_SIZE, D_DECNTT_SIZE, D_TESTVEC_SIZE, D_N_LWE, D_BKUSIZE, D_M_MASK, D_RNS_COEFF_MODULUS, D_COEFF_MODULUS, D_POLY_DEGREE, D_POLY_MLWE_DEGREE, D_LWE_N, D_BGBIT, D_BKU_M, D_MLWE_K, D_LGSW;

__device__ __constant__ uint64_t q_arr[MAX_RNS_MODULUS];
__device__ __constant__ uint64_t mu_4q_arr[MAX_RNS_MODULUS];
__device__ __constant__ uint64_t mu_2q_arr[MAX_RNS_MODULUS];
__device__ __constant__ uint32_t qbit_arr[MAX_RNS_MODULUS];
__device__ __constant__ uint64_t rns_q_inv[MAX_RNS_MODULUS];
__device__ __constant__ uint64_t rns_q_rst[MAX_RNS_MODULUS * MAX_RNS_QWORD];
__device__ __constant__ uint64_t N_inv_q_arr[MAX_RNS_MODULUS];
__device__ __constant__ uint64_t modswitch_arr[MAX_RNS_MODULUS];

vector<uint64_t*> twiddle_factor, inv_twiddle_factor;
vector<uint64_t*> d_decntt_arr, d_testvector_arr, d_initvector_arr;
vector<uint64_t*> d_lwe_cipher, d_res_lwe;
vector<uint64_t*> d_crtdec;
vector<uint64_t*> d_bsk_bku, d_rotkey, d_gadetmat, d_rotpoly, d_rot_bku_idx;

uint64_t *d_rlwes, *d_pack_buffer, *d_prod, *d_rlwe_buf_1, *d_rlwe_buf_2, *d_rlwe_buf_3, *d_lwes, *d_ksk;

vector<cudaStream_t> streams;

__global__ void set_rotpoly_(uint64_t *rotpoly)
{
    for (size_t ss=0; ss<2; ss++)
    {
        for (size_t idx=0; idx<D_POLY_MLWE_DEGREE; idx++)
        {
            uint64_t *poly = rotpoly + (ss * D_POLY_MLWE_DEGREE + idx) * D_COEFF_MODULUS * D_POLY_MLWE_DEGREE;

            for (size_t jj=0; jj<D_COEFF_MODULUS; jj++)
            {
                poly[jj * D_POLY_MLWE_DEGREE + idx] = (ss == 1 ? q_arr[jj] - 1 : 1);
                if (idx == 0)
                    poly[jj * D_POLY_MLWE_DEGREE] -= 1;
                else
                    poly[jj * D_POLY_MLWE_DEGREE] = q_arr[jj] - 1;
            }
        }
    }
}

__global__ void NTT_rotpoly_(uint64_t *rotpoly, const uint64_t *twiddle_fac)
{
    extern __shared__ uint64_t shared_mem[];

    for (size_t ii=blockIdx.y; ii<D_POLY_MLWE_DEGREE * 2; ii+=gridDim.y)
    {
        for (size_t jj=blockIdx.z; jj<D_COEFF_MODULUS; jj+=gridDim.z)
        {
            for (size_t i=threadIdx.x; i<D_POLY_MLWE_DEGREE; i+=blockDim.x)
            {
                shared_mem[i] = rotpoly[(ii * D_COEFF_MODULUS + jj) * D_POLY_MLWE_DEGREE + i];
            }

            __syncthreads();

            NTT_inline(shared_mem, D_POLY_MLWE_DEGREE, twiddle_fac + jj * D_POLY_MLWE_DEGREE, q_arr[jj], qbit_arr[jj], mu_2q_arr[jj]);

            __syncthreads();

            for (size_t i=threadIdx.x; i<D_POLY_MLWE_DEGREE; i+=blockDim.x)
            {
                rotpoly[(ii * D_COEFF_MODULUS + jj) * D_POLY_MLWE_DEGREE + i] = shared_mem[i];
            }
        }
    }
}

__host__ void set_rotpoly(uint32_t gpu, uint64_t *rotpoly, const uint64_t *twiddle_fac, GB_CUDA_Params& gbcudaparam)
{
    cudaSetDevice(gbcudaparam.gpu_id[gpu]);
    int THREAD_SIZE_NTT = (gbcudaparam.poly_mlwe_degree > 2 * MAX_THREAD_SIZE ? MAX_THREAD_SIZE : (gbcudaparam.poly_mlwe_degree >> 1));
    int THREAD_SIZE_N = (gbcudaparam.poly_mlwe_degree > MAX_THREAD_SIZE ? MAX_THREAD_SIZE : gbcudaparam.poly_mlwe_degree);

    set_rotpoly_<<<1, 1, 0, streams[gpu * 2]>>>(rotpoly);
    cudaFuncSetAttribute(NTT_rotpoly_, cudaFuncAttributeMaxDynamicSharedMemorySize, gbcudaparam.poly_mlwe_degree * sizeof(uint64_t));
    NTT_rotpoly_<<<dim3(1, gbcudaparam.poly_mlwe_degree * 2, gbcudaparam.coeff_modulus_size), THREAD_SIZE_NTT, gbcudaparam.poly_mlwe_degree * sizeof(uint64_t), streams[gpu * 2]>>>(rotpoly, twiddle_fac);
}