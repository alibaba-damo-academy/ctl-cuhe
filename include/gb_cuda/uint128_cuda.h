#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cinttypes>
#include <string>
#include <math.h>

/**
 * @brief 
 * arithmetic library for computing uint128_t on GPUs
 * thanks to https://github.com/ozgunozerk/NTT-Cuda/blob/master/BFV_Scheme/uint128.h
 */
class uint128_t_cu
{
public:
	
	unsigned long long low;
	unsigned long long high;

	__host__ __device__ __forceinline__ uint128_t_cu()
	{
		low = 0;
		high = 0;
	}

	__host__ __device__ __forceinline__ uint128_t_cu(const uint64_t& x)
	{
		low = x;
		high = 0;
	}

	__host__ __device__ __forceinline__ void operator=(const uint128_t_cu& r)
	{
		low = r.low;
		high = r.high;
	}

	__host__ __device__ __forceinline__ void operator=(const uint64_t& r)
	{
		low = r;
		high = 0;
	}

	__host__ __device__ __forceinline__ uint128_t_cu operator<<(const unsigned& shift)
	{
		uint128_t_cu z;

		z.high = high << shift;
		z.high = (low >> (64 - shift)) | z.high;
		z.low = low << shift;

		return z;
	}

	__host__ __device__ __forceinline__ uint128_t_cu operator>>(const unsigned& shift)
	{
		uint128_t_cu z;

		z.low = low >> shift;
		z.low = (high << (64 - shift)) | z.low;
		z.high = high >> shift;

		return z;
	}

	__host__ __device__ __forceinline__ static void shiftr(uint128_t_cu& x, const unsigned& shift)
	{
		x.low = x.low >> shift;
		x.low = (x.high << (64 - shift)) | x.low;
		x.high = x.high >> shift;

	}

	__host__ static uint128_t_cu exp2(const int& e)
	{
		uint128_t_cu z;

		if (e < 64)
			z.low = 1ull << e;
		else
			z.high = 1ull << (e - 64);

		return z;
	}

	__host__ static int log_2(const uint128_t_cu& x)
	{
		int z = 0;

		if (x.high != 0)
			z = log2((double)x.high) + 64;
		else
			z = log2((double)x.low);

		return z;
	}

	__host__ __device__ __forceinline__ static int clz(uint128_t_cu x)
	{
		unsigned cnt = 0;

		if (x.high == 0)
		{
			while (x.low != 0)
			{
				cnt++;
				x.low = x.low >> 1;
			}

			return 128 - cnt;
		}		
		else
		{
			while (x.high != 0)
			{
				cnt++;
				x.high = x.high >> 1;
			}

			return 64 - cnt;
		}
	}

};

__host__ __device__ __forceinline__ static void operator<<=(uint128_t_cu& x, const unsigned& shift)
{
	x.low = x.low >> shift;
	x.low = (x.high << (64 - shift)) | x.low;
	x.high = x.high >> shift;

}

__host__ __device__ __forceinline__ bool operator==(const uint128_t_cu& l, const uint128_t_cu& r)
{
	if ((l.low == r.low) && (l.high == r.high))
		return true;
	else
		return false;
}

__host__ __device__ __forceinline__ bool operator<(const uint128_t_cu& l, const uint128_t_cu& r)
{
	if (l.high < r.high)
		return true;
	else if (l.high > r.high)
		return false;
	else if (l.low < r.low)
		return true;
	else
		return false;
}

__host__ __device__ __forceinline__ bool operator<(const uint128_t_cu& l, const uint64_t& r)
{
	if (l.high != 0)
		return false;
	else if (l.low > r)
		return false;
	else
		return true;
}

__host__ __device__ __forceinline__ bool operator>(const uint128_t_cu& l, const uint128_t_cu& r)
{
	if (l.high > r.high)
		return true;
	else if (l.high < r.high)
		return false;
	else if (l.low > r.low)
		return true;
	else
		return false;
}

__host__ __device__ __forceinline__ bool operator<=(const uint128_t_cu& l, const uint128_t_cu& r)
{
	if (l.high < r.high)
		return true;
	else if (l.high > r.high)
		return false;
	else if (l.low <= r.low)
		return true;
	else
		return false;
}

__host__ __device__ __forceinline__ bool operator>=(const uint128_t_cu& l, const uint128_t_cu& r)
{
	if (l.high > r.high)
		return true;
	else if (l.high < r.high)
		return false;
	else if (l.low >= r.low)
		return true;
	else
		return false;
}

__host__ __device__ __forceinline__ uint128_t_cu operator+(const uint128_t_cu& x, const uint128_t_cu& y)
{
	uint128_t_cu z;

	z.low = x.low + y.low;
	z.high = x.high + y.high + (z.low < x.low);

	return z;
}

__host__ __device__ __forceinline__ uint128_t_cu operator+(const uint128_t_cu& x, const uint64_t& y)
{
	uint128_t_cu z;

	z.low = x.low + y;
	z.high = x.high + (z.low < x.low);

	return z;
}

__host__ __device__ __forceinline__ uint128_t_cu operator-(const uint128_t_cu& x, const uint128_t_cu& y)
{
	uint128_t_cu z;

	z.low = x.low - y.low;
	z.high = x.high - y.high - (x.low < y.low);

	return z;
	
}

__host__ __device__ __forceinline__ void operator-=(uint128_t_cu& x, const uint128_t_cu& y)
{
	x.high = x.high - y.high - (x.low < y.low);
	x.low = x.low - y.low;
}

__host__ __device__ __forceinline__ uint128_t_cu operator-(const uint128_t_cu& x, const uint64_t& y)
{
	uint128_t_cu z;

	z.low = x.low - y;
	z.high = x.high - (x.low < y);

	return z;

}

__host__ __device__ __forceinline__ uint128_t_cu operator/(uint128_t_cu x, const uint64_t& y)
{
	uint128_t_cu z;
	uint128_t_cu ycomp(y);
	uint128_t_cu d(y);

	unsigned shift = uint128_t_cu::clz(d) - uint128_t_cu::clz(x);

	d = d << shift;

	while (shift != 0)
	{
		shift--;
		z = z << 1;
		if (d <= x)
		{
			x = x - d;
			z = z + 1;
		}
		d = d >> 1;
	}

	z = z << 1;
	if (d <= x)
	{
		x = x - d;
		z = z + 1;
	}
	d = d >> 1;

	return z;
}

__host__ __device__ __forceinline__ uint128_t_cu operator%(uint128_t_cu x, const uint64_t& y)
{
	if (x < y)
		return x;

	uint128_t_cu z;
	uint128_t_cu ycomp(y);
	uint128_t_cu d(y);

	unsigned shift = uint128_t_cu::clz(d) - uint128_t_cu::clz(x);

	d = d << shift;

	while (shift != 0)
	{
		shift--;
		z = z << 1;
		if (d <= x)
		{
			x = x - d;
			z = z + 1;
		}
		d = d >> 1;
	}

	z = z << 1;
	if (d <= x)
	{
		x = x - d;
		z = z + 1;
	}
	d = d >> 1;

	return x;
}

__host__ inline static uint128_t_cu host64x2(const uint64_t& x, const uint64_t& y)
{
	uint128_t_cu z;

	uint128_t_cu ux(x);
	uint128_t_cu uy(y);

	int shift = 0;

	// hello elementary school
	while (uy.low != 0)
	{
		if (uy.low & 1)
		{
			if (shift == 0)
				z = z + ux;
			else
				z = z + (ux << shift);
		}

		shift++;

		uint128_t_cu::shiftr(uy, 1);

	}

	return z;
}

/**
 * @brief compute a = a - b
 */
__device__ __forceinline__ void sub128(uint128_t_cu& a, const uint128_t_cu& b)
{
	asm("{\n\t"
		"sub.cc.u64      %1, %3, %5;    \n\t"
		"subc.u64        %0, %2, %4;    \n\t"
		"}"
		: "=l"(a.high), "=l"(a.low)
		: "l"(a.high), "l"(a.low), "l"(b.high), "l"(b.low));
}

/**
 * @brief compute c = a * b
 */
__device__ __forceinline__ void mul64(const unsigned long long& a, const unsigned long long& b, uint128_t_cu& c)
{
	uint4 res;

	asm("{\n\t"
		"mul.lo.u32      %3, %5, %7;    \n\t"
		"mul.hi.u32      %2, %5, %7;    \n\t" //alow * blow
		"mad.lo.cc.u32   %2, %4, %7, %2;\n\t"
		"madc.hi.u32     %1, %4, %7,  0;\n\t" //ahigh * blow
		"mad.lo.cc.u32   %2, %5, %6, %2;\n\t"
		"madc.hi.cc.u32  %1, %5, %6, %1;\n\t" //alow * bhigh
		"madc.hi.u32     %0, %4, %6,  0;\n\t"
		"mad.lo.cc.u32   %1, %4, %6, %1;\n\t" //ahigh * bhigh
		"addc.u32        %0, %0, 0;     \n\t" //add final carry
		"}"
		: "=r"(res.x), "=r"(res.y), "=r"(res.z), "=r"(res.w)
		: "r"((unsigned)(a >> 32)), "r"((unsigned)a), "r"((unsigned)(b >> 32)), "r"((unsigned)b));

	c.high = ((unsigned long long)res.x << 32) + res.y;
	c.low = ((unsigned long long)res.z << 32) + res.w;;
}

/**
 * @brief Lazy Barrett Reduction
 * input must be within [0, 2^124)
 * output [0, 4q)
 * 
 * @param a number to be reduced
 * @param q modulus
 * @param mu 1 << (qbit * 2 + 3)
 * @param qbit bit length of modulus
 */
__device__ __forceinline__ void singleBarrett_4q_lazy(uint128_t_cu& a, uint64_t& q, uint64_t& mu, uint32_t& qbit)
{
    uint64_t rx = a.low >> qbit;

    rx = (a.high << (64 - qbit)) | rx;

    uint128_t_cu temp;

    mul64(rx, mu, temp);

    rx = temp.low >> (qbit + 3);

    rx = (temp.high << (64 - (qbit + 3))) | rx;

    mul64(rx, q, temp);

    sub128(a, temp);
}

/**
 * @brief Barrett Reduction
 * input must be within [0, 2^124)
 * output [0, q)
 * 
 * @param a number to be reduced
 * @param q modulus
 * @param mu 1 << (qbit * 2 + 3)
 * @param qbit bit length of modulus
 */
__device__ __forceinline__ void singleBarrett_4q(uint128_t_cu& a, uint64_t& q, uint64_t& mu, uint32_t& qbit)
{
    singleBarrett_4q_lazy(a, q, mu, qbit);

    if (a.low >= 3 * q)
        a.low -= 3 * q;
    else if (a.low >= 2 * q)
        a.low -= 2 * q;
    else if (a.low >= q)
        a.low -= q;
}

/**
 * @brief Lazy Barrett Reduction
 * input must be within [0, 2^120)
 * output [0, 2q)
 * 
 * @param a number to be reduced
 * @param q modulus
 * @param mu 1 << (qbit * 2)
 * @param qbit bit length of modulus
 */
__device__ __forceinline__ void singleBarrett_2q_lazy(uint128_t_cu& a, uint64_t& q, uint64_t& mu, uint32_t& qbit)
{
    uint64_t rx = a.low >> (qbit - 2);

    rx = (a.high << (64 - (qbit - 2))) | rx;

    uint128_t_cu temp;

    mul64(rx, mu, temp);

    rx = temp.low >> (qbit + 2);

    rx = (temp.high << (64 - (qbit + 2))) | rx;

    mul64(rx, q, temp);

    sub128(a, temp);
}

/**
 * @brief Barrett Reduction
 * input must be within [0, 2^120)
 * output [0, q)
 * 
 * @param a number to be reduced
 * @param q modulus
 * @param mu 1 << (qbit * 2)
 * @param qbit bit length of modulus
 */
__device__ __forceinline__ void singleBarrett_2q(uint128_t_cu& a, uint64_t& q, uint64_t& mu, uint32_t& qbit)
{
    singleBarrett_2q_lazy(a, q, mu, qbit);

    if (a.low >= q)
        a.low -= q;
}