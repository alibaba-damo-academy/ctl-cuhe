#pragma once
#include <cmath>
#include <cstdint>
struct domainP {
    using T = uint64_t;
    static constexpr size_t nbit_ = 10;
    static constexpr size_t n_ = 1 << nbit_;
    static constexpr size_t digits = 32;
    // static constexpr size_t mbit_ = 0;
    static constexpr size_t m_ = 3;
    static constexpr size_t mmask_ = (1 << m_) - 1;
    static constexpr size_t ksize_ = (n_ / m_) * ((1 << m_) - 1) + (n_ % m_);
    static constexpr size_t bkusize_ = (n_ / m_) + (n_ % m_);
    static const inline long double stddev = std::pow(2.0, -25);
    static const inline long double maxdev = 3 * std::pow(2.0, -25);
    static constexpr uint64_t mu = 1ULL << 4;
};

struct targetP {
    using T = uint64_t;
    static constexpr size_t nbit_ = 13;
    static constexpr size_t n_ = 1 << nbit_;
    static constexpr size_t l_ = 5; // lvl2 param
    static constexpr size_t n_modulus_ = 3;
    static constexpr size_t k_ = 2;
    static constexpr size_t N_ = (n_ / k_);
    static constexpr size_t digits = 144; // uint64_t
    static constexpr size_t Bgbit = 23;   // lvl2 param
    static constexpr uint64_t Bg = 1 << Bgbit;
    static const inline long double stddev = std::pow(2.0, -82);
    static const inline long double maxdev = 3 * std::pow(2.0, -82);
    static constexpr uint64_t mu = 1ULL << 40;
};

struct tdP // lvl22param
{
    using tP = domainP;
    using dP = targetP;
    static constexpr size_t t = 8;
    static constexpr size_t baseBit = 4;
};

struct GateBootStrap_targetP {
    using T = uint64_t;
    static constexpr size_t nbit_ = 13;
    static constexpr size_t n_ = 1 << nbit_;
    static constexpr size_t l_ = 5; // lvl2 param
    static constexpr size_t k_ = 4;
    static constexpr size_t N_ = (n_ / k_);
    static constexpr size_t digits = 32; // uint64_t
    static constexpr size_t Bgbit = 23;  // lvl2 param
    static constexpr uint64_t Bg = 1 << Bgbit;
    static const inline long double stddev = std::pow(2.0, -82);
    static const inline long double maxdev = 3 * std::pow(2.0, -82);
    static constexpr uint64_t mu = 1ULL << 40;
};