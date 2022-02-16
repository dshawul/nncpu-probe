#ifndef SIMD_H
#define SIMD_H

#if defined(USE_AVX2)
#include <immintrin.h>
#endif

/****************************
* FP32 AVX 
*****************************/
#if defined(USE_FLOAT)

//add
#if defined(USE_AVX2)
template<int offsetRegs>
INLINE void add32(float* p1, const float* p2)
{
    constexpr int lanes = offsetRegs * 8;
    __m256 *a = (__m256 *)(p1 + lanes);
    __m256 *b = (__m256 *)(p2 + lanes);
    a[0] = _mm256_add_ps(a[0], b[0]);
}
#endif

template<int n_input, typename T>
INLINE void addVectors( 
    T* __restrict  p1,
    const T* __restrict  p2)
{
#if !defined(USE_AVX2)
    for(int i = 0; i < n_input; i++)
        p1[i] += p2[i];
#else
    for (int i = 0; i < n_input; i += 32)
    {
        add32<0>(p1, p2);
        add32<1>(p1, p2);
        add32<2>(p1, p2);
        add32<3>(p1, p2);
        p1 += 32;
        p2 += 32;
    }
#endif
}

//subtract
#if defined(USE_AVX2)
template<int offsetRegs>
INLINE void sub32(float* p1, const float* p2)
{
    constexpr int lanes = offsetRegs * 8;
    __m256 *a = (__m256 *)(p1 + lanes);
    __m256 *b = (__m256 *)(p2 + lanes);
    a[0] = _mm256_sub_ps(a[0], b[0]);
}
#endif

template<int n_input, typename T>
INLINE void subVectors( 
    T* __restrict  p1,
    const T* __restrict  p2)
{
#if !defined(USE_AVX2)
    for(int i = 0; i < n_input; i++)
        p1[i] -= p2[i];
#else
    for (int i = 0; i < n_input; i += 32)
    {
        sub32<0>(p1, p2);
        sub32<1>(p1, p2);
        sub32<2>(p1, p2);
        sub32<3>(p1, p2);
        p1 += 32;
        p2 += 32;
    }
#endif
}

//clamp
#if defined(USE_AVX2)
template<int offsetRegs>
INLINE void clamp32(const float* p1, float* p2)
{
    constexpr int lanes = offsetRegs * 8;
    __m256 *a = (__m256 *)(p1 + lanes);
    __m256 *b = (__m256 *)(p2 + lanes);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 maxv = _mm256_set1_ps(SCALE_ACT);
    b[0] = _mm256_min_ps(_mm256_max_ps(a[0], zero), maxv);
}
#endif

template<int n_input, typename IT, typename T>
INLINE void clampVector(
  const IT* __restrict p1,
  T* __restrict p2)
{

#if !defined(USE_AVX2)

#define clamp(a, b, c) ((a) < (b) ? (b) : (a) > (c) ? (c) : (a))
    for(int i = 0; i < n_input; i++)
        p2[i] = clamp(p1[i],0,SCALE_ACT);
#undef clamp

#else
    for (int i = 0; i < n_input; i += 32)
    {
        clamp32<0>(p1, p2);
        clamp32<1>(p1, p2);
        clamp32<2>(p1, p2);
        clamp32<3>(p1, p2);
        p1 += 32;
        p2 += 32;
    }
#endif
}

//dot product
#if defined(USE_AVX2)
template<int offsetRegs>
INLINE __m256 fma32( __m256 acc, const float* p1, const float* p2 )
{
    constexpr int lanes = offsetRegs * 8;
    __m256 *a = (__m256 *)(p1 + lanes);
    __m256 *b = (__m256 *)(p2 + lanes);
    return _mm256_fmadd_ps( a[0], b[0], acc );
}
#endif

template<int n_input>
float dotProduct( const float* p1, const float* p2)
{
#if !defined(USE_AVX2)
    float sum = 0;
    for(int i = 0; i < n_input; i++)
        sum += p1[i] * p2[i];
    return sum;
#else
    __m256 dot0 = _mm256_setzero_ps();
    __m256 dot1 = dot0;
    __m256 dot2 = dot0;
    __m256 dot3 = dot0;

    for(int i = 0; i < n_input; i += 32)
    {
        dot0 = fma32<0>( dot0, p1, p2 );
        dot1 = fma32<1>( dot1, p1, p2 );
        dot2 = fma32<2>( dot2, p1, p2 );
        dot3 = fma32<3>( dot3, p1, p2 );
        p1 += 32;
        p2 += 32;
    }

    const __m256 dot01 = _mm256_add_ps( dot0, dot1 );
    const __m256 dot23 = _mm256_add_ps( dot2, dot3 );
    const __m256 dot0123 = _mm256_add_ps( dot01, dot23 );

    const __m128 r4 = _mm_add_ps( 
        _mm256_castps256_ps128( dot0123 ), _mm256_extractf128_ps( dot0123, 1 ) );
    const __m128 r2 = _mm_add_ps( r4, _mm_movehl_ps( r4, r4 ) );
    const __m128 r1 = _mm_add_ss( r2, _mm_movehdup_ps( r2 ) );
    return _mm_cvtss_f32( r1 );
#endif
}

/****************************
* INT8 AVX 
*****************************/
#else

//add
#if defined(USE_AVX2)
template<int offsetRegs>
INLINE void add8(int16_t* p1, const int16_t* p2)
{
    constexpr int lanes = offsetRegs * 16;
    __m256i *a = (__m256i *)(p1 + lanes);
    __m256i *b = (__m256i *)(p2 + lanes);
    a[0] = _mm256_add_epi16(a[0], b[0]);
}
#endif

template<int n_input, typename T>
INLINE void addVectors( 
    T* __restrict  p1,
    const T* __restrict  p2)
{
#if !defined(USE_AVX2)
    for(int i = 0; i < n_input; i++)
        p1[i] += p2[i];
#else
    for (int i = 0; i < n_input; i += 64)
    {
        add8<0>(p1, p2);
        add8<1>(p1, p2);
        add8<2>(p1, p2);
        add8<3>(p1, p2);
        p1 += 64;
        p2 += 64;
    }
#endif
}

//subtract
#if defined(USE_AVX2)
template<int offsetRegs>
INLINE void sub8(int16_t* p1, const int16_t* p2)
{
    constexpr int lanes = offsetRegs * 16;
    __m256i *a = (__m256i *)(p1 + lanes);
    __m256i *b = (__m256i *)(p2 + lanes);
    a[0] = _mm256_sub_epi16(a[0], b[0]);
}
#endif

template<int n_input, typename T>
INLINE void subVectors( 
    T* __restrict  p1,
    const T* __restrict  p2)
{
#if !defined(USE_AVX2)
    for(int i = 0; i < n_input; i++)
        p1[i] -= p2[i];
#else
    for (int i = 0; i < n_input; i += 64)
    {
        sub8<0>(p1, p2);
        sub8<1>(p1, p2);
        sub8<2>(p1, p2);
        sub8<3>(p1, p2);
        p1 += 64;
        p2 += 64;
    }
#endif
}

//clamp
#if defined(USE_AVX2)
template<int offsetRegs>
INLINE void clamp32to8(const int32_t* p1, int8_t* p2)
{
  constexpr int lanes = offsetRegs * 8 * 4;
  __m256i *a = (__m256i *)(p1 + lanes);
  __m256i *b = (__m256i *)(p2 + lanes);
  const __m256i zero = _mm256_setzero_si256();
  const __m256i control = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
  __m256i r1  = _mm256_srai_epi16(
      _mm256_packs_epi32(a[0],a[1]), SCALE_WEIGHT_SHIFT);
  __m256i r2  = _mm256_srai_epi16(
      _mm256_packs_epi32(a[2],a[3]), SCALE_WEIGHT_SHIFT);
  b[0] = _mm256_permutevar8x32_epi32(
      _mm256_max_epi8(_mm256_packs_epi16(r1, r2), zero), control);
}
#endif

template<int n_input, typename IT, typename T>
INLINE void clampVector (
  const int32_t* __restrict p1,
  T* __restrict p2)
{
#if !defined(USE_AVX2)

#define clamp(a, b, c) ((a) < (b) ? (b) : (a) > (c) ? (c) : (a))
    for(int i = 0; i < n_input; i++)
        p2[i] = clamp((p1[i] >> SCALE_WEIGHT_SHIFT),0,SCALE_ACT);
#undef clamp

#else
    for (int i = 0; i < n_input; i += 32)
    {
        clamp32to8<0>(p1, p2);
        p1 += 32;
        p2 += 32;
    }
#endif
}

//clamp for input layer does no shift
#if defined(USE_AVX2)
template<int offsetRegs>
INLINE void clamp16to8(const int16_t* p1, int8_t* p2)
{
  constexpr int lanes = offsetRegs * 16 * 2;
  constexpr int control = 0b11011000;
  const __m256i zero = _mm256_setzero_si256();
  __m256i *a = (__m256i *)(p1 + lanes);
  __m256i *b = (__m256i *)(p2 + lanes);
#ifdef STOCK
  __m256i a0 = a[0];
  __m256i a1 = a[1];
#else
  __m256i a0 = _mm256_srai_epi16(a[0], SCALE_WEIGHT_SHIFT - 1);
  __m256i a1 = _mm256_srai_epi16(a[1], SCALE_WEIGHT_SHIFT - 1);
#endif
  b[0] =  _mm256_permute4x64_epi64(_mm256_max_epi8(
      _mm256_packs_epi16(a0, a1), zero), control);
}
#endif

template<int n_input, typename IT, typename T>
INLINE void clampVector (
  const int16_t* __restrict p1,
  T* __restrict p2)
{
#if !defined(USE_AVX2)

#define clamp(a, b, c) ((a) < (b) ? (b) : (a) > (c) ? (c) : (a))
    for(int i = 0; i < n_input; i++)
#ifdef STOCK
        p2[i] = clamp(p1[i],0,SCALE_ACT);
#else
        p2[i] = clamp((p1[i] >> SCALE_WEIGHT_SHIFT - 1),0,SCALE_ACT);
#endif
#undef clamp

#else
    for (int i = 0; i < n_input; i += 64)
    {
        clamp16to8<0>(p1, p2);
        clamp16to8<1>(p1, p2);
        p1 += 64;
        p2 += 64;
    }
#endif
}

//dot product
#if defined(USE_AVX2)
template<int offsetRegs>
INLINE __m256i fma8( const int8_t* p1, const int8_t* p2 )
{
    constexpr int lanes = offsetRegs * 32;
    __m256i *a = (__m256i *)(p1 + lanes);
    __m256i *b = (__m256i *)(p2 + lanes);
    __m256i prod = _mm256_maddubs_epi16(a[0], b[0]);
    prod = _mm256_madd_epi16(prod, _mm256_set1_epi16(1));
    return prod;
}
#endif

template<int n_input>
int32_t dotProduct( const int8_t* p1, const int8_t* p2)
{
#if !defined(USE_AVX2)
    int32_t sum = 0;
    for(int i = 0; i < n_input; i++)
        sum += p1[i] * p2[i];
    return sum;
#else
    __m256i prod = _mm256_setzero_si256();
    for(int i = 0; i < n_input; i += 32)
    {
        prod = _mm256_add_epi32(prod, fma8<0>( p1, p2 ));
        p1 += 32;
        p2 += 32;
    }
    __m128i sum = _mm_add_epi32(
        _mm256_castsi256_si128(prod), _mm256_extracti128_si256(prod, 1));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x1b));
    return _mm_cvtsi128_si32(sum) + _mm_extract_epi32(sum, 1);
#endif
}

#endif
/****************************
* END AVX 
*****************************/

#endif