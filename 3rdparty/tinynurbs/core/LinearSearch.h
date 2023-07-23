#include <emmintrin.h>


int LinearSearch(double *left , double *right ,double val) {
    __m128d val2 = _mm_set1_pd(val);
    //two 64-bit counters:
    __m128i cnt = _mm_setzero_si128();
    for (; left < right; left+=2) {
        __m128d knots = _mm_load_pd(left);
        //compare two knots with val:
        __m128d lessmask = _mm_cmplt_pd(knots , val2);
        //if less , than increment counter:
        __m128i maskI = _mm_castpd_si128(lessmask);
        cnt = _mm_sub_epi32(cnt, maskI);
    }
    //extract sum of counters as 16-bit integer:
    cnt = _mm_add_epi32(cnt, _mm_shuffle_epi32(cnt,_MM_SHUFFLE(0, 1, 2, 3)));
    return _mm_extract_epi16(cnt, 0) & 0xFFFF;
}