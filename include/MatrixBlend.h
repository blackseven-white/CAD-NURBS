#include <emmintrin.h>


void MatrixBlend(int rows , int cols , double *matrix , int rowStride , double *rowMult , double *colMult , double *res) {
    __m128d Rxy, Rzw; //4D point accumulator
    Rxy = Rzw = _mm_setzero_pd();
    for (int i = 0; i<rows; i++) {
        //load i-th row coefficient into both halves
        __m128d ai = _mm_load1_pd(rowMult + i);
        for (int j = 0; j<cols; j+=2) {     
            //load two 4D points into 4 xmm registers
            __m128d M0xy = _mm_load_pd(matrix + 4*j+0);
            __m128d M0zw = _mm_load_pd(matrix + 4*j+2);
            __m128d M1xy = _mm_load_pd(matrix + 4*j+4);
            __m128d M1zw = _mm_load_pd(matrix + 4*j+6);
            //load two column coefficients
            __m128d bj = _mm_load_pd(colMult + j);
            //obtain two full point coefficients
            __m128d coeff = _mm_mul_pd(ai, bj);
            //split them into two registers
            __m128d c0 = _mm_unpacklo_pd(coeff , coeff);
            __m128d c1 = _mm_unpackhi_pd(coeff , coeff);
            //multiply points on coeff -s and accumulate
            Rxy = _mm_add_pd(Rxy, _mm_mul_pd(c0, M0xy));
            Rzw = _mm_add_pd(Rzw, _mm_mul_pd(c0, M0zw));
            Rxy = _mm_add_pd(Rxy, _mm_mul_pd(c1, M1xy));
            Rzw = _mm_add_pd(Rzw, _mm_mul_pd(c1, M1zw));
        }
        //move to next row
        matrix += rowStride; 
    }
    //save 4D point result
    _mm_store_pd(res+0, Rxy);
    _mm_store_pd(res+2, Rzw);
}