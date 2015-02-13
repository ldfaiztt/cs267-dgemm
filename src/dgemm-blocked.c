#include <mmintrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <emmintrin.h>
#include <string.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#define BLOCK_L1 256
#define BLOCK_L2 512

#define turn_even(x) (((x) & 1) ? (x+1) : (x))
#define min(a,b) (((a)<(b))?(a):(b))

#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)

#define ARRAY(A,i,j) (A)[(j)*lda + (i)]

static inline void calc_4x4(int lda, int K, double* a, double* b, double* c)
{
  __m128d a0x_1x, a2x_3x, 
    bx0, bx1, bx2, bx3, 
    c00_10, c20_30, 
    c01_11, c21_31,
    c02_12, c22_32,
    c03_13, c23_33;

  double* c01_11_ptr = c + lda;
  double* c02_12_ptr = c01_11_ptr + lda;
  double* c03_13_ptr = c02_12_ptr + lda;

  c00_10 = _mm_loadu_pd(c);
  c20_30 = _mm_loadu_pd(c+2);
  c01_11 = _mm_loadu_pd(c01_11_ptr);
  c21_31 = _mm_loadu_pd(c01_11_ptr+2);
  c02_12 = _mm_loadu_pd(c02_12_ptr);
  c22_32 = _mm_loadu_pd(c02_12_ptr+2);
  c03_13 = _mm_loadu_pd(c03_13_ptr);
  c23_33 = _mm_loadu_pd(c03_13_ptr+2);

  for (int x = 0; x < K; ++x) 
  {
    a0x_1x = _mm_load_pd(a);
    a2x_3x = _mm_load_pd(a+2);
    a += 4;

    bx0 = _mm_loaddup_pd(b++);
    bx1 = _mm_loaddup_pd(b++);
    bx2 = _mm_loaddup_pd(b++);
    bx3 = _mm_loaddup_pd(b++);

    c00_10 = _mm_add_pd(c00_10, _mm_mul_pd(a0x_1x, bx0));
    c20_30 = _mm_add_pd(c20_30, _mm_mul_pd(a2x_3x, bx0));
    c01_11 = _mm_add_pd(c01_11, _mm_mul_pd(a0x_1x, bx1));
    c21_31 = _mm_add_pd(c21_31, _mm_mul_pd(a2x_3x, bx1));
    c02_12 = _mm_add_pd(c02_12, _mm_mul_pd(a0x_1x, bx2));
    c22_32 = _mm_add_pd(c22_32, _mm_mul_pd(a2x_3x, bx2));
    c03_13 = _mm_add_pd(c03_13, _mm_mul_pd(a0x_1x, bx3));
    c23_33 = _mm_add_pd(c23_33, _mm_mul_pd(a2x_3x, bx3));
  }

  _mm_storeu_pd(c, c00_10);
  _mm_storeu_pd((c+2), c20_30);
  _mm_storeu_pd(c01_11_ptr, c01_11);
  _mm_storeu_pd((c01_11_ptr+2), c21_31);
  _mm_storeu_pd(c02_12_ptr, c02_12);
  _mm_storeu_pd((c02_12_ptr+2), c22_32);
  _mm_storeu_pd(c03_13_ptr, c03_13);
  _mm_storeu_pd((c03_13_ptr+2), c23_33);
}

static inline void copy_a (int lda, const int K, double* a_src, double* a_dest) {
  /* For each 4xK block-row of A */
  for (int i = 0; i < K; ++i) 
  {
    *a_dest++ = *a_src;
    *a_dest++ = *(a_src+1);
    *a_dest++ = *(a_src+2);
    *a_dest++ = *(a_src+3);
    a_src += lda;
  }
}

static inline void copy_b (int lda, const int K, double* b_src, double* b_dest) {
  double *b_ptr0, *b_ptr1, *b_ptr2, *b_ptr3;
  b_ptr0 = b_src;
  b_ptr1 = b_ptr0 + lda;
  b_ptr2 = b_ptr1 + lda;
  b_ptr3 = b_ptr2 + lda;

  for (int i = 0; i < K; ++i) 
  {
    *b_dest++ = *b_ptr0++;
    *b_dest++ = *b_ptr1++;
    *b_dest++ = *b_ptr2++;
    *b_dest++ = *b_ptr3++;
  }
}
/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static inline void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  double A_block[M*K], B_block[K*N];
  double *a_ptr, *b_ptr, *c;

  const int Nmax = N-3;
  int Mmax = M-3;
  int fringe1 = M%4;
  int fringe2 = N%4;

  int i = 0, j = 0, p = 0;

  /* For each column of B */
  for (j = 0 ; j < Nmax; j += 4) 
  {
    b_ptr = &B_block[j*K];
    // copy and transpose B_block
    copy_b(lda, K, B + j*lda, b_ptr);
    /* For each row of A */
    for (i = 0; i < Mmax; i += 4) {
      a_ptr = &A_block[i*K];
      if (j == 0) copy_a(lda, K, A + i, a_ptr);
      c = C + i + j*lda;
      calc_4x4(lda, K, a_ptr, b_ptr, c);
    }
  }

  /* Handle "fringes" */
  if (fringe1 != 0) 
  {
    /* For each row of A */
    for ( ; i < M; ++i)
      /* For each column of B */ 
      for (p = 0; p < N; ++p) 
      {
        /* Compute C[i,j] */
        double c_ip = ARRAY(C,i,p);
        for (int k = 0; k < K; ++k)
          c_ip += ARRAY(A,i,k) * ARRAY(B,k,p);
        ARRAY(C,i,p) = c_ip;
      }
  }
  if (fringe2 != 0) 
  {
    Mmax = M - fringe1;
    /* For each column of B */
    for ( ; j < N; ++j)
      /* For each row of A */ 
      for (i = 0; i < Mmax; ++i) 
      {
        /* Compute C[i,j] */
        double cij = ARRAY(C,i,j);
        for (int k = 0; k < K; ++k)
          cij += ARRAY(A,i,k) * ARRAY(B,k,j);
        ARRAY(C,i,j) = cij;
      }
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. 
 * Optimization: Two levels of blocking. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  for (int t = 0; t < lda; t += BLOCK_L2) 
  {
    int end_k = t + min(BLOCK_L2, lda-t);
    /* For each L2-sized block-column of B */
    for (int s = 0; s < lda; s += BLOCK_L2) 
    {
      int end_j = s + min(BLOCK_L2, lda-s);
      /* For each L2-sized block-row of A */ 
      for (int r = 0; r < lda; r += BLOCK_L2) 
      {
        int end_i = r + min(BLOCK_L2, lda-r);
        for (int k = t; k < end_k; k += BLOCK_L1) 
        {
          int K = min(BLOCK_L1, end_k-k);
          /* For each L1-sized block-column of B */
          for (int j = s; j < end_j; j += BLOCK_L1) 
          {
            int N = min(BLOCK_L1, end_j-j);
            /* For each L1-sized block-row of A */ 
            for (int i = r; i < end_i; i += BLOCK_L1) 
            {
              int M = min(BLOCK_L1, end_i-i);
              /* Performs a smaller dgemm operation
               *  C' := C' + A' * B'
               * where C' is M-by-N, A' is M-by-K, and B' is K-by-N. */
              do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
            }
          }
        }
      }
    }
  }
}

void square_dgemm_impl(int lda, double* A, double* B, double* C, int block_size_row, int block_size_col, int block_size_inner)
{
}
