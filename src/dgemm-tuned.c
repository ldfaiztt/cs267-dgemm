#include <emmintrin.h>
#include <string.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#define RSIZE_M 2
#define RSIZE_K 2
#define RSIZE_N 2
#define I_STRIDE 2
  
#define BLOCK_ROW 222 
#define BLOCK_COL 12 
#define BLOCK_INNER 222 

#define turn_even(x) (((x) & 1) ? (x+1) : (x))
#define min(a,b) (((a)<(b))?(a):(b))

#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)

static inline void do_block(int M, int K, int N, double* restrict A, double* restrict B, double* restrict C) 
{
  __m128d c0, c1, a0, a1, b0, b1, b2, b3, d0, d1;    

  for (int k=0; k<K; k+=RSIZE_K) 
  {
    for (int j=0; j<N; j+=RSIZE_N) 
    {
      b0 = _mm_load1_pd(B+k+j*K);
      b1 = _mm_load1_pd(B+k+1+j*K);
      b2 = _mm_load1_pd(B+k+(j+1)*K);
      b3 = _mm_load1_pd(B+k+1+(j+1)*K);
      for (int i=0; i<M; i+=RSIZE_M) 
      {
        a0 = _mm_load_pd(A+i+k*M);
        a1 = _mm_load_pd(A+i+(k+1)*M);

        c0 = _mm_load_pd(C+i+j*M);
        c1 = _mm_load_pd(C+i+(j+1)*M);

        d0 = _mm_add_pd(c0, _mm_mul_pd(a0,b0));
        d1 = _mm_add_pd(c1, _mm_mul_pd(a0,b2));
        c0 = _mm_add_pd(d0, _mm_mul_pd(a1,b1));
        c1 = _mm_add_pd(d1, _mm_mul_pd(a1,b3));

        _mm_store_pd(C+i+j*M,c0);
        _mm_store_pd(C+i+(j+1)*M,c1); 
      }
    }
  }
}

static inline void copy_block(int lda, int M, int N, double* restrict A, double* restrict new_A) 
{
  int M_even = turn_even(M);
  int N_even = turn_even(N);

  for (int j=0; j<N; j++) 
  {
    int new_A_idx = j * M_even;
    int A_idx = j * lda;
    memcpy(&new_A[new_A_idx], &A[A_idx], M*sizeof(double));
  }
  if (N % 2) 
  {
    int idx = (N_even-1) * M_even;
    new_A += idx;
    memset(new_A, 0, M_even * sizeof(double));
  } 
}

static inline void add_block(double* new_A, double*  A, int M, int N, int lda, int M_even) 
{
  __m128d a; 
  int i_step;
  for (int j=0; j<N; j++) 
  {
    for (int i=0; i<M; i+=I_STRIDE) 
    {
      i_step = min(I_STRIDE,M-i); 
      if (unlikely(i_step == 1))
      {
        A[i+j*lda] = new_A[i+j*M_even];
      } 
      else 
      {
        a = _mm_load_pd(new_A + i + j*M_even);
        _mm_storeu_pd(A+i+j*lda,a);
      }
    }
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm(int lda, double* A, double* B, double* C)
{
  int M_even, K_even, N_even;

  double new_A[BLOCK_ROW * BLOCK_INNER] __attribute__((aligned(16)));
  double new_B[BLOCK_INNER * turn_even(lda)] __attribute__((aligned(16)));
  double new_C[BLOCK_ROW * BLOCK_COL] __attribute__((aligned(16)));

  for (int k = 0; k < lda; k += BLOCK_INNER) 
  {
    int K = min(BLOCK_INNER, lda-k);
    copy_block(lda, K, lda, B+k, new_B);
    K_even = turn_even(K);

    for (int i = 0; i < lda; i += BLOCK_ROW) 
    {
      int M = min (BLOCK_ROW, lda-i);
      copy_block(lda, M, K, A+i+k*lda, new_A);
      M_even = turn_even(M);

      for (int j = 0; j < lda; j += BLOCK_COL) 
      {
        int N = min (BLOCK_COL, lda-j);
        N_even = turn_even(N);               
        copy_block(lda, M, N, C+i+j*lda, new_C);

        do_block(M_even, K_even, N_even, new_A, new_B+j*K_even, new_C);
        add_block(new_C, C+i+j*lda, M, N, lda, M_even);
      }
    }
  }
}

void square_dgemm_impl(int lda, double* A, double* B, double* C, int block_size_row, int block_size_col, int block_size_inner)
{
}
