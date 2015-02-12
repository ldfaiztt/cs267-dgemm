#include <emmintrin.h>
#include <string.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#define RSIZE_M 2
#define RSIZE_K 2
#define RSIZE_N 2
#define I_STRIDE 2

#define turn_even(M) (((M)+1)&(~0x1))
#define min(x,y)  ((y)^(((x)^(y)) & -((x)<(y))))

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

static inline double* copy_block(int lda, int M, int N, double* restrict A, double* restrict new_A) 
{
  int M_even = turn_even(M);
  int N_even = turn_even(N);
  int i_step;
  __m128d a;

  for (int j=0; j<N; j++) 
  {
    for (int i=0; i<M; i+=I_STRIDE) 
    {
      i_step = min(I_STRIDE, M-i);
      if (i_step == 1) 
      {            
        new_A[i+j*M_even] = A[i+j*lda];
      } 
      else 
      {
        a = _mm_loadu_pd(A+i+j*lda);
        _mm_store_pd(new_A+i+j*M_even, a);
      }
    }
  }
  if (N % 2) 
  {
    for (int i=0; i<M_even; i++) 
    {
      new_A[i+(N_even-1)*M_even] = 0.0;
    }
  } 
  return new_A;
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
      if (i_step == 1) 
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
void square_dgemm_impl(int lda, double* A, double* B, double* C, int block_size_row, int block_size_col, int block_size_inner) 
{
  int M_even, K_even, N_even;

  double new_A[block_size_row * block_size_inner] __attribute__((aligned(16)));
  double new_B[block_size_inner * turn_even(lda)] __attribute__((aligned(16)));
  double new_C[block_size_row * block_size_col] __attribute__((aligned(16)));

  for (int k = 0; k < lda; k += block_size_inner) 
  {
    int K = min(block_size_inner, lda-k);
    copy_block(lda, K, lda, B+k, new_B);
    K_even = turn_even(K);

    for (int i = 0; i < lda; i += block_size_row) 
    {
      int M = min (block_size_row, lda-i);
      copy_block(lda, M, K, A+i+k*lda, new_A);
      M_even = turn_even(M);

      for (int j = 0; j < lda; j += block_size_col) 
      {
        int N = min (block_size_col, lda-j);
        N_even = turn_even(N);               
        copy_block(lda, M, N, C+i+j*lda, new_C);

        do_block(M_even, K_even, N_even, new_A, new_B+j*K_even, new_C);
        add_block(new_C, C+i+j*lda, M, N, lda, M_even);
      }
    }
  }
}

void square_dgemm(int lda, double* A, double* B, double* C)
{
  int block_size_row = 222; 
  int block_size_col = 12; 
  int block_size_inner = 222; 
  square_dgemm_impl(lda, A, B, C, block_size_row, block_size_col, block_size_inner);
}
