const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static inline void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  for (int k = 0; k < K; ++k)
    for (int j = 0; j < N; ++j) 
    {
      double b_kj = B[k+j*lda];
      int i = 0;
      /*
      for (; i+3 < M; i+=4) {
        double a_ik = A[i+k*lda];
        double a_i1k = A[i+1+k*lda];
        double a_i2k = A[i+2+k*lda];
        double a_i3k = A[i+3+k*lda];
        C[i+j*lda] += a_ik * b_kj;
        C[i+1+j*lda] += a_i1k * b_kj;
        C[i+2+j*lda] += a_i2k * b_kj;
        C[i+3+j*lda] += a_i3k * b_kj;
      }
      */
      for (; i < M; ++i)
        C[i+j*lda] += A[i+k*lda] * b_kj;
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  /* For each block-row of A */ 
  for (int i = 0; i < lda; i += BLOCK_SIZE)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int M = min (BLOCK_SIZE, lda-i);
        int N = min (BLOCK_SIZE, lda-j);
        int K = min (BLOCK_SIZE, lda-k);

        /* Perform individual block dgemm */
        do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
}
