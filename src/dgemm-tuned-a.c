const char* dgemm_desc = "Naive, three-loop dgemm.";

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */    
void square_dgemm (int n, double* A, double* B, double* C)
{
  double A1[n];
  /*  double B1[n];
   * For each row i of A */
  for (int i = 0; i < n; ++i)
{
    for (int m = 0; m < n; ++m)
    {
     A1[m] = A[i+m*n];
    }   
    /* For each column j of B */
    for (int j = 0; j < n; ++j) 
    {
     /*  for (int m = 0; m < n; ++m)
      *    {
      *    B1[m] = B[m+j*n];
      *    }
      * Compute C(i,j) */
      double cij = C[i+j*n];
      for( int k = 0; k < n; k++ )
      {
	cij += A1[k] * B[k+j*n];
        C[i+j*n] = cij;
      }
    }
}
}
