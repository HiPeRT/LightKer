__device__ void WORK(float *Md , float *Nd , float *Pd , const int width, unsigned int col, unsigned int row)
{
  for (int k = 0 ; k<WIDTH ; k++ )
  {
    Pd[row*WIDTH + col] += Md[row * WIDTH + k ] * Nd[ k * WIDTH + col] ;
//     printf("k %d Md[%d] %f Nd[%d] %f Pd[%d] %f\n", k, row * WIDTH + k, Md[row * WIDTH + k ], k * WIDTH + col, Nd[ k * WIDTH + col], row*WIDTH + col, Pd[row*WIDTH + col]);
  }
}