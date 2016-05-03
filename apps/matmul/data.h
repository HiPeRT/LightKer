#ifndef __DATA_H__
#define __DATA_H__

#define  WIDTH 8
#define TILE_WIDTH 2

//   float array1_d[WIDTH][WIDTH/TILE_WIDTH];
//   float array2_d[WIDTH][WIDTH/TILE_WIDTH];
struct data_t
{
  float *array1_d;
  float *array2_d;
  float *result_array_d;
};

struct res_t
{
  /* Nothing */
};

#endif /* __DATA_H__*/
