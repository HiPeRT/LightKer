#ifndef __DATA_H__
#define __DATA_H__

#define   WIDTH 8
#define   TILE_WIDTH 2

struct data_t
{
  float *array1_d;
  float *array2_d;
  int startRow;
};

struct res_t
{
  float *result_array_d;
};

#endif /* __DATA_H__*/
