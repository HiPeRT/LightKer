#ifndef __DATA_H__
#define __DATA_H__

#ifndef L_MAX_LENGTH
#define L_MAX_LENGTH 20
#endif /* L_MAX_LENGTH */


#ifndef WORK_TIME
#define WORK_TIME 200000
#endif

struct data_t
{
  char str[L_MAX_LENGTH];
};

struct res_t
{
  unsigned int num;
};

#endif /* __DATA_H__*/
