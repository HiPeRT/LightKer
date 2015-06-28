#ifndef __DATA_H__
#define __DATA_H__

#include "header.h"

typedef struct rel rel_t;
struct rel
{
        int synconid;
        int tab;
};

typedef struct syncon syncon_t;
struct syncon
{
        rel_t* rel;
        int n_rel;
};

int totSize;

struct data_t {
	syncon_t *syncon;
	int *synconid;
	int *n_dads;
	int *dads;
	int *result;
};

#endif
