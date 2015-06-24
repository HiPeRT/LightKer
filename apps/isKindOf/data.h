#ifndef __DATA_H__
#define __DATA_H__

#include "header.h"

struct data_t {
	syncon_t s;
	int synconid;
	int n_dads;
	int dads[100];
};

#endif
