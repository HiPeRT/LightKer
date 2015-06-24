#ifndef __HEADER_H__
#define __HEADER_H__

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <ctype.h>

#define MAXLEN 1000 	//massima lunghezza di una stringa
#define MAXLEVEL 100 	//massimo livello di iperinomia
#define MAXDADS 100	//massimo numero di padri nel file di test
#define NSYNCON 201200 	//numero di syncon
#define NTHREADS 100	//numero di thread per blocco
#define NBLOCKS 1	//numero di blocchi 

#define test1 "testIsKindOf1.txt" 			//test padre singolo
#define test2 "testIsKindOfPadri.txt" 		//test padri multipli
#define link0 "../link/french.link0.dump.txt"	//link di iperinomia

#define clock_getdiff_nsec(start, stop) ((stop.tv_sec - start.tv_sec)*1000000000 + (stop.tv_nsec - start.tv_nsec))

//#define DEBUGSEARCH
#ifdef DEBUGSEARCH
    #define dbgsrc(...)  printf(__VA_ARGS__)
#else
    #define dbgsrc(...)
#endif

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

#endif /* __HEADER_H__ */
