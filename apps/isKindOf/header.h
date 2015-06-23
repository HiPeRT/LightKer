#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <cuda.h>
#include <ctype.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define MAXLEN 1000 	//massima lunghezza di una stringa
#define MAXLEVEL 100 	//massimo livello di iperinomia
#define MAXDADS 100	//massimo numero di padri nel file di test
#define NSYNCON 201200 	//numero di syncon
#define NTHREADS 192	//numero di thread per blocco
#define NBLOCKS 64	//numero di blocchi 

#define test "testIsKindOfPadri.txt" 		//test
#define link0 "../link/french.link0.dump.txt"	//link di iperinomia

#define clock_getdiff_nsec(start, stop) ((stop.tv_sec - start.tv_sec)*1000000000 + (stop.tv_nsec - start.tv_nsec))

//#define DEBUGSEARCH
#ifdef DEBUGSEARCH
    #define dbgsrc(...)  printf(__VA_ARGS__)
#else
    #define dbgsrc(...)
#endif

#define checkCudaErrors(val) checkErr( (val), #val, __FILE__, __LINE__)

template<typename T>
void checkErr(T err, const char* const func, const char* const file, const int line) 
{
	if (err != cudaSuccess) 
	{
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << ": " << func << " " << err <<std::endl;
		exit(1);
	}
}

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


