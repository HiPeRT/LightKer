#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <cuda.h>
#include <ctype.h>
#include "header.h"

/*	Inizializza la struttura dati	*/
void initialize (syncon_t *s);

/*	Inizializza il contatore	*/
void init (int *curr_i);

/*	Dalla riga estrae padre e figlio e 
	li inserisce nella struttura dati
*/
void workOnRow (char * row, syncon_t *s, int tab, int *curr_i);

/*	Stampa la struttura dati	*/
void print(syncon_t *s, int n);

/*	Conta il numero di padri di ogni syncon	*/
void contDadsAndSons (syncon_t *s);

/*	Legge una tabella dei link	*/
void readTable ( syncon_t *s, int *curr_i);

/*	Legge dal file passato (il file di test) NBLOCKS istanze di test 
	e inserisce i dati nelle apposite strutture dati
	Ritorna 1 quando il file è terminato, 0 altrimenti
*/
int readNewTest(FILE *infile,int *n_dads,int *syncon,int *dads);

/*	Stampa i risultati di un'esecuzione del kernel	*/
void printResults (int *result, int j);

/*
	Kernel cuda: implementazione della primitiva isKindOf su GPU
	Vengono usati NBLOCKS blocchi, ognuno con NTHREADS thread.
	Ogni blocco riceve un'istanza differente di test che risolve il parallelo.
	
	Dato un array di padri e un synconid la funzione deve trovare un 
	legame tra il syncon e uno dei padri nella gerarchia di iperinomia 
	(memorizzata nella struttura dati s). 
	Se non esiste relazione ritorna -1, se esiste ritorna la profondità
	
*/
__device__ void isKindOf(syncon_t *s, int *synconid, int *n_dads,int *dads,int *result);


/************************************* MAIN ***********************************/
int curr_i[NSYNCON]; 
syncon_t *s;
int *n_dads, *dads,*syncon,*result;
FILE *infile;
struct timespec spec_start, spec_stop;

syncon_t *d_s, *temp_s;
int *d_dads, *d_n_dads, *d_syncon, *d_result;

static int APP_num_blocks;

void init_data(data_t **data, int numblocks)
{
	totSize =0; 
	totSize += sizeof(syncon_t)*NSYNCON;
	data_t *data_p;

	log("init_data 1\n");

	//PARTE HOST

	checkCudaErrors(cudaHostAlloc((void **)&s, NSYNCON*sizeof(syncon_t), cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc((void**)&dads, numblocks*MAXDADS*sizeof(int),cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc((void **)&result, numblocks*sizeof(int), cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc((void**)&syncon,numblocks*sizeof(int),cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc((void**)&n_dads,numblocks*sizeof(int),cudaHostAllocDefault));

	log("init_data 2\n");
	initialize(s);
	log("init_data 3\n");
	init(curr_i);
	log("init_data 4\n");
	contDadsAndSons (s);
	log("init_data 5\n");
	readTable(s,curr_i);
	log("init_data 6\n");

	//PARTE DEVICE

	checkCudaErrors(cudaHostAlloc((void **)&temp_s, NSYNCON*sizeof(syncon_t), cudaHostAllocDefault));
	for ( int i = 0; i < NSYNCON; i++ )
	{
		rel_t * temp;
		checkCudaErrors(cudaMalloc( (void**) &temp, s[i].n_rel*sizeof(rel_t) ));
		checkCudaErrors(cudaMemcpy(temp, s[i].rel, s[i].n_rel*sizeof(rel_t) , cudaMemcpyHostToDevice));
	
		temp_s[i].n_rel = s[i].n_rel;
		temp_s[i].rel = temp;
	}
	log("init_data 7\n");

	checkCudaErrors(cudaMalloc ((void **)&d_s, NSYNCON*sizeof(syncon_t)));
	checkCudaErrors(cudaMemcpy(d_s, temp_s, sizeof(syncon_t)*NSYNCON, cudaMemcpyHostToDevice));
	log("init_data 8\n");

	checkCudaErrors(cudaHostAlloc((void **)data, sizeof(data_t), cudaHostAllocDefault));
	log("init_data 8x\n");
	data_p = *data;
	checkCudaErrors(cudaHostAlloc((void**)&data_p->dads, numblocks*MAXDADS*sizeof(int), cudaHostAllocDefault));
	log("init_data 8a\n");
	checkCudaErrors(cudaHostAlloc((void**)&data_p->syncon, numblocks*sizeof(int), cudaHostAllocDefault));
	log("init_data 8b\n");
	checkCudaErrors(cudaHostAlloc((void**)&data_p->dads, numblocks*sizeof(int), cudaHostAllocDefault));
	log("init_data 8c\n");
	checkCudaErrors(cudaHostAlloc((void**)&data_p->result, numblocks*sizeof(int), cudaHostAllocDefault));
	log("init_data 9\n");

	APP_num_blocks = numblocks;

	//printf("La dimensione totale è %f MB\n\n",(float)totSize/1024/1024);
	
	infile=fopen(test, "r");

	if( infile==NULL ) 
	{
		perror("Errore in apertura del file");
		exit(2);
	}
}

void assign_data(data_t *data, void *payload, int sm)
{		
	if(readNewTest(infile, n_dads, syncon, dads))
		return;		

	//clock_gettime(CLOCK_MONOTONIC, &spec_start);	

	log("ASSIGN\n");
#if 0
	cudaMemcpy(data->dads, dads, APP_num_blocks*MAXDADS*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(data->syncon, syncon, APP_num_blocks*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(data->n_dads, n_dads, APP_num_blocks*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(data->result, result, APP_num_blocks*sizeof(int), cudaMemcpyHostToDevice);
#endif
}

__device__ int work_cuda(volatile data_t data)
{
	int result;

	log("WORK\n");

	isKindOf((syncon_t *)data.syncon, (int *)data.synconid, (int *)data.n_dads, (int *)data.dads, (int *)data.result);

	result = *(data.result);
	return result;
}

__device__ int work_nocuda(volatile data_t data)
{
	return 0;
}

/***********************************FUNZIONI**********************************/

void initialize (syncon_t *s)
{
	for(int i=0;i<NSYNCON; i++)
	{
		s[i].rel = NULL;
		s[i].n_rel = 0;
	}
}

void init (int *curr_i)
{
	for(int i=0; i<NSYNCON; i++)
		curr_i[i]=0;
}

void workOnRow (char * row, syncon_t *s, int tab, int *curr_i)
{
	int dad;
	int son;
	sscanf (row,"#%d\t#%d",&son,&dad);	

	if (s[son].rel == NULL)
	{
		checkCudaErrors(cudaHostAlloc((void **)&s[son].rel, s[son].n_rel*sizeof(rel_t), cudaHostAllocPortable));
		totSize+=sizeof(rel_t)*s[son].n_rel;
	}
	s[son].rel[curr_i[son]].tab =tab;
	s[son].rel[curr_i[son]].synconid= dad;
	curr_i[son]++;
}

void print(syncon_t *s, int n)
{
	for(int i=0; i<n; i++)
	{
		printf("Synconid %d\n",i);
		printf("\tPadri: \n");
		for(int j=0;j<s[i].n_rel;j++)
			printf("\tSynconid : %d, tabella %d\n",s[i].rel[j].synconid,s[i].rel[j].tab);
		printf("\tNumero di padri: %d \n",s[i].n_rel);
	}
}

void contDadsAndSons (syncon_t *s)
{
	FILE *infile;
	char row[MAXLEN];
	char *check;
	int dad,son;

	infile=fopen(link0, "r");
	if( infile==NULL ) 
	{
		perror("Errore in apertura del file");
		exit(1);
	}	

	while(1) 
	{
		check=fgets(row, MAXLEN, infile);

		if( check == NULL )
			break;
		if(row[0]=='/' || row[0]=='\n')
			continue;

		sscanf (row,"#%d\t#%d",&son,&dad);	

		s[son].n_rel++;
	}
	fclose(infile);
}

void readTable ( syncon_t *s, int *curr_i)
{
	FILE *infile;
	char row[MAXLEN];
	char *check;
	
	infile=fopen(link0, "r");
	if( infile==NULL ) 
	{
		perror("Errore in apertura del file");
		exit(1);
	}	

	while(1) 
	{
		check=fgets(row, MAXLEN, infile);

		if( check == NULL )
			break;
		if(row[0]=='/' || row[0]=='\n')
			continue;
		workOnRow(row, s, 0,curr_i);

	}

	fclose(infile);
}

int readNewTest(FILE *infile,int *n_dads,int *syncon,int *dads)
{
	char row [MAXLEN];
	char *p,*check;
	bool testErr = false;

	for(int i=0; i<APP_num_blocks; i++)
	{
		check=fgets(row, MAXLEN, infile);
		if (check == NULL)
			return 1;
		p = row;
		n_dads[i] = -1;
		while (*p) 
		{
			if (isdigit(*p)) 
			{ 
				if (n_dads[i] == -1)
				{
					syncon[i] = strtol(p, &p, 10);
					if(syncon[i]> NSYNCON-1)
					{
						//printf("Errore nel testcase syncon!\n");
						testErr = true;
						break;
					}
						
				}
				else
				{
					dads[i*MAXDADS+n_dads[i]] = strtol(p, &p, 10);
				}

				n_dads[i]++;
			} 
			else 
			    p++;
		}
		if(testErr)
		{
			i--;
			testErr = false;
			continue;
		}
			
	}
	return 0;
}

void printResults (int *result, int j)
{
	for(int i=0; i< APP_num_blocks; i++)	
	{
		if(result[i] == -1)
			printf("Non ci sono relazioni tra i syncon\n");
		else
			printf("La profondità è : %d e sono all'iterazione %d\n",result[i],i+j*APP_num_blocks);
	}
}

__device__ void isKindOf(syncon_t *s, int *synconid, int *n_dads, int *dads, int *result)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int threadRunning = blockDim.x;
	
	__shared__ volatile int done,curr_syn_glob,level;
	__shared__ int s_dim[MAXLEVEL];
	__shared__ rel_t *s_ptr[MAXLEVEL];
	
	int curr_syn;

	if(tid==0)
	{
		result[bid] = -1;
		level = 0;
		done = 0;
		s_dim[0] = 0;
		s_ptr[0] = NULL;
	}

	curr_syn =synconid[bid];
	__syncthreads();

	dbgsrc("Sono prima del while\n");
	
	while (1)
	{
		if(tid==0)
			dbgsrc("\n\n\n\nNUOVO GIRO:\tControllo il syncon %d\n", curr_syn);

		for(int i=0; i<(n_dads[bid]/threadRunning+1); i++,tid+=threadRunning)
			if(tid < n_dads[bid])
			{
				dbgsrc("Controllo il padre %d\n",dads[bid*MAXDADS+tid]);
				if(curr_syn == dads[bid*MAXDADS+tid])
				{
					if(result[bid] == -1)
						result[bid]= level+1;
					else if (result[bid]> level +1)
						result[bid] = level+1;
				}
			}

		tid = threadIdx.x;
		__syncthreads();

		if(tid == 0)
		{

			dbgsrc(	"Il syncon non ha dato match\n"
				"Numero di padri di %d : %d\n",curr_syn,s[curr_syn].n_rel);

			if(s[curr_syn].n_rel != 0)
			{	
				dbgsrc("Il livello è %d\n",level);
	
				s_ptr[level] = s[curr_syn].rel;
				s_dim[level] = s[curr_syn].n_rel;
					
				curr_syn = s_ptr[level]->synconid;	//mi sposto sul figlio		
				level++;
	
				dbgsrc(	"curr_ptr punta a %d\t curr_dim è %d\til livello è %d\n",
						s_ptr[level-1]->synconid, s_dim[level-1], level);
				dbgsrc("Il syncon corrente è %d\n", curr_syn);
			}
			else 
			{
				s_dim[level] = 0;
				while (s_dim[level] < 2 && level >=0)
				{
					dbgsrc(	"Sono entrato nel while:\t s_dim vale %d"
							"\til livello è %d\n",s_dim[level],level);
	
					s_ptr[level] = NULL;
					s_dim[level] = 0;
					level--;
					if(level >= 0)
						dbgsrc(	"Sono alla fine del while:\t curr_ptr punta a %d\t "
							"curr_dim è %d\t il livello è %d\n",
							s_ptr[level]->synconid, s_dim[level], level);
				}

				if(s_ptr[0] == NULL )
				{
					done = 1;
				}
				else 
				{	
					s_ptr[level]++;
					s_dim[level]--;
					curr_syn = s_ptr[level]->synconid; //mi sposto sul fratello
					level++;
	
					dbgsrc(	"Il livello è %d\t il fratello è %d\t e la dimensione è %d\n",
						level, curr_syn,s_dim[level-1]);
				}
			}
		
			curr_syn_glob = curr_syn;
		}
		__syncthreads();
		if(done)
			break;
		curr_syn = curr_syn_glob;
		__syncthreads();
		
	}
	
	dbgsrc("Il risultato è %d\n",*result);
}


