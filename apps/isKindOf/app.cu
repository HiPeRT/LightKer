#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<time.h>
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

/*	Stampa i risultati di un'esecuzione del kernel	*/
void printResults (int *result, int j);

/*
	Dato un array di padri e un synconid la funzione deve trovare un 
	legame tra il syncon e uno dei padri nella gerarchia di iperinomia 
	(memorizzata nella struttura dati s). 
	Se non esiste relazione ritorna -1, se esiste ritorna la profondità
*/
__device__ int isKindOf(syncon_t *s, int synconid, int *dads, int n_dads) ;

/************************************* MAIN ***********************************/

FILE *infile = NULL;
syncon_t *s, *d_s;
int *curr_i, *d_curr_i;

data_t temp;

void init_data(data_t **data, int numblocks)
{
	totSize = 0;

	checkCudaErrors(cudaHostAlloc((void **)data, numblocks * sizeof(data_t), cudaHostAllocDefault));

	checkCudaErrors(cudaHostAlloc((void **)&s, NSYNCON*sizeof(syncon_t), cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc((void **)&curr_i, NSYNCON*sizeof(int), cudaHostAllocDefault));

	initialize(s);
	init(curr_i);
	contDadsAndSons(s);
	readTable(s,curr_i);

	checkCudaErrors(cudaMalloc((void**)&d_s, NSYNCON*sizeof(syncon_t)));
	checkCudaErrors(cudaMemcpy(d_s, s, NSYNCON*sizeof(syncon_t) , cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&d_curr_i, NSYNCON*sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_curr_i, curr_i, NSYNCON*sizeof(int) , cudaMemcpyHostToDevice));

	infile=fopen(test2, "r");
	if( infile==NULL ) 
	{
		perror("Errore in apertura del file");
		exit(1);
	}

	totSize += sizeof(syncon_t) * NSYNCON;

	printf("La dimensione totale è %d byte\n",totSize);
}

void assign_data(data_t *data, void *payload, int sm)
{
	char row[MAXLEN];
	char *check, *p;

	check = fgets(row, MAXLEN, infile);
	if( check == NULL )
		return;

	p = row;
	temp.n_dads = -1;

	while (*p) {
		if (isdigit(*p)) { 
			if (temp.n_dads == -1) {
				temp.synconid = strtol(p, &p, 10);
				if(temp.synconid > NSYNCON-1)
					break;
			} else
				temp.dads[temp.n_dads] = strtol(p, &p, 10);
			temp.n_dads++;
		} else 
			p++;
	}
	
	memcpy(&data[sm], &temp, sizeof(data_t));
	log("assigned data to thread %d\n", sm);
}

__device__ int work_nocuda(volatile data_t data)
{
        log("Hi! I'm block %d and I'm working on data [NOCUDA]\n", blockIdx.x);
	int deep = isKindOf((syncon_t *)&data.s, data.synconid, (int *)data.dads, data.n_dads);
        return deep;
}

__device__ int work_cuda(volatile data_t data)
{
        log("Hi! I'm block %d and I'm working on data [CUDA]\n", blockIdx.x);
	int deep = isKindOf((syncon_t *)&data.s, data.synconid, (int *)data.dads, data.n_dads);
        return deep;
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
		s[son].rel = (rel_t*)malloc(sizeof(rel_t)*s[son].n_rel);
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


void printResults (int result, int j)
{
	if(result == -1)
			printf("Non ci sono relazioni tra i syncon\n");
	else
			printf("La profondità è : %d e sono all'iterazione %d\n",result,j);
}


__device__ int isKindOf(syncon_t *s, int synconid, int *dads, int n_dads) 
{
	
	int i;
	int s_dim[MAXLEVEL];
	rel_t *s_ptr[MAXLEVEL];
	int level =0;
	int curr_syn = synconid;
	int result = -1;

	s_dim[0] =0;
	s_ptr[0] = NULL;

	while (1)
	{
		dbgsrc("\n\n\n\nNUOVO GIRO:\tControllo il syncon %d\n", curr_syn);

		for(i=0; i<n_dads; i++)
		{
			dbgsrc("Controllo il padre %d\n",dads[i]);
			if(curr_syn == dads[i])
			{
				if(result == -1)
					result = level+1;
				else if(result > level+1)
					result = level+1;
			}
		}

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
				return result;

			s_ptr[level]++;
			s_dim[level]--;
			curr_syn = s_ptr[level]->synconid; //mi sposto sul fratello
			level++;

			dbgsrc(	"Il livello è %d\t il fratello è %d\t e la dimensione è %d\n",
					level, curr_syn,curr_dim);
		}
	}

}

