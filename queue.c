#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> 


typedef struct 
{
    int* data;
    block* next;
    int index;
} block;


typedef struct 
{
    block* start;
    int full_size;
    block* end;
} queue;


static queue init(int block_size) {
    int*
}