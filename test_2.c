#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> 

pthread_t writer;
int initiated = 0;
int end = 0;
int b_size;
int b_temp;
int* block;
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t c = PTHREAD_COND_INITIALIZER;
pthread_cond_t d = PTHREAD_COND_INITIALIZER;

void *write_disk(void *vargp) {
    pthread_mutex_lock(&m);
    while (1) {
        pthread_cond_wait(&c, &m);
        printf("%d ",end);
        if (end == 0) {
            int i;
            for(i = 0; i < b_size; i++) {
                printf("%d ", block[i]);
            }
            printf("\n");
            b_temp = 0;
        }   
        else {
            int i;
            for(i = 0; i < b_temp; i++) {
                printf("%d ", block[i]);
            }
            b_temp = 0;
            initiated = 0;
            end = 0;
            free(block);
            break;
        }
    }
    pthread_mutex_unlock(&m);
    return NULL;
}


static int
thread_initializer(int block_size){
    pthread_mutex_lock(&m);
    if (initiated == 0) {
        pthread_create(&writer, NULL, write_disk, NULL);
        initiated = 1;
        b_size = block_size;
        b_temp = 0;
        block = malloc(sizeof (int) * 100);
        pthread_mutex_unlock(&m);
        return 1;
    }
    else {
        pthread_mutex_unlock(&m);
        return 0;
    }
}

static int
thread_end(void){
    int i;
    pthread_mutex_lock(&m);
    end = 1;
    pthread_cond_signal(&c);
    pthread_mutex_unlock(&m);
    i = pthread_join(writer, NULL);
    return i;
}

int main(void) {
    int i;
    thread_initializer(12);
    for(i = 0; i < 100; i ++) {
        pthread_mutex_lock(&m);
        block[b_temp] = i;
        b_temp += 1;
        printf("%d ", b_temp);
        printf("%d ", b_size);
        printf("\n");
        if (b_temp%12 == 0 && b_temp != 0) {
            pthread_cond_signal(&c);
        }
        pthread_mutex_unlock(&m);
        sleep(1);
    }
    thread_end();
}