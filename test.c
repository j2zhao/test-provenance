#include <stdio.h>
#include <string.h>

// static int b_add(int a, int b) {
//     return a + b;
// }

#define AS_RATIONAL(dst,object) \
    { \
        dst = object; \
    }


#define RATIONAL_BINOP_2(name,exp) \
    static int \
    a_##name(int a, int b) { \
        int x, y, z; \
        x = 7; \
        y = 10; \
        z = exp; \
        return z; \
    }
#define RATIONAL_BINOP(name, op) RATIONAL_BINOP_2(name, a op b)
    
RATIONAL_BINOP(add, +)

int main(void){
    int a[0];
    // int e[3] = {4, 4, 4};
    // memcpy(a, e, 3*sizeof(int));
    
    // for(int i = 0; i < 100; i++) {
    //     printf("%d ", a[i]);
    // }
    // printf("\n");
    

    int i = 1;
    int j = 2;
    
    int b = a_add(i, j);
    printf("%d", b);
    return 0;
}
