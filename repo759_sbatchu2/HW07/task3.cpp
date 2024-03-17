#include<iostream>
#include<omp.h>
using namespace std;
int main()
{
     omp_set_num_threads(4);
     cout<<"Number of threads:  "<<4<<endl;
     int fac[8];
     #pragma omp parallel for
     for(int i=1; i<=8; i++)
     { 
        fac[i]=1;
        for(int j=1; j<=i; j++){
            fac[i] = fac[i]*j;
        }
        int myId = omp_get_thread_num();
        printf("I am thread No.  %d\n",myId);
        printf("%d!=%d\n",i, fac[i]);
        //cout<<"I am thread No.  "<<myId<<endl;
        //cout<<i<<"!="<<fac[i]<<endl;
     } 


return 0;
}