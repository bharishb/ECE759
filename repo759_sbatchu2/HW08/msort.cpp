
#include "msort.h"
#include <algorithm>
#include <iostream>

void merge_sorted_arrays(int* arr, std::size_t left, std::size_t mid, std::size_t right){
   
     std::size_t arr1_size = mid - left + 1;            // left --> mid
     std::size_t arr2_size = right - (mid+1) + 1;       // mid+1 --> right

     int* arr1 = new int[arr1_size];
     int* arr2 = new int[arr2_size];

     for(std::size_t i =0; i<arr1_size; i++)
     {
        arr1[i] = arr[i+left];
     } 
     for(std::size_t i =0; i<arr2_size; i++)
     {
        arr2[i] = arr[i+mid+1];
     } 

    std::size_t i=0, j=0, k=left;
     //merge two sorted arrays
     while((i<arr1_size) && (j<arr2_size)){
          if(arr1[i] <= arr2[j]){
            arr[k] = arr1[i];
            i++;
            k++;
          } else
          {
            arr[k] = arr2[j];
            j++;
            k++;  
          }
     }

     //copy left over chuncks of one array if we run out of other array
     while(i<arr1_size){
        arr[k] = arr1[i];
        i++;
        k++;
     }

     while(j<arr2_size){
        arr[k] = arr2[j];
        j++;
        k++;
     }

     //delete temp arrays created
     delete [] arr1;
     delete [] arr2;
     arr1 = nullptr;
     arr2 = nullptr;

}



void merge_recursive(int* arr, std::size_t left, std::size_t right, std::size_t threshold){  //taking arguments as left, right indices

//print thread id
//int myId = omp_get_thread_num();
//std::printf("I am thread No.  %d (left, right, threshold) = (%d, %d, %d)\n",myId, left, right, threshold);


if((right - left + 1) < threshold) {

   //do sequential sorting
    std::sort(arr+left, arr+right+1); 
    //std::printf("Sequential Sort : I am thread No.  %d (left, right, threshold) = (%d, %d, %d)\n",myId, left, right, threshold);

}
else {
    size_t mid = left + (right-left)/2;
    #pragma omp parallel
    {
        #pragma omp single nowait
        {

            #pragma omp task
            {
                //std::printf("Parallel Left Sort : I am thread No.  %d (left, right, mid) = (%d, %d, %d)\n",myId, left, right, mid);
                merge_recursive(arr, left, mid, threshold);
            }
            #pragma omp task
            {
                //std::printf("Parallel Right Sort : I am thread No.  %d (left, right, mid) = (%d, %d, %d)\n",myId, left, right, mid);
                merge_recursive(arr, mid+1, right, threshold);
            }
            #pragma omp taskwait
            {
                merge_sorted_arrays(arr, left, mid, right);
            }

        }

    }

}


}

void msort(int* arr, const std::size_t n, const std::size_t threshold){
    
    merge_recursive(arr, 0, n-1,threshold);

}