#include "count.cuh"
#include<thrust/reduce.h>
#include<thrust/pair.h>
#include<iostream>
#include<thrust/sort.h>


void count(const thrust::device_vector<int>& d_in,
                 thrust::device_vector<int>& values,
                 thrust::device_vector<int>& counts)
{

//sorted array location
thrust::device_vector<int> d_in_sorted(d_in.size());

//Inialize
thrust::device_vector<int> values_in(d_in.size(),1);

//copy to get it sorted
d_in_sorted = d_in;

//sort by thrust
thrust::sort(d_in_sorted.begin(), d_in_sorted.end());

//reduce by key
//thrust::pair<int*,int*> new_end;
auto new_end = thrust::reduce_by_key(d_in_sorted.begin(), d_in_sorted.end() , values_in.begin(), values.begin(), counts.begin());
int values_size = new_end.first - values.begin();

values.resize(values_size);
counts.resize(values_size);
}