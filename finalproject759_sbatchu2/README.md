Implementation of RNN in Cuda

This project implements various Kernels involved in LSTM inference in cpp(cpu), Cuda(GPU), OpenMP(cpu) and does a performance study with respect to various optimizations of the kernels

One LSTM cell operations can be described as below.

1. i(t) = sigmoid(W_ii * x(t)   +  W_hi * h(t-1)  +  b_hi)
2. f(t) = sigmoid(W_if * x(t)   +  W_hf * h(t-1)  +  b_hf)
3. g(t) = tanh(W_ig * x(t)   +  b_ig  +  W_hg * h(t-1)  +  b_hg)
4. o(t) = sigmoid(W_io * x(t)  +  b_io  +  W_ho * h(t-1)  +  b_ho)
5. c(t) = f(t) op c(t-1)  +  i(t) op  g(t)
6. h(t) = o(t) op tanh(c(t)) 

In general, any LSTM based neural network applications contain Linear layer too.

One Linear Layer does the following operation
1. y = x * A_T  +  b 


