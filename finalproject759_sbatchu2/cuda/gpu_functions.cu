#include<cuda.h>
#include "LSTM.cuh"
void LSTM::move_params_to_gpu(){
    printf("Moving params to GPU\n");
    cudaMalloc((void**)&W_ii_gpu, sizeof(float)*m_input_size*m_hidden_size);
    cudaMemcpy(W_ii_gpu, W_ii, m_input_size*m_hidden_size*(sizeof(float)), cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&W_if_gpu, sizeof(float)*m_input_size*m_hidden_size);
    cudaMemcpy(W_if_gpu, W_if, m_input_size*m_hidden_size*(sizeof(float)), cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&W_ig_gpu, sizeof(float)*m_input_size*m_hidden_size);
    cudaMemcpy(W_ig_gpu, W_ig, m_input_size*m_hidden_size*(sizeof(float)), cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&W_io_gpu, sizeof(float)*m_input_size*m_hidden_size);
    cudaMemcpy(W_io_gpu, W_io, m_input_size*m_hidden_size*(sizeof(float)), cudaMemcpyHostToDevice); 

    
    cudaMalloc((void**)&b_ii_gpu, sizeof(float)*m_hidden_size);
    cudaMemcpy(b_ii_gpu, b_ii, m_hidden_size*(sizeof(float)), cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&b_if_gpu, sizeof(float)*m_hidden_size);
    cudaMemcpy(b_if_gpu, b_if, m_hidden_size*(sizeof(float)), cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&b_ig_gpu, sizeof(float)*m_hidden_size);
    cudaMemcpy(b_ig_gpu, b_ig, m_hidden_size*(sizeof(float)), cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&b_io_gpu, sizeof(float)*m_hidden_size);
    cudaMemcpy(b_io_gpu, b_io, m_hidden_size*(sizeof(float)), cudaMemcpyHostToDevice); 

    cudaMalloc((void**)&b_iiB_gpu, sizeof(float)*m_batch_size*m_hidden_size);
    cudaMemcpy(b_iiB_gpu, b_iiB, m_batch_size*m_hidden_size*(sizeof(float)), cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&b_ifB_gpu, sizeof(float)*m_batch_size*m_hidden_size);
    cudaMemcpy(b_ifB_gpu, b_ifB, m_batch_size*m_hidden_size*(sizeof(float)), cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&b_igB_gpu, sizeof(float)*m_batch_size*m_hidden_size);
    cudaMemcpy(b_igB_gpu, b_igB, m_batch_size*m_hidden_size*(sizeof(float)), cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&b_ioB_gpu, sizeof(float)*m_batch_size*m_hidden_size);
    cudaMemcpy(b_ioB_gpu, b_ioB, m_batch_size*m_hidden_size*(sizeof(float)), cudaMemcpyHostToDevice); 

    cudaMalloc((void**)&W_hi_gpu, sizeof(float)*m_hidden_size*m_hidden_size);
    cudaMemcpy(W_hi_gpu, W_hi, m_hidden_size*m_hidden_size*(sizeof(float)), cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&W_hf_gpu, sizeof(float)*m_hidden_size*m_hidden_size);
    cudaMemcpy(W_hf_gpu, W_hf, m_hidden_size*m_hidden_size*(sizeof(float)), cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&W_hg_gpu, sizeof(float)*m_hidden_size*m_hidden_size);
    cudaMemcpy(W_hg_gpu, W_hg, m_hidden_size*m_hidden_size*(sizeof(float)), cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&W_ho_gpu, sizeof(float)*m_hidden_size*m_hidden_size);
    cudaMemcpy(W_ho_gpu, W_ho, m_hidden_size*m_hidden_size*(sizeof(float)), cudaMemcpyHostToDevice); 

    cudaMalloc((void**)&b_hi_gpu, sizeof(float)*m_hidden_size);
    cudaMemcpy(b_hi_gpu, b_hi, m_hidden_size*(sizeof(float)), cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&b_hf_gpu, sizeof(float)*m_hidden_size);
    cudaMemcpy(b_hf_gpu, b_hf, m_hidden_size*(sizeof(float)), cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&b_hg_gpu, sizeof(float)*m_hidden_size);
    cudaMemcpy(b_hg_gpu, b_hg, m_hidden_size*(sizeof(float)), cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&b_ho_gpu, sizeof(float)*m_hidden_size);
    cudaMemcpy(b_ho_gpu, b_ho, m_hidden_size*(sizeof(float)), cudaMemcpyHostToDevice); 

    cudaMalloc((void**)&b_hiB_gpu, sizeof(float)*m_batch_size*m_hidden_size);
    cudaMemcpy(b_hiB_gpu, b_hiB, m_batch_size*m_hidden_size*(sizeof(float)), cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&b_hfB_gpu, sizeof(float)*m_batch_size*m_hidden_size);
    cudaMemcpy(b_hfB_gpu, b_hfB, m_batch_size*m_hidden_size*(sizeof(float)), cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&b_hgB_gpu, sizeof(float)*m_batch_size*m_hidden_size);
    cudaMemcpy(b_hgB_gpu, b_hgB, m_batch_size*m_hidden_size*(sizeof(float)), cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&b_hoB_gpu, sizeof(float)*m_batch_size*m_hidden_size);
    cudaMemcpy(b_hoB_gpu, b_hoB, m_batch_size*m_hidden_size*(sizeof(float)), cudaMemcpyHostToDevice); 

    cudaMalloc((void**)&h_t_gpu, sizeof(float)*m_batch_size*m_hidden_size);
    cudaMalloc((void**)&h_t_minus_1_gpu, sizeof(float)*m_batch_size*m_hidden_size);
    cudaMalloc((void**)&c_t_gpu, sizeof(float)*m_batch_size*m_hidden_size);
    cudaMalloc((void**)&c_t_minus_1_gpu, sizeof(float)*m_batch_size*m_hidden_size);

    cudaMalloc((void**)&W_nn_gpu, sizeof(float)*m_hidden_size);
    cudaMalloc((void**)&b_nn_gpu, sizeof(float));
    cudaMalloc((void**)&b_nnB_gpu, sizeof(float)*m_batch_size);

    cudaMalloc((void**)&m_gpu_buff_ptr, sizeof(float)*m_batch_size*m_input_size);
    //cudaDeviceSynchronize();
}

void LSTM::move_inputs_to_gpu(const float* x, int num_inputs){
    printf("Moving Inputs to GPU\n");
    cudaMalloc((void**)&m_gpu_input_ptr, sizeof(float)*num_inputs*m_seq_length*m_input_size);
    cudaMemcpy(m_gpu_input_ptr, x, num_inputs*m_seq_length*m_input_size*(sizeof(float)), cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&m_gpu_output_ptr, sizeof(float)*num_inputs);
    //cudaDeviceSynchronize();
}

void LSTM::move_outputs_to_cpu(float* x, int num_inputs){
    cudaMemcpy(x, m_gpu_output_ptr, num_inputs*(sizeof(float)), cudaMemcpyDeviceToHost); 
    //cudaDeviceSynchronize();
}
