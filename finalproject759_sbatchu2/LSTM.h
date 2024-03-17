#include <stdio.h>
#include <iostream>
#include <math.h>
#include <bits/stdc++.h>
#include <string.h>

class LSTM {

        //input x_t weight matrices, size : hidden_size x input_size
	float* W_ii;
	float* W_if;
        float* W_ig;
        float* W_io;

	//input x_t bias matrices, size : hidden_size
	float* b_ii;
	float* b_if;
        float* b_ig;
        float* b_io;

	//input x_t bias matrices, size : m_batch_size X hidden_size : replicated batch size times
	float* b_iiB;
	float* b_ifB;
        float* b_igB;
        float* b_ioB;

	//hidden state weight matrices, size : hidden_size X hidden_size
	float* W_hi;
        float* W_hf;
        float* W_hg;
	float* W_ho;

	//hidden state bias matrices, size : hidden_size
	float* b_hi;
	float* b_hf;
        float* b_hg;
        float* b_ho;

	//hidden state bias matrices, size : Batch_size X hidden_size : replicated batch size times
	float* b_hiB;
	float* b_hfB;
        float* b_hgB;
        float* b_hoB;

	//intermediate gate outputs : input gate, output gate, forget gate, tanh(used in controlling the range), size : Batch_size X hidden_size
	float* i_t;
        float* f_t;
        float* g_t;
        float* o_t;

	//two buffers each for cell state, hidden state. All we need is (t-1) timestamp and (t) timestamp, size : Batch_size X hidden_size
	float* c_t_minus_1;
        float* h_t_minus_1;
	float* c_t;
	float* h_t;

        //NN Layer weights
	float* W_nn;
        float* b_nn;	
        float* b_nnB; // replicated batch size times

        //GPU params
        //input x_t weight matrices, size : hidden_size x input_size
	float* W_ii_gpu;
	float* W_if_gpu;
        float* W_ig_gpu;
        float* W_io_gpu;

	//input x_t bias matrices, size : hidden_size
	float* b_ii_gpu;
	float* b_if_gpu;
        float* b_ig_gpu;
        float* b_io_gpu;

	//input x_t bias matrices, size : m_batch_size X hidden_size : replicated batch size times
	float* b_iiB_gpu;
	float* b_ifB_gpu;
        float* b_igB_gpu;
        float* b_ioB_gpu;

	//hidden state weight matrices, size : hidden_size X hidden_size
	float* W_hi_gpu;
        float* W_hf_gpu;
        float* W_hg_gpu;
	float* W_ho_gpu;

	//hidden state bias matrices, size : hidden_size
	float* b_hi_gpu;
	float* b_hf_gpu;
        float* b_hg_gpu;
        float* b_ho_gpu;

	//hidden state bias matrices, size : Batch_size X hidden_size : replicated batch size times
	float* b_hiB_gpu;
	float* b_hfB_gpu;
        float* b_hgB_gpu;
        float* b_hoB_gpu;

	//intermediate gate outputs : input gate, output gate, forget gate, tanh(used in controlling the range), size : Batch_size X hidden_size
	float* i_t_gpu;
        float* f_t_gpu;
        float* g_t_gpu;
        float* o_t_gpu;

	//two buffers each for cell state, hidden state. All we need is (t-1) timestamp and (t) timestamp, size : Batch_size X hidden_size
	float* c_t_minus_1_gpu;
        float* h_t_minus_1_gpu;
	float* c_t_gpu;
	float* h_t_gpu;

        //NN Layer weights
	float* W_nn_gpu;
        float* b_nn_gpu;	
        float* b_nnB_gpu; // replicated batch size times
	

	int m_input_size, m_hidden_size, m_seq_length, m_batch_size;  // m_input_size : num features in a input
	char* m_filename;
	char* m_device;
	int m_block_dim;
	char* m_prog_model;
        
	//GPU Input pointer
	float* m_gpu_input_ptr;
	float* m_gpu_buff_ptr;  //Buffer to hold batch size inputs of each element of the sequence
        //GPU Output pointer
	float* m_gpu_output_ptr;

        //Transpose
        //A : pXq, C : qXp
        void mat_transpose(const float* A, float* C, int p, int q);
        
	void transpose_all_params();

	//replicate biases for addition
	void replicate(const float* A, float* C, int length, int batch_size);

	void replicate_all_biases();

	//A : pXq, B : qXr, C : pXr
	void mat_mul(const float* A, const float* B, float* C, int p, int q, int r);  //wrapper
	void mat_mul_cpu(const float* A, const float* B, float* C, int p, int q, int r);   //cpu
	void mat_mul_cpu_openmp(const float* A, const float* B, float* C, int p, int q, int r);  //cpu openmp
	void mat_mul_cpu_mpi(const float* A, const float* B, float* C, int p, int q, int r);  //mpi
	void mat_mul_gpu(const float* A, const float* B, float* C, int p, int q, int r);   //gpu
	
	//A : pXq, B : qXr, C : pXr
	void mat_mac(const float* A, const float* B, float* C, int p, int q, int r);
	void mat_mac_cpu(const float* A, const float* B, float* C, int p, int q, int r);
	void mat_mac_cpu_openmp(const float* A, const float* B, float* C, int p, int q, int r);
	void mat_mac_cpu_mpi(const float* A, const float* B, float* C, int p, int q, int r);
	void mat_mac_gpu(const float* A, const float* B, float* C, int p, int q, int r);
	
	//A : pXq, B : pXq, C : pXq
	void mat_add(const float* A, const float* B, float* C, int p, int q);
	void mat_add_cpu(const float* A, const float* B, float* C, int p, int q);
	void mat_add_cpu_openmp(const float* A, const float* B, float* C, int p, int q);
	void mat_add_cpu_mpi(const float* A, const float* B, float* C, int p, int q);
	void mat_add_gpu(const float* A, const float* B, float* C, int p, int q);

	//sigmoid A : pXq, C : pXq
	void mat_sgm(const float* A, float* C, int p, int q);
	void mat_sgm_cpu(const float* A, float* C, int p, int q);
	void mat_sgm_cpu_openmp(const float* A, float* C, int p, int q);
	void mat_sgm_cpu_mpi(const float* A, float* C, int p, int q);
	void mat_sgm_gpu(const float* A, float* C, int p, int q);

        //tanh A : pXq, C : pXq
	void mat_tanh(const float* A, float* C, int p, int q);
	void mat_tanh_cpu(const float* A, float* C, int p, int q);
	void mat_tanh_cpu_openmp(const float* A, float* C, int p, int q);
	void mat_tanh_cpu_mpi(const float* A, float* C, int p, int q);
	void mat_tanh_gpu(const float* A, float* C, int p, int q);

	//Hadamard or element wise product : "op" as part of README. A : pXq, B : pXq, C : pXq
	void mat_hadamard(const float* A, const float* B, float* C, int p, int q);
	void mat_hadamard_cpu(const float* A, const float* B, float* C, int p, int q);
	void mat_hadamard_cpu_openmp(const float* A, const float* B, float* C, int p, int q);
	void mat_hadamard_cpu_mpi(const float* A, const float* B, float* C, int p, int q);
	void mat_hadamard_gpu(const float* A, const float* B, float* C, int p, int q);

	//Hadamard or element wise product : "op" as part of README. A : pXq, B : pXq, C : pXq
	void mat_hadamard_mac(const float* A, const float* B, float* C, int p, int q);
	void mat_hadamard_mac_cpu(const float* A, const float* B, float* C, int p, int q);
	void mat_hadamard_mac_cpu_openmp(const float* A, const float* B, float* C, int p, int q);
	void mat_hadamard_mac_cpu_mpi(const float* A, const float* B, float* C, int p, int q);
	void mat_hadamard_mac_gpu(const float* A, const float* B, float* C, int p, int q);

	//LSTM cell operation at a time stamp
	void lstm_operation(const float* h_t_minus_1, const float* c_t_minus_1, const float* x_t, float* c_t, float* h_t);

	//LSTM iterations over sequence length
	float* lstm_sequence(const float* x);

	//NN Layer
	void lstm_nn_layer(const float* x, float* y);

	//Print Matrix
	void print_matrix(const float* A, int p, int q);

	//Reset network states for every sequence 
	void reset_all_network_states();

	//move batch size date to buffer - cache hit rate increase since we do input access
	void move_gpu_input_data_to_buffer(const float* A, float* C);

	public :
	   //Forward pass
	   void forward_pass(const float* x, float* y, int num_inputs);

	   //move params to gpu
	   void move_params_to_gpu();

	   void move_inputs_to_gpu(const float* x, int num_inputs);
	   void move_outputs_to_cpu(float* x, int num_inputs);
          
	   //Load Weights
	   void load_weights();
        
           //print Weights
	   void print_weights();
           
	   //Constructor
	   LSTM(int input_size, int hidden_size, int seq_length, int batch_size, const char* filename, const char* device, int block_dim, const char* prog_model);
        
	   //Destructor
	   ~LSTM();

};
