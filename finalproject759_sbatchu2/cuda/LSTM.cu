#include "LSTM.cuh"
using namespace std;

LSTM::LSTM(int input_size, int hidden_size, int seq_length, int batch_size, const char* filename, const char* device, int block_dim, const char* prog_model) {
     printf("Constructor Called\n");  
     m_input_size = input_size;
     m_hidden_size = hidden_size;
     m_seq_length = seq_length;
     m_filename = (char*) filename;
     m_batch_size = batch_size; 
     m_device = (char*)device;
     m_block_dim = block_dim;
     m_prog_model = (char*)prog_model;

     //Order : m_hidden_size X m_input_size
     W_ii = new float[m_input_size*m_hidden_size];
     W_if = new float[m_input_size*m_hidden_size];
     W_ig = new float[m_input_size*m_hidden_size];
     W_io = new float[m_input_size*m_hidden_size];
     for(int i=0; i<m_input_size*m_hidden_size; i++) {
         W_ii[i] = 0.0;
	 W_if[i] = 0.0;
	 W_ig[i] = 0.0;
	 W_io[i] = 0.0;
     }

     b_ii = new float[m_hidden_size];
     b_if = new float[m_hidden_size];
     b_ig = new float[m_hidden_size];
     b_io = new float[m_hidden_size];
     b_iiB = new float[m_batch_size*m_hidden_size];
     b_ifB = new float[m_batch_size*m_hidden_size];
     b_igB = new float[m_batch_size*m_hidden_size];
     b_ioB = new float[m_batch_size*m_hidden_size];
     for(int i=0; i<m_hidden_size; i++) {
         b_ii[i] = 0.0;
	 b_if[i] = 0.0;
	 b_ig[i] = 0.0;
	 b_io[i] = 0.0;
     }


     W_hi = new float[m_hidden_size*m_hidden_size];
     W_hf = new float[m_hidden_size*m_hidden_size];
     W_hg = new float[m_hidden_size*m_hidden_size];
     W_ho = new float[m_hidden_size*m_hidden_size];
     for(int i=0; i<m_hidden_size*m_hidden_size; i++){
         W_hi[i] = 0.0;
	 W_hf[i] = 0.0;
	 W_hg[i] = 0.0;
	 W_ho[i] = 0.0;
     }

     b_hi = new float[m_hidden_size];
     b_hf = new float[m_hidden_size];
     b_hg = new float[m_hidden_size];
     b_ho = new float[m_hidden_size];
     b_hiB = new float[m_batch_size*m_hidden_size];
     b_hfB = new float[m_batch_size*m_hidden_size];
     b_hgB = new float[m_batch_size*m_hidden_size];
     b_hoB = new float[m_batch_size*m_hidden_size];
     for(int i=0; i<m_hidden_size; i++){
         b_hi[i] = 0.0;
	 b_hf[i] = 0.0;
	 b_hg[i] = 0.0;
	 b_ho[i] = 0.0;
     }

     i_t = new float[m_batch_size*m_hidden_size];
     f_t = new float[m_batch_size*m_hidden_size];
     g_t = new float[m_batch_size*m_hidden_size];
     o_t = new float[m_batch_size*m_hidden_size];
     for(int i=0; i<m_batch_size*m_hidden_size; i++){
         i_t[i] = 0.0;
	 f_t[i] = 0.0;
	 g_t[i] = 0.0;
	 o_t[i] = 0.0;
     }
     
     c_t_minus_1 = new float[m_batch_size*m_hidden_size];
     h_t_minus_1 = new float[m_batch_size*m_hidden_size];
     c_t = new float[m_batch_size*m_hidden_size];
     h_t = new float[m_batch_size*m_hidden_size];
     for(int i=0; i< m_batch_size*m_hidden_size; i++) {
         c_t_minus_1[i] = 0.0;
	 h_t_minus_1[i] = 0.0;
	 c_t[i] = 0.0;
	 h_t[i] = 0.0;
     }

     W_nn = new float[m_hidden_size];
     b_nn = new float;
     b_nnB = new float[m_batch_size];

     }

LSTM::~LSTM(){
    delete [] W_ii;
    W_ii = nullptr;    
    delete [] W_if;
    W_if = nullptr;    
    delete [] W_ig;
    W_ig = nullptr;    
    delete [] W_io;
    W_io = nullptr;    
    delete [] W_hi;
    W_hi = nullptr;    
    delete [] W_hf;
    W_hf = nullptr;    
    delete [] W_hg;
    W_hg = nullptr;    
    delete [] W_ho;
    W_ho = nullptr;    
    delete [] b_ii;
    b_ii = nullptr;    
    delete [] b_if;
    b_if = nullptr;    
    delete [] b_ig;
    b_ig = nullptr;    
    delete [] b_io;
    b_io = nullptr;    
    delete [] b_hi;
    b_hi = nullptr;    
    delete [] b_hf;
    b_hf = nullptr;    
    delete [] b_hg;
    b_hg = nullptr;    
    delete [] b_ho;
    b_ho = nullptr;
    delete [] c_t_minus_1;
    c_t_minus_1 = nullptr;  
    delete [] h_t_minus_1;
    h_t_minus_1 = nullptr;
    delete [] c_t;
    c_t = nullptr;
    delete [] h_t;
    h_t = nullptr; 


   /* delete [] W_ii_gpu;
    W_ii_gpu = nullptr;    
    delete [] W_if_gpu;
    W_if_gpu = nullptr;    
    delete [] W_ig_gpu;
    W_ig_gpu = nullptr;    
    delete [] W_io_gpu;
    W_io_gpu = nullptr;    
    delete [] W_hi_gpu;
    W_hi_gpu = nullptr;    
    delete [] W_hf_gpu;
    W_hf_gpu = nullptr;    
    delete [] W_hg_gpu;
    W_hg_gpu = nullptr;    
    delete [] W_ho_gpu;
    W_ho_gpu = nullptr;    
    delete [] b_ii_gpu;
    b_ii_gpu = nullptr;    
    delete [] b_if_gpu;
    b_if_gpu = nullptr;    
    delete [] b_ig_gpu;
    b_ig_gpu = nullptr;    
    delete [] b_io_gpu;
    b_io_gpu = nullptr;    
    delete [] b_hi_gpu;
    b_hi_gpu = nullptr;    
    delete [] b_hf_gpu;
    b_hf_gpu = nullptr;    
    delete [] b_hg_gpu;
    b_hg_gpu = nullptr;    
    delete [] b_ho_gpu;
    b_ho_gpu = nullptr;
    delete [] c_t_minus_1_gpu;
    c_t_minus_1_gpu = nullptr;  
    delete [] h_t_minus_1_gpu;
    h_t_minus_1_gpu = nullptr;
    delete [] c_t_gpu;
    c_t_gpu = nullptr;
    delete [] h_t_gpu;
    h_t_gpu = nullptr;*/

}


void LSTM::print_matrix(const float* A, int p, int q){
	printf("Printing Matrix\n");
    for(int i=0; i<p; i++){
	    for(int j=0; j<q; j++){
	     printf("%f ",A[i*q+j]);
	    }
	    printf("\n");
    }

}

//Transpose
//A : pXq, C : qXp
void LSTM::mat_transpose(const float* A, float* C, int p, int q){
    if((p!=1)&&(q!=1)){
        printf("Transpose Matrix\n");
        float* temp = new float [q*p];
        for(int i=0; i<p; i++){
            for(int j=0; j<q; j++){
                temp[j*p + i] = A[i*q + j];	
            }
        }
        for(int i=0; i<q; i++){
            for(int j=0; j<p; j++){
                C[i*p + j] = temp[i*p + j];	
            }
        }

        delete [] temp;
        temp=nullptr;
    }
}


 
//LSTM cell operation at a time stamp
void LSTM::lstm_operation(const float* Ah_t_minus_1, const float* Ac_t_minus_1, const float* x_t, float* Ac_t, float* Ah_t){

	printf("LSTM Operation\n");
       
         float* x_t_gpu = m_gpu_input_ptr;	
	     printf("LSTM Operation : GPU\n");
	     // computing i_t
             mat_mul_gpu(x_t, W_ii_gpu, i_t_gpu, m_batch_size, m_input_size, m_hidden_size);
	     mat_mac_gpu(Ah_t_minus_1, W_hi_gpu, i_t_gpu, m_batch_size, m_hidden_size, m_hidden_size);
	     mat_add_gpu(i_t_gpu, b_iiB_gpu, i_t_gpu, m_batch_size, m_hidden_size);
	     mat_add_gpu(i_t_gpu, b_hiB_gpu, i_t_gpu, m_batch_size, m_hidden_size);
	     mat_sgm_gpu(i_t_gpu, i_t_gpu, m_batch_size,m_hidden_size);

	     // computing f_t
	     mat_mul_gpu(x_t_gpu, W_if_gpu, f_t_gpu, m_batch_size, m_input_size, m_hidden_size);
	     mat_mac_gpu(Ah_t_minus_1, W_hf_gpu, f_t_gpu, m_batch_size, m_hidden_size, m_hidden_size);
	     mat_add_gpu(f_t_gpu, b_ifB_gpu, f_t_gpu, m_batch_size, m_hidden_size);
	     mat_add_gpu(f_t_gpu, b_hfB_gpu, f_t_gpu, m_batch_size, m_hidden_size);
	     mat_sgm_gpu(f_t_gpu, f_t_gpu, m_batch_size,m_hidden_size);

	     // computing g_t
	     mat_mul_gpu(x_t_gpu, W_ig_gpu, g_t_gpu, m_batch_size, m_input_size, m_hidden_size);
	     mat_mac_gpu(Ah_t_minus_1, W_hg_gpu, g_t_gpu, m_batch_size, m_hidden_size, m_hidden_size);
	     mat_add_gpu(g_t_gpu, b_igB_gpu, g_t_gpu, m_batch_size, m_hidden_size);
	     mat_add_gpu(g_t_gpu, b_hgB_gpu, g_t_gpu, m_batch_size, m_hidden_size);
	     mat_tanh_gpu(g_t_gpu, g_t_gpu, m_batch_size, m_hidden_size);

	     //computing o_t
	     mat_mul_gpu(x_t_gpu, W_io_gpu, o_t_gpu, m_batch_size, m_input_size, m_hidden_size);
	     mat_mac_gpu(Ah_t_minus_1, W_ho_gpu, o_t_gpu, m_batch_size, m_hidden_size, m_hidden_size);
	     mat_add_gpu(o_t_gpu, b_ioB_gpu, o_t_gpu, m_batch_size, m_hidden_size);
	     mat_add_gpu(o_t_gpu, b_hoB_gpu, o_t_gpu, m_batch_size, m_hidden_size);
	     mat_sgm_gpu(o_t_gpu, o_t_gpu, m_batch_size, m_hidden_size);

	     //computing c_t
	     mat_hadamard_gpu(Ac_t_minus_1, f_t_gpu, c_t_gpu, m_batch_size, m_hidden_size);
	     mat_hadamard_mac_gpu(i_t_gpu, g_t_gpu, c_t_gpu, m_batch_size, m_hidden_size);

	     //computing h_t
	     mat_tanh_gpu(Ac_t, Ah_t, m_batch_size, m_hidden_size);
	     mat_hadamard_gpu(Ah_t, o_t_gpu, Ah_t, m_batch_size, m_hidden_size);

}


void LSTM::load_weights(){
  printf("Loading Weights\n"); // W_i*, W_h*, b_i*, b_h*
  ifstream in(m_filename); 
  int weight_index = 0;
  int i=0;
  while(!in.eof() && (weight_index<18)) {
	if(weight_index == 0)
	       in >> W_ii[i];
	if(weight_index == 1)
	       in >> W_if[i];
	if(weight_index == 2)
	       in >> W_ig[i];
	if(weight_index == 3)
	       in >> W_io[i];
	if(weight_index == 4)
	       in >> W_hi[i];
	if(weight_index == 5)
	       in >> W_hf[i];
	if(weight_index == 6)
	       in >> W_hg[i];
	if(weight_index == 7)
	       in >> W_ho[i];
	if(weight_index == 8)
	       in >> b_ii[i];
	if(weight_index == 9)
	       in >> b_if[i];
	if(weight_index == 10)
	       in >> b_ig[i];
	if(weight_index == 11)
	       in >> b_io[i];
	if(weight_index == 12)
	       in >> b_hi[i];
	if(weight_index == 13)
	       in >> b_hf[i];
	if(weight_index == 14)
	       in >> b_hg[i];
	if(weight_index == 15)
	       in >> b_ho[i];
	if(weight_index == 16)
	       in >> W_nn[i];
	if(weight_index == 17)
	       in >> b_nn[i];

	if(weight_index < 8) // first 8 indexes -> weights
	{
		if(i == (((weight_index<4) ? m_input_size : m_hidden_size) *m_hidden_size-1)){
		    i=0;
		    weight_index++;
		}
		else
		    i++;
	
	} else if((weight_index >=8) && (weight_index<17))  //biases, except index 16 is NN weight
	{
		if(i == (m_hidden_size-1)){
		    i=0;
		    weight_index++;
		}
		else
		    i++;
	}else if(weight_index == 17) // NN bias
	{
	      weight_index++;
	}
  }
    move_params_to_gpu();
}
void LSTM::transpose_all_params(){
    mat_transpose(W_ii, W_ii, m_hidden_size, m_input_size);
    mat_transpose(W_if, W_if, m_hidden_size, m_input_size);
    mat_transpose(W_ig, W_ig, m_hidden_size, m_input_size);
    mat_transpose(W_io, W_io, m_hidden_size, m_input_size);
    mat_transpose(W_hi, W_hi, m_hidden_size, m_hidden_size);
    mat_transpose(W_hf, W_hf, m_hidden_size, m_hidden_size);
    mat_transpose(W_hg, W_hg, m_hidden_size, m_hidden_size);
    mat_transpose(W_ho, W_ho, m_hidden_size, m_hidden_size);
    mat_transpose(b_ii, b_ii, m_hidden_size, 1);
    mat_transpose(b_if, b_if, m_hidden_size, 1);
    mat_transpose(b_ig, b_ig, m_hidden_size, 1);
    mat_transpose(b_io, b_io, m_hidden_size, 1);
    mat_transpose(b_hi, b_hi, m_hidden_size, 1);
    mat_transpose(b_hf, b_hf, m_hidden_size, 1);
    mat_transpose(b_hg, b_hg, m_hidden_size, 1);
    mat_transpose(b_ho, b_ho, m_hidden_size, 1);
    mat_transpose(W_nn, W_nn, 1, m_hidden_size); //output dimension here is 1
    mat_transpose(b_nn, b_nn, 1, 1); //output dimension here is 1
}

//replicating a row to batch_size times : output matrix is m_batch_size X size(bias)
void LSTM::replicate(const float* A, float* C, int length, int batch_size){
   for(int i=0; i<batch_size*length; i++)
   {
      C[i] = A[i%length];
   
   }
}

void LSTM::replicate_all_biases(){

    replicate(b_ii, b_iiB, m_hidden_size, m_batch_size);
    replicate(b_if, b_ifB, m_hidden_size, m_batch_size);
    replicate(b_ig, b_igB, m_hidden_size, m_batch_size);
    replicate(b_io, b_ioB, m_hidden_size, m_batch_size);
    replicate(b_hi, b_hiB, m_hidden_size, m_batch_size);
    replicate(b_hf, b_hfB, m_hidden_size, m_batch_size);
    replicate(b_hg, b_hgB, m_hidden_size, m_batch_size);
    replicate(b_ho, b_hoB, m_hidden_size, m_batch_size);
    replicate(b_nn, b_nnB, 1, m_batch_size);
}

void LSTM::print_weights(){
  printf("Printing Weights\n");
  int weight_index = 0;
  int i=0;
  while(weight_index <18) {
	if(weight_index == 0)
	       printf("W_ii[%d] = %f\n",i, W_ii[i]);
	if(weight_index == 1)
	       printf("W_if[%d] = %f\n",i, W_if[i]);
	if(weight_index == 2)
	       printf("W_ig[%d] = %f\n",i, W_ig[i]);
	if(weight_index == 3)
	       printf("W_io[%d] = %f\n",i, W_io[i]);
	if(weight_index == 4)
	       printf("W_hi[%d] = %f\n",i, W_hi[i]);
	if(weight_index == 5)
	       printf("W_hf[%d] = %f\n",i, W_hf[i]);
	if(weight_index == 6)
	       printf("W_hg[%d] = %f\n",i, W_hg[i]);
	if(weight_index == 7)
	       printf("W_ho[%d] = %f\n",i, W_ho[i]);
	if(weight_index == 8)
	       printf("b_ii[%d] = %f\n",i, b_ii[i]);
	if(weight_index == 9)
	       printf("b_if[%d] = %f\n",i, b_if[i]);
	if(weight_index == 10)
	       printf("b_ig[%d] = %f\n",i, b_ig[i]);
	if(weight_index == 11)
	       printf("b_io[%d] = %f\n",i, b_io[i]);
	if(weight_index == 12)
	       printf("b_hi[%d] = %f\n",i, b_hi[i]);
	if(weight_index == 13)
	       printf("b_hf[%d] = %f\n",i, b_hf[i]);
	if(weight_index == 14)
	       printf("b_hg[%d] = %f\n",i, b_hg[i]);
	if(weight_index == 15)
	       printf("b_ho[%d] = %f\n",i, b_ho[i]);
	if(weight_index == 16)
	       printf("W_nn[%d] = %f\n",i, W_nn[i]);
	if(weight_index == 17)
	       printf("b_nn[%d] = %f\n",i, b_nn[i]);

	if((weight_index < 8))
	{
		if(i == (((weight_index<4) ? m_input_size : m_hidden_size) *m_hidden_size-1)){
		    i=0;
		    weight_index++;
		}
		else
		    i++;
	
	} else if((weight_index >= 8) && (weight_index<17))
	{
		if(i == (m_hidden_size-1)){
		    i=0;
		    weight_index++;
		}
		else
		    i++;
	}else if(weight_index == 17)
	{
	      weight_index ++;
	}
  }

}

 
//LSTM iterations over sequence length
float* LSTM::lstm_sequence(const float* x, int offset){

	float* y;
	printf("LSTM over input sequence\n");

	//first get batch size of inputs - one element of the sequnec
        
	for(int i=0; i<m_seq_length; i++){

	        move_gpu_input_data_to_buffer(x, offset + i*m_input_size, m_gpu_buff_ptr);
                printf("GPU LSTM sequence iteration : %d\n", i);
	        if(i%2==0)
                {
	              printf("Buffering in ct ht\n");
	              lstm_operation(h_t_minus_1_gpu, c_t_minus_1_gpu, m_gpu_buff_ptr, c_t_gpu, h_t_gpu);
	        }else{ 
	      	       printf("Buffering in ct_minus_1 ht_minus_1\n");
	               lstm_operation(h_t_gpu, c_t_gpu, m_gpu_buff_ptr, c_t_minus_1_gpu, h_t_minus_1_gpu);
	         }
	}

	if(m_seq_length%2 == 0)
		y = h_t_minus_1_gpu;
	else
		y = h_t_gpu;


return y;
}

//reset all network states
void LSTM:: reset_all_network_states() {
     for(int i=0; i< m_batch_size*m_hidden_size; i++) {
         c_t_minus_1[i] = 0.0;
         h_t_minus_1[i] = 0.0;
         c_t[i] = 0.0;
         h_t[i] = 0.0;
     }
     for(int i=0; i<m_batch_size*m_hidden_size; i++){
         i_t[i] = 0.0;
         f_t[i] = 0.0;
         g_t[i] = 0.0;
         o_t[i] = 0.0;
     } 
}

void LSTM::forward_pass(const float *x, float* y, int num_inputs){
  	
    printf("Forward Pass\n");
    transpose_all_params();
    replicate_all_biases();
    move_inputs_to_gpu(x, num_inputs);
    for(int i=0; i< num_inputs; i=i+m_batch_size)
    {
      printf("Batch Index : %d, Input index : %d\n",i/m_batch_size, i);
      const float* lstm_input_ptr;
      lstm_input_ptr = m_gpu_input_ptr;
      float* lstm_output_ptr;
      lstm_output_ptr = m_gpu_output_ptr + i;
      if((num_inputs -i) < m_batch_size) 
	     m_batch_size = num_inputs-i;
      float* lstm_output;  
      lstm_output=lstm_sequence(lstm_input_ptr, i*m_seq_length*m_input_size);
      printf("LSTM Sequence Output Matrix\n");
      //print_matrix(lstm_output, m_batch_size, m_hidden_size);
      lstm_nn_layer(lstm_output,lstm_output_ptr);
      reset_all_network_states_gpu();
    }
    move_outputs_to_cpu(y,num_inputs);
}

void LSTM::lstm_nn_layer(const float* x, float* y){
    printf("LSTM NN Layer\n");

     mat_mul_gpu(x, W_nn_gpu,y,m_batch_size,m_hidden_size,1);
     mat_add_gpu(y,b_nnB_gpu,y,m_batch_size,1);
     //mat_sgm(y,y,1,1); // Non linear layer is not a part of NN layer

}

