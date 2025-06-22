// Implementing MLP for MNIST in GGML 
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-metal.h"
#include <numeric>

#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream> 
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>

#define MNIST_NINPUT 784 
#define MNIST_NCLASSES 10 
#define HIDDEN_SIZE 128
#define N_LAYERS 2 
#define BATCH_SIZE 1 

#define MODEL_PRECISION GGML_TYPE_F32


// taken from examples/simple-backend.cpp 
static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}


// parameter struct 
// BELLO ALL 
struct mlp_model {
    struct ggml_context * ctx; 
    struct ggml_tensor * fc1_weight; 
    struct ggml_tensor * fc1_bias; 
    struct ggml_tensor * fc2_weight; 
    struct ggml_tensor * fc2_bias; 
    struct ggml_tensor * logits; 
    struct ggml_tensor * images; 
    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer;

    struct ggml_context * ctx_gguf    = nullptr;
    struct ggml_context * ctx_static  = nullptr;
    struct ggml_context * ctx_compute = nullptr;
    ggml_backend_buffer_t buf_gguf    = nullptr;
    ggml_backend_buffer_t buf_static  = nullptr;

}; 

const char * file_name = "./data/model.bin";


// .bin file format:
// for each tensor:
// <num_dims> <dim0> <dim1> ... <dimN> <data>
// where <data> is a contiguous array of floats
// <num_dims> is the number of dimensions of the tensor
// <dim0> <dim1> ... <dimN> are the dimensions of the tensor in reverse order
// <data> is a contiguous array of floats
// fc1 weight, fc1 bias, fc2 weight, fc2 bias

// TODO: READ IN PRE-TRAINED MNIST WEIGHTS FROM .bin GGUF file to the float arrays in row major order

bool read_weights(std::string file_name, float * fc1_weight, float * fc1_bias, float * fc2_weight, float * fc2_bias) {
    // for now, just set all the weights to 0
    memset(fc1_weight, 1, MNIST_NINPUT * HIDDEN_SIZE * sizeof(float));
    memset(fc1_bias, 1, HIDDEN_SIZE * sizeof(float));
    memset(fc2_weight, 1, HIDDEN_SIZE * MNIST_NCLASSES * sizeof(float));
    memset(fc2_bias, 1, MNIST_NCLASSES * sizeof(float));
    std::vector<float*> blocks = {fc1_weight, fc1_bias, fc2_weight, fc2_bias};

    auto fin = std::ifstream(file_name, std::ios::binary);
    for (float* block : blocks){
        std::cout << "reading block: " << block << std::endl;
        int tensor_dims;
        fin.read((char *) &tensor_dims, sizeof(int)); 
        std::vector<int> values(tensor_dims);
        fin.read((char*)values.data(), sizeof(int) * tensor_dims);
        int tensor_size = std::accumulate(values.begin(), values.end(), 1, std::multiplies<int>());
        fin.read((char*) block, sizeof(float) * tensor_size);

    }

    fin.close(); 

    return true;

}

bool load_model(mlp_model & model, float * fc1_weight, float * fc1_bias, float * fc2_weight, float * fc2_bias) {
    

    // STEPS TO LOAD IN A GGML MODEL THAT CAN BE RUN 

    // STEP 1: 
    // figure out the backend buffer size in terms of the model tensors 
    size_t buffer_size = 0; 
    {
        // fc1_weight
        buffer_size = MNIST_NINPUT * HIDDEN_SIZE * ggml_type_size(MODEL_PRECISION);
        // fc1_bias
        buffer_size += HIDDEN_SIZE * ggml_type_size(MODEL_PRECISION);
        // fc2_weight
        buffer_size += HIDDEN_SIZE * MNIST_NCLASSES * ggml_type_size(MODEL_PRECISION);
        // fc2_bias
        buffer_size += MNIST_NCLASSES * ggml_type_size(MODEL_PRECISION);
    }
    printf("ggml tensor size: %d bytes\n", (int) sizeof(ggml_tensor));
    printf("backend buffer size: %d bytes\n", (int) buffer_size);

    // STEP 2: create the initialization parameters (memory size of the CPU context buffer)
    int num_tensors = N_LAYERS * 2; 

    // because we don't allocate tensors on the CPU, our memory size is just # of tensors times overhead for each one
    // aka metadata 
    struct ggml_init_params params = {
        .mem_size = ggml_tensor_overhead() * num_tensors, 
        .mem_buffer = NULL, 
        .no_alloc = true, 
    }; 
    model.ctx = ggml_init(params); 

    // STEP 3: SELECT MODEL BACKEND (this is just copy pasted from example code)
    bool use_gpu = false; 


#ifdef GGML_USE_CUBLAS
    if (use_gpu) {
        fprintf(stderr, "%s: using CUDA backend\n", __func__);
        model.backend = ggml_backend_cuda_init();
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
        }
    }
#endif

#ifdef GGML_USE_METAL
    if (use_gpu) {
        fprintf(stderr, "%s: using Metal backend\n", __func__);
        model.backend = ggml_backend_metal_init();
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
        }
    }
#endif

    if(!model.backend) {
        // fallback to CPU backend
        std::cout << "using CPU backend" << std::endl;
        model.backend = ggml_backend_cpu_init();
    }
    // STEP 4: ALLOCATE THE BACKEND BUFFER 
    // returns: ggml_backend_buffer_t
    //model.buffer = ggml_backend_alloc_buffer(model.backend, buffer_size);

    // STEP 5: INITIALIZE THE GGML CONTEXT -- this returns a 
    model.ctx = ggml_init(params); // this returns a struct ggml_context which is a mem buffer + linked list of ggml_objects

    // STEP 6: ADD THE ACTUAL TENSORS TO THE CONTEXT 
    model.fc1_weight = ggml_new_tensor_2d(model.ctx, MODEL_PRECISION, MNIST_NINPUT, HIDDEN_SIZE);
    model.fc1_bias = ggml_new_tensor_1d(model.ctx, MODEL_PRECISION, HIDDEN_SIZE);
    model.fc2_weight = ggml_new_tensor_2d(model.ctx, MODEL_PRECISION, HIDDEN_SIZE, MNIST_NCLASSES);
    model.fc2_bias = ggml_new_tensor_1d(model.ctx, MODEL_PRECISION, MNIST_NCLASSES);

    // initialize images/input tensor 
    // 1. meta-context only (no_alloc = true)
    const size_t meta = BATCH_SIZE * ggml_tensor_overhead();
    ggml_init_params p_static = { meta, nullptr, /*no_alloc=*/true };
    model.ctx_static = ggml_init(p_static);

    // 2. create the tensor(s) in that context
    model.images = ggml_new_tensor_2d(model.ctx_static,
                                    MODEL_PRECISION,
                                    MNIST_NINPUT,          // 784
                                    BATCH_SIZE);// batch

    ggml_set_name(model.images, "images");
    ggml_set_input(model.images);

    // 3. allocate the backing buffer for all tensors in ctx_static
    model.buf_static = ggml_backend_alloc_ctx_tensors(model.ctx_static,
                                                    model.backend);


    // STEP 4 ACTUAL: ALLOCATE THE BUFFER ACCORDING TO THE TENSORS ON THE CONTEXT 
    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend); 

    ggml_backend_tensor_set(model.fc1_weight, fc1_weight, 0, ggml_nbytes(model.fc1_weight));
    ggml_backend_tensor_set(model.fc1_bias, fc1_bias, 0, ggml_nbytes(model.fc1_bias));
    ggml_backend_tensor_set(model.fc2_weight, fc2_weight, 0, ggml_nbytes(model.fc2_weight));
    ggml_backend_tensor_set(model.fc2_bias, fc2_bias, 0, ggml_nbytes(model.fc2_bias));

    return true; 
}

// build the compute graph to perform a matrix multiplication
struct ggml_cgraph * build_graph(mlp_model& model) {
    static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
    };

    // create a temporally context to build the graph
    struct ggml_context * ctx0 = ggml_init(params0);

    struct ggml_cgraph  * gf = ggml_new_graph(ctx0);



    // Build the MLP: fc1 -> relu -> fc2
    struct ggml_tensor * fc1 = ggml_relu(ctx0, ggml_add(ctx0,
            ggml_mul_mat(ctx0, model.fc1_weight, model.images),
            model.fc1_bias));
    
    struct ggml_tensor * results = ggml_add(ctx0,
            ggml_mul_mat(ctx0, model.fc2_weight, fc1),
            model.fc2_bias);

    // build operations nodes
    ggml_build_forward_expand(gf, results);

    // delete the temporally context used to build the graph
    ggml_free(ctx0);
    return gf;
}


// output is in model.logits 
struct ggml_tensor* compute(mlp_model & model, ggml_gallocr_t allocr) {
    // reset the allocator to free all the memory allocated during the previous inference

    struct ggml_cgraph * gf = build_graph(model);
    ggml_graph_dump_dot(gf, NULL, "mnist_graph.dot"); 
    // allocate tensors
    ggml_gallocr_alloc_graph(allocr, gf);

    std::cout << "HIIII" << std::endl;

    int n_threads = 1; // number of threads to perform some operations with multi-threading

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

    std::cout << "running graph" << std::endl;

    ggml_backend_graph_compute(model.backend, gf);

    return ggml_graph_node(gf, -1);

    // in this case, the output tensor is the last one in the graph
}

// Load the first <images->ne[1]> samples from a plaintext MNIST file (label + 784 pixels)
// into the provided GGML tensor `images` (shape: MNIST_NINPUT x batch). Pixels are
// normalized to the range [0,1]. Return true on success.
bool mnist_batch_load(const std::string & fname,
                        struct ggml_tensor * images,
                        struct ggml_tensor * labels = nullptr) {
    GGML_ASSERT(images);
    GGML_ASSERT(images->type == GGML_TYPE_F32);
    GGML_ASSERT(images->ne[0] == MNIST_NINPUT);

    std::cout << "images->ne[1]: " << images->ne[1] << std::endl;

    const size_t batch = images->ne[1];

    if (labels) {
        GGML_ASSERT(labels->type == GGML_TYPE_F32);
        GGML_ASSERT(labels->ne[0] == MNIST_NCLASSES);
        GGML_ASSERT(labels->ne[1] == (int64_t) batch);
    }

    std::ifstream fin(fname);
    if (!fin) {
        fprintf(stderr, "%s: failed to open %s\n", __func__, fname.c_str());
        return false;
    }

    std::vector<float> img_buf(batch * MNIST_NINPUT);
    std::vector<float> lbl_buf; // one-hot if needed
    if (labels) {
        lbl_buf.resize(batch * MNIST_NCLASSES, 0.0f);
    }

    std::string line;
    for (size_t b = 0; b < batch; ++b) {
        if (!std::getline(fin, line)) {
            fprintf(stderr, "%s: reached EOF after %zu samples (need %zu)\n", __func__, b, batch);
            return false;
        }
        std::stringstream ss(line);
        int label_int;
        ss >> label_int;
        for (size_t i = 0; i < MNIST_NINPUT; ++i) {
            int px;
            ss >> px;
            if (!ss) {
                fprintf(stderr, "%s: parse error on line %zu, pixel %zu\n", __func__, b, i);
                return false;
            }
            img_buf[b*MNIST_NINPUT + i] = px / 255.0f;
        }
        if (labels) {
            for (size_t c = 0; c < MNIST_NCLASSES; ++c) {
                lbl_buf[b*MNIST_NCLASSES + c] = (c == (size_t)label_int) ? 1.0f : 0.0f;
            }
        }
    }

    ggml_backend_tensor_set(images, img_buf.data(), 0, img_buf.size()*ggml_type_size(MODEL_PRECISION));
    if (labels) {
        ggml_backend_tensor_set(labels, lbl_buf.data(), 0, lbl_buf.size()*ggml_type_size(MODEL_PRECISION));
    }

    return true;
}

void print_image(struct ggml_tensor * images) {
    std::cout << "images->ne[1]: " << images->ne[1] << std::endl;
    std::cout << "images->ne[0]: " << images->ne[0] << std::endl;
    std::cout << "images->data: " << images->data << std::endl;
    
    // Get the image data from the tensor
    std::vector<float> img_data(ggml_nelements(images));
    ggml_backend_tensor_get(images, img_data.data(), 0, ggml_nbytes(images));
    
    // Print the first image (assuming batch size >= 1)
    std::cout << "\nFirst MNIST image (28x28):" << std::endl;
    for (int row = 0; row < 28; ++row) {
        for (int col = 0; col < 28; ++col) {
            float pixel = img_data[row * 28 + col];
            // Convert normalized [0,1] back to [0,255] for display
            int intensity = (int)(pixel * 255);
            // Use ASCII characters to represent intensity
            if (intensity > 200) {
                std::cout << "##";
            } else if (intensity > 150) {
                std::cout << "**";
            } else if (intensity > 100) {
                std::cout << "++";
            } else if (intensity > 50) {
                std::cout << "..";
            } else {
                std::cout << "  ";
            }
        }
        std::cout << std::endl;
    }
    
    // Print some pixel values for debugging
    std::cout << "\nFirst 10 pixel values (normalized): ";
    for (int i = 0; i < 10; ++i) {
        std::cout << img_data[i] << " ";
    }
    std::cout << std::endl;
}

void print_weights(mlp_model & model) {
    std::cout << "=== FC1 WEIGHTS ===" << std::endl;
    std::vector<float> fc1_w_data(ggml_nelements(model.fc1_weight));
    ggml_backend_tensor_get(model.fc1_weight, fc1_w_data.data(), 0, ggml_nbytes(model.fc1_weight));
    
    std::cout << "FC1 weights shape: " << model.fc1_weight->ne[0] << " x " << model.fc1_weight->ne[1] << std::endl;
    std::cout << "First 10 FC1 weights: ";
    for (int i = 0; i < 10 && i < (int)fc1_w_data.size(); ++i) {
        std::cout << fc1_w_data[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "\n=== FC1 BIAS ===" << std::endl;
    std::vector<float> fc1_b_data(ggml_nelements(model.fc1_bias));
    ggml_backend_tensor_get(model.fc1_bias, fc1_b_data.data(), 0, ggml_nbytes(model.fc1_bias));
    
    std::cout << "FC1 bias shape: " << model.fc1_bias->ne[0] << std::endl;
    std::cout << "First 10 FC1 biases: ";
    for (int i = 0; i < 10 && i < (int)fc1_b_data.size(); ++i) {
        std::cout << fc1_b_data[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "\n=== FC2 WEIGHTS ===" << std::endl;
    std::vector<float> fc2_w_data(ggml_nelements(model.fc2_weight));
    ggml_backend_tensor_get(model.fc2_weight, fc2_w_data.data(), 0, ggml_nbytes(model.fc2_weight));
    
    std::cout << "FC2 weights shape: " << model.fc2_weight->ne[0] << " x " << model.fc2_weight->ne[1] << std::endl;
    std::cout << "First 10 FC2 weights: ";
    for (int i = 0; i < 10 && i < (int)fc2_w_data.size(); ++i) {
        std::cout << fc2_w_data[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "\n=== FC2 BIAS ===" << std::endl;
    std::vector<float> fc2_b_data(ggml_nelements(model.fc2_bias));
    ggml_backend_tensor_get(model.fc2_bias, fc2_b_data.data(), 0, ggml_nbytes(model.fc2_bias));
    
    std::cout << "FC2 bias shape: " << model.fc2_bias->ne[0] << std::endl;
    std::cout << "All FC2 biases: ";
    for (int i = 0; i < (int)fc2_b_data.size(); ++i) {
        std::cout << fc2_b_data[i] << " ";
    }
    std::cout << std::endl;
}

int main(void) {

    float fc1_weight[MNIST_NINPUT * HIDDEN_SIZE];
    float fc1_bias[HIDDEN_SIZE];
    float fc2_weight[HIDDEN_SIZE * MNIST_NCLASSES];
    float fc2_bias[MNIST_NCLASSES];

    read_weights(file_name, fc1_weight, fc1_bias, fc2_weight, fc2_bias);

    mlp_model model; 
    // read in images 

    load_model(model, fc1_weight, fc1_bias, fc2_weight, fc2_bias);

    print_weights(model);

    if (!mnist_batch_load("/Users/kailashr/ggml/examples/my_mlp/mnist_raw.txt", model.images)) {
        fprintf(stderr, "failed to load images\n");
        return 1;
    }

    print_image(model.images);


    // STEP 7: BUILD THE COMPUTE GRAPH 
    
    ggml_gallocr_t allocr = NULL; 
    {
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        struct ggml_cgraph * gf = build_graph(model);
        ggml_gallocr_reserve(allocr, gf);
        size_t mem_size = ggml_gallocr_get_buffer_size(allocr, 0);
        fprintf(stderr, "compute buffer size: %.2f KB\n", mem_size/1024.0);

        // reserve function already does this? 
        //ggml_gallocr_alloc_graph(allocr, gf);
    }

    struct ggml_tensor* result_node = compute(model, allocr); 
    model.logits = result_node; 

    // create a array to print result
    std::vector<float> out_data(ggml_nelements(model.logits));

    // bring the data from the backend memory
    ggml_backend_tensor_get(model.logits, out_data.data(), 0, ggml_nbytes(model.logits));

    // print the result 
    for (int i = 0; i < MNIST_NCLASSES; i++) {
        printf("%.2f ", out_data[i]);
    }
    printf("\n");


    // STEP WHATEVER THE FOOK: FREE EVERYTHING 
    // release backend memory used for computation
    ggml_gallocr_free(allocr);

    // free memory
    ggml_free(model.ctx);

    // release backend memory and free backend
    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);


    return 0; 
}



