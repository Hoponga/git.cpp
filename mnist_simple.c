#include "ggml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// MNIST constants
#define IMAGE_SIZE 28*28
#define NUM_CLASSES 10
#define HIDDEN_SIZE 128
#define BATCH_SIZE 32
#define NUM_EPOCHS 5
#define LEARNING_RATE 0.01f

// Simple random number generator
static float rand_float() {
    return (float)rand() / RAND_MAX;
}

static float rand_normal() {
    float u1 = rand_float();
    float u2 = rand_float();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}

// Load MNIST data (simplified - you'd normally read from files)
static void generate_mnist_data(float* images, int* labels, int num_samples) {
    printf("Generating synthetic MNIST data...\n");
    
    for (int i = 0; i < num_samples; i++) {
        // Generate synthetic image data (normally you'd load from MNIST files)
        for (int j = 0; j < IMAGE_SIZE; j++) {
            images[i * IMAGE_SIZE + j] = rand_float();
        }
        
        // Generate synthetic label
        labels[i] = rand() % NUM_CLASSES;
    }
}

// Simple neural network model
struct mnist_model {
    struct ggml_context* ctx;
    
    // Weights and biases
    struct ggml_tensor* fc1_w;  // [HIDDEN_SIZE, IMAGE_SIZE]
    struct ggml_tensor* fc1_b;  // [HIDDEN_SIZE]
    struct ggml_tensor* fc2_w;  // [NUM_CLASSES, HIDDEN_SIZE] 
    struct ggml_tensor* fc2_b;  // [NUM_CLASSES]
};

// Initialize model
static struct mnist_model* mnist_model_init() {
    struct mnist_model* model = malloc(sizeof(struct mnist_model));
    
    // Calculate context size needed
    size_t ctx_size = 0;
    ctx_size += ggml_row_size(GGML_TYPE_F32, HIDDEN_SIZE * IMAGE_SIZE);  // fc1_w
    ctx_size += ggml_row_size(GGML_TYPE_F32, HIDDEN_SIZE);               // fc1_b
    ctx_size += ggml_row_size(GGML_TYPE_F32, NUM_CLASSES * HIDDEN_SIZE); // fc2_w
    ctx_size += ggml_row_size(GGML_TYPE_F32, NUM_CLASSES);               // fc2_b
    ctx_size += 1024*1024; // Extra space for computations
    
    struct ggml_init_params params = {
        .mem_size = ctx_size,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    model->ctx = ggml_init(params);
    
    // Create tensors
    model->fc1_w = ggml_new_tensor_2d(model->ctx, GGML_TYPE_F32, IMAGE_SIZE, HIDDEN_SIZE);
    model->fc1_b = ggml_new_tensor_1d(model->ctx, GGML_TYPE_F32, HIDDEN_SIZE);
    model->fc2_w = ggml_new_tensor_2d(model->ctx, GGML_TYPE_F32, HIDDEN_SIZE, NUM_CLASSES);
    model->fc2_b = ggml_new_tensor_1d(model->ctx, GGML_TYPE_F32, NUM_CLASSES);
    
    // Initialize weights with Xavier/Glorot initialization
    for (int i = 0; i < HIDDEN_SIZE * IMAGE_SIZE; i++) {
        ((float*)model->fc1_w->data)[i] = rand_normal() * sqrtf(2.0f / IMAGE_SIZE);
    }
    
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        ((float*)model->fc1_b->data)[i] = 0.0f;
    }
    
    for (int i = 0; i < NUM_CLASSES * HIDDEN_SIZE; i++) {
        ((float*)model->fc2_w->data)[i] = rand_normal() * sqrtf(2.0f / HIDDEN_SIZE);
    }
    
    for (int i = 0; i < NUM_CLASSES; i++) {
        ((float*)model->fc2_b->data)[i] = 0.0f;
    }
    
    printf("Model initialized with %zu bytes context\n", ctx_size);
    return model;
}

// Forward pass
static struct ggml_tensor* mnist_forward(struct mnist_model* model, struct ggml_context* ctx, struct ggml_tensor* x) {
    // x shape: [BATCH_SIZE, IMAGE_SIZE]
    
    // First layer: fc1_w * x + fc1_b
    struct ggml_tensor* h = ggml_add(ctx,
        ggml_mul_mat(ctx, model->fc1_w, x),  // [HIDDEN_SIZE, BATCH_SIZE]
        ggml_repeat(ctx, model->fc1_b, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HIDDEN_SIZE, BATCH_SIZE))
    );
    
    // ReLU activation
    h = ggml_relu(ctx, h);
    
    // Second layer: fc2_w * h + fc2_b
    struct ggml_tensor* logits = ggml_add(ctx,
        ggml_mul_mat(ctx, model->fc2_w, h),  // [NUM_CLASSES, BATCH_SIZE]
        ggml_repeat(ctx, model->fc2_b, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NUM_CLASSES, BATCH_SIZE))
    );
    
    return logits;
}

// Cross entropy loss
static struct ggml_tensor* cross_entropy_loss(struct ggml_context* ctx, struct ggml_tensor* logits, struct ggml_tensor* targets) {
    // Apply softmax to get probabilities
    struct ggml_tensor* probs = ggml_soft_max(ctx, logits);
    
    // For simplicity, we'll use MSE loss instead of proper cross-entropy
    // In a real implementation, you'd implement proper cross-entropy
    struct ggml_tensor* loss = ggml_mean(ctx, ggml_sqr(ctx, ggml_sub(ctx, probs, targets)));
    
    return loss;
}

// Training step
static float train_step(struct mnist_model* model, float* batch_images, int* batch_labels, int batch_size) {
    // Create computation context
    size_t ctx_size = 1024*1024; // 1MB for computations
    struct ggml_init_params params = {
        .mem_size = ctx_size,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    
    // Create input tensors
    struct ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, IMAGE_SIZE, batch_size);
    memcpy(x->data, batch_images, batch_size * IMAGE_SIZE * sizeof(float));
    
    // Create target tensor (one-hot encoded)
    struct ggml_tensor* targets = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NUM_CLASSES, batch_size);
    memset(targets->data, 0, NUM_CLASSES * batch_size * sizeof(float));
    
    for (int i = 0; i < batch_size; i++) {
        ((float*)targets->data)[batch_labels[i] * batch_size + i] = 1.0f;
    }
    
    // Forward pass
    struct ggml_tensor* logits = mnist_forward(model, ctx, x);
    struct ggml_tensor* loss = cross_entropy_loss(ctx, logits, targets);
    
    // Build forward graph
    struct ggml_cgraph gf = ggml_build_forward(loss);
    
    // Build backward graph
    struct ggml_cgraph gb = ggml_build_backward(ctx, &gf, false);
    
    // Compute forward and backward
    ggml_graph_compute_with_ctx(ctx, &gf, 4);
    ggml_graph_compute_with_ctx(ctx, &gb, 4);
    
    // Get loss value
    float loss_value = ggml_get_f32_1d(loss, 0);
    
    // Manual SGD update (simplified)
    // In practice, you'd use a proper optimizer
    struct ggml_tensor* weights[] = {model->fc1_w, model->fc1_b, model->fc2_w, model->fc2_b};
    
    for (int i = 0; i < 4; i++) {
        struct ggml_tensor* w = weights[i];
        struct ggml_tensor* g = w->grad;
        
        if (g != NULL) {
            int n = ggml_nelements(w);
            float* w_data = (float*)w->data;
            float* g_data = (float*)g->data;
            
            for (int j = 0; j < n; j++) {
                w_data[j] -= LEARNING_RATE * g_data[j];
            }
        }
    }
    
    ggml_free(ctx);
    return loss_value;
}

// Evaluation
static float evaluate(struct mnist_model* model, float* test_images, int* test_labels, int num_test) {
    int correct = 0;
    
    for (int i = 0; i < num_test; i += BATCH_SIZE) {
        int batch_size = (i + BATCH_SIZE <= num_test) ? BATCH_SIZE : (num_test - i);
        
        // Create computation context
        size_t ctx_size = 1024*1024;
        struct ggml_init_params params = {
            .mem_size = ctx_size,
            .mem_buffer = NULL,
            .no_alloc = false,
        };
        
        struct ggml_context* ctx = ggml_init(params);
        
        // Create input tensor
        struct ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, IMAGE_SIZE, batch_size);
        memcpy(x->data, &test_images[i * IMAGE_SIZE], batch_size * IMAGE_SIZE * sizeof(float));
        
        // Forward pass
        struct ggml_tensor* logits = mnist_forward(model, ctx, x);
        struct ggml_tensor* probs = ggml_soft_max(ctx, logits);
        
        // Build and compute graph
        struct ggml_cgraph gf = ggml_build_forward(probs);
        ggml_graph_compute_with_ctx(ctx, &gf, 4);
        
        // Check predictions
        for (int j = 0; j < batch_size; j++) {
            int pred_class = 0;
            float max_prob = -1.0f;
            
            for (int k = 0; k < NUM_CLASSES; k++) {
                float prob = ggml_get_f32_nd(probs, k, j, 0, 0);
                if (prob > max_prob) {
                    max_prob = prob;
                    pred_class = k;
                }
            }
            
            if (pred_class == test_labels[i + j]) {
                correct++;
            }
        }
        
        ggml_free(ctx);
    }
    
    return (float)correct / num_test;
}

int main() {
    srand(time(NULL));
    
    printf("=== Simple MNIST Classifier with GGML ===\n");
    
    // Generate synthetic data (in real usage, load from MNIST files)
    int num_train = 1000;  // Small dataset for demo
    int num_test = 200;
    
    float* train_images = malloc(num_train * IMAGE_SIZE * sizeof(float));
    int* train_labels = malloc(num_train * sizeof(int));
    float* test_images = malloc(num_test * IMAGE_SIZE * sizeof(float));
    int* test_labels = malloc(num_test * sizeof(int));
    
    generate_mnist_data(train_images, train_labels, num_train);
    generate_mnist_data(test_images, test_labels, num_test);
    
    printf("Generated %d training samples and %d test samples\n", num_train, num_test);
    
    // Initialize model
    struct mnist_model* model = mnist_model_init();
    
    // Training loop
    printf("\nStarting training...\n");
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        float epoch_loss = 0.0f;
        int num_batches = 0;
        
        for (int i = 0; i < num_train; i += BATCH_SIZE) {
            int batch_size = (i + BATCH_SIZE <= num_train) ? BATCH_SIZE : (num_train - i);
            
            float batch_loss = train_step(model, &train_images[i * IMAGE_SIZE], &train_labels[i], batch_size);
            epoch_loss += batch_loss;
            num_batches++;
            
            if (i % (BATCH_SIZE * 10) == 0) {
                printf("Epoch %d, Batch %d/%d, Loss: %.4f\n", 
                       epoch + 1, i / BATCH_SIZE + 1, (num_train + BATCH_SIZE - 1) / BATCH_SIZE, batch_loss);
            }
        }
        
        epoch_loss /= num_batches;
        printf("Epoch %d complete, Average Loss: %.4f\n", epoch + 1, epoch_loss);
        
        // Evaluate on test set
        float accuracy = evaluate(model, test_images, test_labels, num_test);
        printf("Test Accuracy: %.2f%%\n", accuracy * 100.0f);
    }
    
    // Final evaluation
    printf("\n=== Final Results ===\n");
    float final_accuracy = evaluate(model, test_images, test_labels, num_test);
    printf("Final Test Accuracy: %.2f%%\n", final_accuracy * 100.0f);
    
    // Cleanup
    ggml_free(model->ctx);
    free(model);
    free(train_images);
    free(train_labels);
    free(test_images);
    free(test_labels);
    
    printf("Training complete!\n");
    return 0;
} 