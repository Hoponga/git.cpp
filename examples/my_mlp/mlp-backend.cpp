#include "ggml/ggml.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define IN  4          // input dimension
#define HID 8          // hidden dimension
#define OUT 3          // output dimension
#define LEARNING_RATE 0.05f

// —— helper to fill tensor with random N(0,1) ——
static void rnd_normal(float *p, int n) {
    for (int i = 0; i < n; ++i) {
        float u = (rand()+1.0f)/(RAND_MAX+1.0f);
        float v = (rand()+1.0f)/(RAND_MAX+1.0f);
        p[i] = sqrtf(-2*logf(u))*cosf(2*M_PI*v);
    }
}

int main() {
    srand(time(NULL));

    // 1) create context big enough for weights + graph
    const size_t ctx_size = 16*1024*1024;        // 16 MB is plenty here
    struct ggml_init_params params = { ctx_size, NULL, /*no_alloc=*/false };
    struct ggml_context *ctx = ggml_init(params);

    // 2) define learnable parameters
    struct ggml_tensor *W1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, IN, HID);
    struct ggml_tensor *b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HID);
    struct ggml_tensor *W2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HID, OUT);
    struct ggml_tensor *b2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, OUT);

    rnd_normal(W1->data, IN*HID);
    rnd_normal(b1->data, HID);
    rnd_normal(W2->data, HID*OUT);
    rnd_normal(b2->data, OUT);

    // 3) single input & target
    float x_data[IN]  = { 0.5, -1.2, 0.3, 2.0 };
    float t_data[OUT] = { 0.0, 1.0, 0.0 };   // one-hot target

    struct ggml_tensor *x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, IN);
    memcpy(x->data, x_data, sizeof(x_data));

    struct ggml_tensor *target = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, OUT);
    memcpy(target->data, t_data, sizeof(t_data));

    // 4) forward graph
    struct ggml_tensor *h   = ggml_add(ctx,
                               ggml_mul_mat(ctx, x,      W1),   // x·W1  (1×IN   · IN×HID -> 1×HID)
                               b1);                              // broadcasting add
    h = ggml_relu(ctx, h);                                       // ReLU
    struct ggml_tensor *y   = ggml_add(ctx,
                               ggml_mul_mat(ctx, h,      W2),   // h·W2  (1×HID  · HID×OUT)
                               b2);

    y = ggml_soft_max(ctx, y);

    // 5) loss = ½‖y - target‖²
    struct ggml_tensor *diff = ggml_sub(ctx, y, target);
    struct ggml_tensor *loss = ggml_mean(ctx,
                                ggml_sqr(ctx, diff));            // MSE (scalar)

    // 6) finalize graph
    struct ggml_cgraph gf = ggml_build_forward(loss);

    // 7) backward graph
    struct ggml_cgraph gb = ggml_build_backward(ctx, &gf, false);

    // 8) run forward+backward
    ggml_graph_compute_with_ctx(ctx, &gf, /*threads=*/4);
    ggml_graph_compute_with_ctx(ctx, &gb, /*threads=*/4);

    printf("loss: %f\n", ggml_get_f32_1d(loss, 0));

    // 9) SGD update  (each grad lives in param->grad)
    struct ggml_tensor *weights[] = { W1, b1, W2, b2 };
    for (int i = 0; i < 4; ++i) {
        struct ggml_tensor *w = weights[i];
        struct ggml_tensor *g = w->grad;
        int n = ggml_nelements(w);
        float *wd = w->data, *gd = g->data;
        for (int j = 0; j < n; ++j)
            wd[j] -= LEARNING_RATE * gd[j];
    }

    ggml_free(ctx);
}