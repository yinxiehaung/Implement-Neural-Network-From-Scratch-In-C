# TinyNN-C

A zero-dependency neural network inference engine in pure C.
Split inference is a zero-cost architectural consequence, not a special feature.

## Overview

TinyNN-C is a lightweight neural network framework designed explicitly for extreme edge computing environments. Built entirely from scratch in C without relying on external mathematical or machine learning libraries, it provides a bare-metal solution for running AI models on resource-constrained hardware such as the Raspberry Pi Zero.

## Motivation

Deploying modern neural networks on highly constrained edge devices reveals the inherent overhead of mainstream deep learning frameworks. Standard solutions often introduce substantial dependency chains, memory bloat, and runtime overhead.

The primary objective of this project is to explore the absolute limits of edge hardware by implementing a minimal, highly optimized inference engine. By stripping away all dependencies, TinyNN-C guarantees predictable memory allocation (via custom Arena allocators) and predictable execution flow.

## Architecture & Extensibility

The core engine does not hardcode network communication. Instead, it is built around a strictly decoupled, polymorphic layer design. By generalizing every operation under the `nn_layer_t` interface, the engine acts as a pure computation graph manager.

This zero-cost abstraction natively enables advanced topologies like Split Neural Networks (SplitNN). Because the engine is agnostic to the internal mechanics of a "layer," developers can effortlessly inject custom operations. 

For example, network communication can be abstracted as custom layers:
- **NetEmitter Layer**: Acts as the final layer on the client (edge device), transmitting intermediate feature matrices over a TCP socket.
- **NetReceiver Layer**: Acts as the initial layer on the server, receiving feature matrices to resume the forward pass.

To the core inference engine, passing data across a network boundary is indistinguishable from standard matrix multiplication.

## Usage

The core library provides the primitives for building, training, and running neural networks.

### 1. Standard Inference (Core API)

```c
#include "nn.h"

int main() {
    arena_t arena;
    arena_init(&arena, MiB(10));

    nn_model_t model;
    nn_model_init(&model, &arena, 10);

    // Load and compile the full model
    nn_model_load(&model, "model.bin");
    nn_model_compile(&model, 1, 784);

    // Execute standard forward pass
    nn_forward_predict(&model, &input_image, &output_pred);
    
    return 0;
}
```
### 2. Architectural Extensibility: Distributed Inference
Because of the polymorphic API, you can implement custom layers to achieve split inference. As demonstrated in the project/splitnn_mnist example directory, you can inject TCP socket layers directly into the computation graph:

#### Client-Side (Edge Device):

```c
#include "nn.h"
#include "net_layer.h" // Custom layer implemented in the example project

int main() {
    // ... setup arena and model ...

    // Load feature extraction layers (e.g., Layer 0 to 1)
    nn_model_load_partial(&model, "model.bin", 0, 1);

    // Inject the custom network socket layer at the end of the graph
    nn_model_add_emitter(&model, server_sockfd);

    nn_model_compile(&model, 1, 784);

    // Engine executes Layer 0, Layer 1, and routes data through the Emitter
    nn_forward_predict(&model, &input_image, &dummy_out);

    return 0;
}
```

#### Server-Side (Compute Node)

```c
#include "nn.h"
#include "net_layer.h"

int main() {
    arena_t arena;
    arena_init(&arena, MiB(100));
    
    nn_model_t model;
    nn_model_init(&model, &arena, 10);

    // Abstract the network socket as the input layer
    nn_model_add_receiver(&model, client_sockfd);
    
    // Load classifier layers (e.g., Layer 2 to 3)
    nn_model_load_partial(&model, "model.bin", 2, 3);
    
    // Compile expecting the intermediate feature dimension (e.g., 128)
    nn_model_compile(&model, 1, 128);

    // Engine blocks until network features arrive, then completes inference
    nn_forward_predict(&model, &dummy_in, &final_prediction);
    
    return 0;
}
```
### 3. Build Instructions
To compile the core library and run the examples, ensure you have a standard C compiler (gcc) and make installed.

```bash
git clone [https://github.com/YourUsername/TinyNN-C.git](https://github.com/YourUsername/TinyNN-C.git)
cd TinyNN-C

# Compile the core MNIST training example
make

# Compile the distributed SplitNN example
cd project/splitnn_mnist
make
```

## Custom Layer Implementation
TinyNN-C is designed around a simple but powerful idea: everything is a layer. The engine does not distinguish between a matrix multiplication, an activation function, or a TCP socket transfer. All of them share the same interface defined in nn_layer_t. This means you can extend the engine with any custom behavior without modifying the core inference loop.
### The Layer Interface
To implement a custom layer, you provide function pointers for the behaviors you need:

- **forward**: The core computation. Takes an input matrix and produces an output matrix. This is the only required callback.
- **compile**: Called before inference begins. Tells the engine what output dimension and buffer size this layer needs. If omitted, the engine assumes the output dimension equals the input dimension.
- **backprop**: The gradient computation for training. Can be left as NULL for inference-only layers.
- **init_params**: Called during parameter initialization. Used for layers that have learnable weights, such as Dense layers. Leave as NULL for stateless layers.
- **serialize and deserialize**: Used when saving and loading models. If your layer has no learnable parameters, these can be left as NULL.

### The layer_type_t Registry
When you need to save and load a model that contains your custom layer, the engine needs to know how to reconstruct it from disk. This is handled through the layer_type_t enum and a corresponding deserializer registry.
You register your custom layer type by adding an entry to the enum and providing a factory function that the loader can call when it encounters your layer type during deserialization. This coupling is intentional and limited to the serialization boundary only. The core inference loop, meaning nn_forward_predict and nn_backprop, never reads the type_id field. It only calls forward and backprop through the function pointers.
If your custom layer has no parameters to save, you do not need to register it at all. Simply add it to the model at runtime and it will participate in inference transparently.
Adding a Custom Layer
The recommended pattern is to write a dedicated constructor function that allocates and initializes your layer struct, then appends it to the model. This is exactly how nn_model_add_emitter and nn_model_add_receiver are implemented in the splitnn_mnist example.
```c
nn_err_t nn_model_add_my_layer(nn_model_t *model, int my_param)
{
    my_layer_t *layer;
    arena_push(model->model_arena, sizeof(my_layer_t), true, (void **)&layer);

    layer->base.name       = "MyLayer";
    layer->base.type_id    = LAYER_TYPE_MY_LAYER;
    layer->base.forward    = my_layer_forward;
    layer->base.compile    = my_layer_compile;
    layer->base.backprop   = NULL;
    layer->base.init_params = NULL;
    layer->base.serialize  = NULL;

    layer->my_param = my_param;

    model->layers[model->layer_count++] = (nn_layer_t *)layer;
    return NN_OK;
}
```
The splitnn_mnist example demonstrates this pattern end to end. The NetEmitter and NetReceiver layers are custom layers that send and receive activation tensors over a TCP socket. From the perspective of nn_forward_predict, they are indistinguishable from any other layer in the pipeline.
