# Ojo

My custom library for transformers research written in jax. Right now, is a work in progress, but my goal is to use this library/repo to keep all my model implementations and tooling in want place.

## What tools?

Other than the implementation of the models, you might be wondering what kind of tools do I plan to implement in here:

1. Hooks and activation patching (AST Patching): Turns out this is a pain to implement in jax. The simplest way to do it is to add callbacks to cache (or inject) activations in different place of the model's code. However, this means that I would have to rewrite the code every time I want to run a different experiment. The solution: AST patching. [Learn more here](https://www.alignmentforum.org/posts/C5KAZQib3bzzpeyrg/progress-update-1-from-the-gdm-mech-interp-team-full-update#Instrumenting_LLM_model_internals_in_JAX)
2. Quick architectural modifications: Pretty similar to the above. I want a fast way to switch different implementations of different parts of a given model. My current plan is to also use AST Patching.
3. Weight Conversion tools: Torch -> Orbax.
4. Bunch of pallas kernels... I really want to experiment with mgpu and tpus
5. Quantization?
6. Tools for model parallelism?


## Current Roadmap 

* Implement the building blocks (Attention, mlps, norms...)
    - KVCache initial
* Implement and run Llama 3.2
    - Weight conversion
* Implement NanoGPT
* Block tests
    - Start building tests for future implementations
    - Symbolic shape checking
    - KVCache profiling and improvements
* Layer hooks (Hooks only for the output of layers)
    - AST patcher
* Hooks test
    - Correctness
    - Performance
* Layer patching and interventions
* Flash Attention
    - Pallas triton backend
* Paged Attention
    - Pallas triton backend

