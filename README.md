# Inferno

Inferno is a C++ deep learning and tokenizer project focused on building a GPT-style training stack from the ground up.

The repo is organized around several related components:

- **Inferno** — tensor, autograd, neural network modules, optimizers, CUDA/CPU ops, and training utilities.
- **Pretokenizer** — low-level byte/string pre-tokenization utilities shared by tokenizer tools.
- **BPETrainer** — trains byte-pair encoding merge rules and vocabulary files from a text corpus.
- **InfernoTokenizer** — loads trained BPE merges/vocab and encodes/decodes text.
- **DatasetTokenizer** — converts large text datasets into binary token files for model training.

The goal is to support an end-to-end workflow:

```text
raw text corpus
    -> Pretokenizer
    -> BPETrainer
    -> merges/vocab files
    -> InfernoTokenizer
    -> DatasetTokenizer
    -> binary token dataset
    -> Inferno DataLoader
    -> GPT-style model training
```

---

## Project Structure

A typical layout looks like:

```text
Inferno/
├── BPETrainer/
│   └── trains BPE merges and vocabulary
├── DatasetTokenizer/
│   └── encodes raw datasets into binary token files
├── Inferno/
│   └── core tensor/autograd/neural-network library
├── InfernoApp/
│   └── example/test application for training and experiments
├── InfernoTokenizer/
│   └── BPE tokenizer library
├── Pretokenizer/
│   └── shared pre-tokenization logic
├── include/
│   └── public headers
├── src/
│   └── implementation files
├── bin/
│   └── built executables
└── lib/
    └── built static libraries
```

Exact folder names may differ depending on the Visual Studio solution layout.

---

# Components

## Inferno Library

The Inferno library contains the core machine learning engine.

Major areas include:

```text
Tensor
TensorImpl
Storage
CPUStorage
CUDAStorage
Device
DType
Autograd Engine
Modules
Optimizers
Loss functions
CPU kernels
CUDA kernels
```

### Tensor

`Tensor` is the public-facing tensor object.

It represents:

- dtype
- shape
- strides
- device
- storage
- autograd metadata
- gradient
- view metadata

Example:

```cpp
Inferno::Tensor x(
    Inferno::DType::Float32,
    {8, 1024, 768},
    "x",
    Inferno::Device::cuda()
);
```

### TensorImpl

`TensorImpl` is the internal implementation object behind `Tensor`.

The public `Tensor` class owns:

```cpp
std::shared_ptr<TensorImpl> m_impl;
```

Most low-level implementation details are hidden behind this object.

### Storage

Storage classes own raw memory.

Examples:

```text
CPUStorage
CUDAStorage
```

The tensor object uses shape, strides, and offset metadata to interpret the underlying storage.

### Device

Inferno supports CPU and CUDA devices.

Example:

```cpp
Inferno::Device cpu = Inferno::Device::cpu();
Inferno::Device gpu = Inferno::Device::cuda(0);
```

### Autograd

Inferno includes an autograd engine for backpropagation.

Typical flow:

```cpp
Inferno::Tensor logits = model.forward(x);
Inferno::Tensor loss = loss_fn.forward(logits, y);

loss.backward();

optimizer.step();
optimizer.zero_grad();
```

Autograd uses graph nodes to represent backward functions for operations such as:

- add
- multiply
- matmul
- select
- sigmoid
- GELU
- layer norm
- MSE loss
- cross entropy loss

### Modules

Inferno supports PyTorch-like module organization.

Examples:

```text
Module
Linear
Embedding
LayerNorm
GELU
Softmax
TransformerBlock
GPT model
```

Modules expose their parameters through a `parameters()` method so optimizers can update them.

Example:

```cpp
std::vector<Inferno::Tensor*> params = model.parameters();
Inferno::AdamW optimizer(params);
```

### Optimizers

Inferno supports optimizers such as:

```text
SGD
AdamW
```

AdamW tracks per-parameter optimizer state:

```text
m = first moment estimate
v = second moment estimate
```

This requires extra memory, but usually trains transformer models much better than plain SGD.

### CUDA Support

Inferno includes CUDA paths for selected tensor operations.

CUDA kernels are used for operations such as:

- elementwise ops
- matmul
- reductions
- optimizer updates
- embedding operations
- attention-related kernels

The intended design is:

```text
public Tensor API
    -> internal TensorImpl/Storage access
    -> CPU/CUDA dispatch
    -> kernel implementation
```

Public headers should expose what users can do. Internal headers and source files should contain how Inferno performs the operation.

---

## Pretokenizer

The Pretokenizer library handles the first pass over raw text before BPE training or tokenization.

Its job is to split raw byte/text input into manageable pieces.

For a simple early tokenizer, this may be as basic as splitting on spaces and preserving byte sequences.

The pretokenizer may be used by both:

```text
BPETrainer
InfernoTokenizer
```

This keeps training-time and inference-time tokenization behavior consistent.

### Responsibilities

The pretokenizer may:

- read text as binary data
- preserve UTF-8 byte sequences
- split pieces consistently
- emit string pieces
- count piece frequencies
- avoid loading huge files all at once

### Why it is separate

Pretokenization must be shared between the trainer and tokenizer.

If BPE training uses one splitting rule and inference uses another, the tokenizer will not reproduce the same tokenization behavior.

---

## BPETrainer

BPETrainer trains byte-pair encoding merge rules and a vocabulary from a corpus.

The typical process is:

```text
1. Read raw text corpus
2. Pretokenize text into pieces
3. Count piece frequencies
4. Represent each piece as byte/token IDs
5. Count adjacent token pairs
6. Repeatedly merge the most frequent pair
7. Write merge rules
8. Write vocabulary
```

### Base Vocabulary

The base vocabulary usually starts with byte values:

```text
0-255
```

Each merge creates a new token ID:

```text
256, 257, 258, ...
```

Example merge:

```text
[104, 101] -> 256
```

This means the pair:

```text
'h', 'e'
```

is replaced by token `256`.

### Merge Rules

Merge rules must be written in creation order.

During encoding, earlier merges have higher priority because they define tokens that later merges may depend on.

Example merge file format:

```text
104 101 256
256 108 257
257 108 258
```

or a more readable format:

```text
[104, 101] -> 256
[256, 108] -> 257
[257, 108] -> 258
```

Choose one format and keep the trainer and tokenizer parser consistent.

### Vocabulary

The vocabulary maps token IDs to byte strings or reconstructed pieces.

For byte-level BPE:

```text
0-255: raw byte tokens
256+: merged byte sequences
```

The vocab is required for decoding tokens back into text.

### Expected Vocabulary Size

For small corpora such as Shakespeare, the trainer may stop before reaching a large target vocab size.

For example, if only about 21,000 useful tokens can be created, the trainer may stop because there are no more repeated pairs to merge.

That is normal for small datasets.

---

## InfernoTokenizer

InfernoTokenizer loads the merge rules and vocabulary produced by BPETrainer.

It provides:

```text
encode(text) -> vector<int32_t or uint32_t>
decode(tokens) -> string
```

### Encoding

Encoding usually follows this process:

```text
1. Pretokenize input text
2. Convert each piece into base byte tokens
3. Apply BPE merges in trained order/rank
4. Return token IDs
```

Example:

```cpp
InfernoTokenizer::BPETokenizer tokenizer;

InfernoTokenizer::TokenizerConfig config;
config.merges_file = "merges.txt";
config.vocab_file = "vocab.txt";

tokenizer.Initialize(config);

std::vector<uint32_t> tokens = tokenizer.encode("Where are we going?");
```

### Decoding

Decoding reverses token IDs back into bytes/text.

Example:

```cpp
std::string text = tokenizer.decode(tokens);
```

### Important Notes

The tokenizer must use the same pretokenization behavior that was used during BPE training.

If the pretokenizer differs between training and encoding, the merge rules may not apply correctly.

---

## DatasetTokenizer

DatasetTokenizer converts a raw text dataset into a binary token file.

This is used before training so the model can read token IDs directly instead of repeatedly tokenizing text during every training step.

### Input

```text
raw text file
```

Example:

```text
shakespeare.txt
openwebtext.txt
```

### Output

```text
binary token file
```

Example:

```text
train_tokens.bin
```

Each token can be written as a fixed-width integer.

For Inferno training, `int32_t` is usually enough:

```cpp
int32_t token_id;
```

A vocabulary of 21,000 or 60,000 tokens easily fits in signed 32-bit integers.

### Binary Format

The binary format is simple:

```text
int32 token
int32 token
int32 token
...
```

When written on x86/Windows, this will be little-endian.

For example, bytes:

```text
28 3 0 0
```

represent integer:

```text
796
```

### Example Encoding Loop

```cpp
std::ifstream in(input_file, std::ios::binary);
std::ofstream out(output_file, std::ios::binary);

const size_t CHUNK_SIZE = 8 * 1024 * 1024;
std::vector<char> buffer(CHUNK_SIZE);

while (in.read(buffer.data(), CHUNK_SIZE) || in.gcount() > 0) {
    size_t n = static_cast<size_t>(in.gcount());

    std::string chunk(buffer.data(), n);

    std::vector<uint32_t> tokens = tokenizer.encode(chunk);

    out.write(
        reinterpret_cast<const char*>(tokens.data()),
        tokens.size() * sizeof(uint32_t)
    );
}
```

If Inferno uses `DType::Int32`, the tokenizer output can be converted or stored as `int32_t`.

---

# Data Loading for Training

For GPT-style training, the training data is a long stream of token IDs.

The model input and target are shifted by one token.

```text
tokens: [t0, t1, t2, t3, t4, ...]

x:      [t0, t1, t2, t3]
y:      [t1, t2, t3, t4]
```

For a batch:

```text
x shape = [batch_size, context_size]
y shape = [batch_size, context_size]
```

Example:

```cpp
std::pair<Inferno::Tensor, Inferno::Tensor> batch = loader.next_batch();

Inferno::Tensor x = batch.first;
Inferno::Tensor y = batch.second;
```

### Streaming DataLoader

For large token files, do not load the entire dataset into memory.

Instead, the DataLoader should read chunks from disk and sample windows from the current chunk.

A simple strategy:

```text
1. Load a chunk of tokens
2. Randomly sample windows from that chunk
3. After N batches, load a new chunk
4. Return x/y tensors from next_batch()
```

Example public interface:

```cpp
class DataLoader {
public:
    DataLoader(
        const std::string& filename,
        size_t batch_size,
        size_t context_size
    );

    std::pair<Inferno::Tensor, Inferno::Tensor> next_batch();

private:
    std::ifstream m_file;
    std::vector<int32_t> m_buffer;

    size_t m_batch_size;
    size_t m_context_size;
    size_t m_tokens_in_buffer;
};
```

### Batch Creation

`next_batch()` should create the tensors internally:

```cpp
Inferno::Tensor x(
    Inferno::DType::Int32,
    {m_batch_size, m_context_size},
    "x_batch",
    Inferno::Device::cpu()
);

Inferno::Tensor y(
    Inferno::DType::Int32,
    {m_batch_size, m_context_size},
    "y_batch",
    Inferno::Device::cpu()
);
```

Then fill:

```cpp
x[b, t] = buffer[start + t];
y[b, t] = buffer[start + t + 1];
```

---

# Training Workflow

A typical model training workflow:

```text
1. Train tokenizer
2. Encode dataset
3. Create model
4. Create DataLoader
5. Create optimizer
6. Train
7. Save checkpoints
```

Example:

```cpp
DataLoader loader("train_tokens.bin", 8, 1024);

GPT model(config);

std::vector<Inferno::Tensor*> params = model.parameters();

Inferno::AdamW optimizer(
    params,
    3e-4f,
    0.9f,
    0.95f,
    1e-8f,
    0.1f
);

for (size_t step = 0; step < total_steps; ++step) {
    std::pair<Inferno::Tensor, Inferno::Tensor> batch = loader.next_batch();

    Inferno::Tensor x = batch.first;
    Inferno::Tensor y = batch.second;

    Inferno::Tensor logits = model.forward(x);

    Inferno::Tensor loss = loss_fn.forward(logits, y);

    loss.backward();

    optimizer.step();
    optimizer.zero_grad();
}
```

---

# Loss Expectations

For cross entropy, the random starting loss is approximately:

```text
ln(vocab_size)
```

Examples:

```text
vocab_size = 21,000  -> random loss ≈ 9.95
vocab_size = 60,000  -> random loss ≈ 11.00
```

For a small Shakespeare-style dataset with a BPE vocab around 21,000:

```text
starting loss:      ~9.9-10.0
learning signal:    7.0
decent:             4-5
good:               3-4
very overfit:       2-3
```

Validation loss should be used to decide when to stop training.

A practical rule:

```text
stop when validation loss has not improved for a fixed number of evaluation intervals
```

---

# Build Notes

This project is currently designed around:

```text
C++20
Visual Studio 2022
CUDA
Windows
```

Recommended project setup:

```text
Inferno              static library
Pretokenizer         static library
InfernoTokenizer     static library
BPETrainer           executable
DatasetTokenizer     executable
InfernoApp           executable
InfernoTests         executable
```

### Common Dependencies

For CUDA builds, link against:

```text
cudart.lib
cublas.lib
cublasLt.lib
```

Typical CUDA library path:

```text
$(CUDA_PATH)\lib\x64
```

Typical CUDA include path:

```text
$(CUDA_PATH)\include
```

For RTX 4090, use an appropriate architecture such as:

```text
compute_89,sm_89
```

---

# Development Notes

## Public vs Internal API

Keep public headers clean.

Public API:

```text
Tensor
Module
Linear
Embedding
LayerNorm
DataLoader
Optimizer
Tokenizer
```

Internal API:

```text
TensorImpl
Storage internals
GetImpl()
raw data pointers
CUDA launch helpers
private kernel functions
```

A useful rule:

```text
Public headers explain what users can do.
Private headers/source files explain how Inferno does it.
```

## Template Functions

C++ template function definitions must generally be visible in headers.

If a template is declared in a header but defined only in a `.cpp`, linker errors may occur unless explicit instantiations are provided.

For internal template helpers, prefer:

```text
private implementation header
```

Example:

```text
tensor_internal.h
optimizer_kernels.h
cuda_ops_templates.h
```

## Optimizer Updates

Optimizer updates should run under:

```cpp
NoGradGuard guard;
```

Optimizer math should not be recorded into the autograd graph.

For CUDA training, optimizers should use direct CPU/CUDA dispatch and update parameters in-place.

---

# Current Status

The project currently supports or is expected to support:

- byte/BPE tokenizer training
- dataset tokenization to binary token files
- GPT-style token stream loading
- Tensor object with CPU/CUDA storage
- autograd graph execution
- neural network modules
- transformer model components
- SGD and AdamW optimizers
- CPU and CUDA operation dispatch

Some APIs are still evolving.

---

# Roadmap

Possible next steps:

```text
1. Stabilize tokenizer file formats
2. Add validation split support to DatasetTokenizer/DataLoader
3. Add checkpoint save/load
4. Add AdamW CUDA optimizer kernel
5. Add learning-rate scheduler
6. Add gradient clipping
7. Add fused softmax / scaled dot-product attention
8. Add checkpoint metadata
9. Add deterministic state_dict
10. Add tests for tokenizer encode/decode round trips
```

---

# Recommended Checks

Before training, verify:

```text
logits shape = [batch_size, context_size, vocab_size]
targets shape = [batch_size, context_size]
targets dtype = Int32
max target token < vocab_size
CrossEntropyLoss does not receive pre-softmaxed probabilities
x/y are shifted by exactly one token
```

For example:

```text
x = tokens[i : i + context_size]
y = tokens[i + 1 : i + context_size + 1]
```

---

# License

Add license information here.
