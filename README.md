go-llama.cpp
==========

## Description

Golang bindings for [LLaMa.cpp](https://github.com/ggerganov/llama.cpp) and a partial port of the main example.

## Usage

Clone and follow the usage instructions for LLaMa.cpp, then:

```bash
# build the llama.cpp dynamic library
cd <path_to_llama_folder>
make libllama.so

# clone this repo
git clone https://github.com/matthewrennie/go-llama.cpp
cd <path_go-llama.cpp_folder>

# copy required files
cp <path_to_llama_folder>/libllama.so ./
cp <path_to_llama_folder>/llama.h ./

# build
go build -o main ./examples/main/main.go

# run the inference
./main -m <path_to_llama_folder>/models/7B/ggml-model-q4_0.bin -n 128

# run the chat example
./examples/chat.sh

# options
./main --help
```