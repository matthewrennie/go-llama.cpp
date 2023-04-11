package llama

/*
#cgo LDFLAGS: libllama.so
#include "llama.h"
#include <stdlib.h>
*/
import "C"

import (
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"unsafe"
)

type Token int32

type LLamaParams struct {
	Model         string     `arg:"required,-m,--model" help:"path to llama model"`
	Prompt        string     `arg:"-p,--prompt" help:"input prompt"`
	RepeatLastN   int        `arg:"--repeat_last_n" help:"last n tokens to consider for penalize" default:"64"`
	TopK          int        `arg:"--top_k" help:"top-k sampling" default:"40"`
	TopP          float32    `arg:"--top_p" help:"top-p sampling" default:"0.95"`
	Temp          float32    `arg:"--temp" help:"temperature" default:"0.80"`
	RepeatPenalty float32    `arg:"--repeat_penalty" help:"penalize repeat sequence of tokens" default:"1.10"`
	NPredict      int        `arg:"-n,--n_predict" help:"number of tokens to predict" default:"128"`
	NKeep         int        `arg:"--keep" help:"number of tokens to keep from the initial prompt" default:"0"`
	NBatch        int        `arg:"-b,--batch_size" help:"batch size for prompt processing" default:"8"`
	NThreads      ArgThreads `arg:"-t,--threads" help:"number of threads to use during computation, -1 to determine based on hardware concurrency" default:"-1"`
	NParts        int        `arg:"--n_parts" help:"number of model parts" default:"-1"`
	UseMLock      bool       `arg:"--mlock" help:"force system to keep model in RAM rather than swapping or compressing" default:"false"`
	MemoryF32     bool       `arg:"--memory_f32" help:"use f32 instead of f16 for memory key+value:"false"`
	Seed          int        `arg:"-s,--seed" help:"RNG seed" default:"-1"`
	ContextSize   int        `arg:"-c,--ctx_size" help:"size of the prompt context" default:"512"`
}

type ArgThreads struct {
	Threads int
}

func (a *ArgThreads) UnmarshalText(b []byte) error {
	s := string(b)
	if s == "-1" {
		a.Threads = int(math.Min(float64(4), float64(runtime.NumCPU())))
		return nil
	}

	t, err := strconv.Atoi(s)
	if err != nil {
		return err
	}
	a.Threads = t
	return nil
}

type Llama struct {
	Ctx    *C.struct_llama_context
	Params LLamaParams
}

func Init(params LLamaParams) (*Llama, error) {
	lparams := C.llama_context_default_params()
	lparams.n_ctx = C.int(params.ContextSize)
	lparams.n_parts = C.int(params.NParts)
	lparams.seed = C.int(params.Seed)
	lparams.f16_kv = C.bool(!params.MemoryF32)
	lparams.use_mlock = C.bool(params.UseMLock)

	if params.ContextSize > 2048 {
		fmt.Fprintf(os.Stderr, "warning model does not support context sizes greater then 2048 tokens (%d specific)", params.ContextSize)
	}

	ctx, err := initFromFile(params.Model, lparams)
	if err != nil {
		return nil, err
	}

	return &Llama{Ctx: ctx, Params: params}, nil
}

func initFromFile(path string, lparams C.struct_llama_context_params) (*C.struct_llama_context, error) {
	ctx := C.llama_init_from_file(C.CString(path), lparams)
	if ctx == nil {
		return nil, errors.New(fmt.Sprintf("unable to load model: %s", path))
	}
	return ctx, nil
}

func (l *Llama) SystemInfo() string {
	return C.GoString(C.llama_print_system_info())
}

func (l *Llama) Tokenize(text string, addBos bool) ([]Token, error) {
	intAddBos := 0
	if addBos {
		intAddBos = 1
	}

	arr := make([]Token, len(text)+intAddBos)
	arr_ptr := (*C.int)(unsafe.Pointer(&arr[0]))
	t := C.llama_tokenize(l.Ctx, C.CString(text), arr_ptr, C.int(len(arr)), C.bool(addBos))
	if t < 0 {
		return []Token{}, errors.New("invalid size")
	}

	tokens := make([]Token, t)
	for i := 0; i < int(t); i++ {
		tokens[i] = Token(*(*C.int)(unsafe.Pointer(uintptr(unsafe.Pointer(arr_ptr)) + uintptr(i)*unsafe.Sizeof(*arr_ptr))))
	}

	return tokens, nil
}

func (l *Llama) ContextSize() int {
	return int(C.llama_n_ctx(l.Ctx))
}

func (l *Llama) TokenToString(t Token) string {
	cstr := C.llama_token_to_str(l.Ctx, C.int(t))
	return C.GoString(cstr)
}

func (l *Llama) Eval(tokens []Token, n_past, n_threads int) error {
	arr_ptr := (*C.int)(unsafe.Pointer(&tokens[0]))

	res := C.llama_eval(l.Ctx, arr_ptr, C.int(len(tokens)), C.int(n_past), C.int(n_threads))
	if res != 0 {
		return errors.New("evaluation failed")
	}

	return nil
}

func (l *Llama) SampleTopPTopK(tokens []Token, params LLamaParams) Token {
	arr_ptr := (*C.int)(unsafe.Pointer(&tokens[0]))

	id := Token(C.llama_sample_top_p_top_k(l.Ctx,
		arr_ptr,
		C.int(params.RepeatLastN),
		C.int(params.TopK),
		C.float(params.TopP),
		C.float(params.Temp),
		C.float(params.RepeatPenalty)))

	return id
}

func TokenBOS() Token {
	return Token(C.llama_token_bos())
}

func TokenEOS() Token {
	return Token(C.llama_token_eos())
}

func (l *Llama) PrintTimings() {
	C.llama_print_timings(l.Ctx)
}

func (l *Llama) ResetTimings() {
	C.llama_reset_timings(l.Ctx)
}

func (l *Llama) Free() {
	C.llama_free(l.Ctx)
}
