package inference

import (
	"fmt"
	"math"

	. "github.com/matthewrennie/go-llama.cpp"
)

type FnOutput func(lastToken Token)
type FnInput func(lastNTokens *[]Token, interject bool) *[]Token
type FnSwapContext func(nCtx int, nProcessed int, nKeep int, lastNTokens, embed *[]Token) (int, *[]Token)

type LlamaInference struct {
	llama       *Llama
	chInterject <-chan bool
}

func NewLlamaInference(llama *Llama) *LlamaInference {
	return &LlamaInference{
		llama: llama,
	}
}

func (li *LlamaInference) DefaultSwapContext(nCtx int, nProcessed int, nKeep int, lastNTokens, embed *[]Token) (int, *[]Token) {
	nleft := nProcessed - li.llama.Params.NKeep
	// insert n_left/2 tokens at the start of embd from last_n_tokens
	keep := (*lastNTokens)[nCtx-nleft/2-len(*embed) : len(*lastNTokens)-len(*embed)]
	kept := append(keep, *embed...)
	return nKeep, &kept
}

func (li *LlamaInference) InferInteractive(embedInput []Token, fnIn FnInput, fnOut FnOutput, fnSwapContext FnSwapContext, interject <-chan bool) error {
	return li.Infer(-1, embedInput, li.llama.ContextSize(), fnIn, fnOut, fnSwapContext, interject)
}

func (li *LlamaInference) InferPromptOnly(embedInput []Token, fnOut FnOutput, fnSwapContext FnSwapContext, interject <-chan bool) error {
	empty := []Token{}
	fIn := func(*[]Token, bool) *[]Token { return &empty }
	return li.Infer(li.llama.Params.NPredict, embedInput, li.llama.ContextSize(), fIn, fnOut, fnSwapContext, interject)
}

func (li *LlamaInference) Infer(nPredict int, embedInput []Token, nCtx int, fnIn FnInput, fnOut FnOutput, fnSwapContext FnSwapContext, chInterject <-chan bool) error {
	nProcessed := 0
	nRemain := nPredict
	lastNTokens := make([]Token, nCtx)
	embed := []Token{}

	if len(embedInput) > nCtx-4 {
		return fmt.Errorf("error: prompt is too long (%d tokens, max %d", len(embedInput), nCtx-4)
	}

	if li.llama.Params.NKeep < 0 || li.llama.Params.NKeep > len(embedInput) {
		li.llama.Params.NKeep = len(embedInput)
	}

	for nRemain != 0 {
		//evaluate
		if len(embed) > 0 {
			if nProcessed+len(embed) > nCtx {
				var e *[]Token
				nProcessed, e = li.DefaultSwapContext(nCtx, nProcessed, li.llama.Params.NKeep, &lastNTokens, &embed)
				embed = *e
			}

			if err := li.llama.Eval(embed, nProcessed, li.llama.Params.NThreads.Threads); err != nil {
				return err
			}
		}

		nProcessed += len(embed)
		embed = []Token{}

		//sample
		if len(embedInput) == 0 {
			id := li.llama.SampleTopPTopK(lastNTokens[nCtx-li.llama.Params.RepeatLastN:], li.llama.Params)

			lastNTokens = lastNTokens[1:]
			lastNTokens = append(lastNTokens, id)

			embed = []Token{id}
			if nRemain > 0 {
				nRemain = nRemain - 1
			}

			fnOut(id)
		}

		// end of text token
		if len(embed) > 0 && embed[len(embed)-1] == TokenEOS() {
			return nil
		}

		//populate
		if len(embedInput) > 0 {
			batchSize := int(math.Min(float64(li.llama.Params.NBatch), float64(len(embedInput))))
			embed = embedInput[:batchSize]
			lastNTokens = lastNTokens[batchSize:]
			lastNTokens = append(lastNTokens, embed...)
			embedInput = embedInput[batchSize:]
		}

		//provide the opportunity for input
		if len(embedInput) == 0 {
			in := fnIn(&lastNTokens, false)
			if in == nil {
				return nil
			}
			embedInput = li.appendInput(in, &embedInput)
		}

		//maybe interject
		select {
		case _, ok := <-chInterject:
			if ok {
				in := fnIn(&lastNTokens, true)
				if in == nil {
					return nil
				}
				embedInput = li.appendInput(in, &embedInput)
			}
		default:
		}
	}

	return nil
}

func (li *LlamaInference) appendInput(in, embedInput *[]Token) []Token {
	if in == nil || embedInput == nil {
		return []Token{}
	}

	if len(*in) > 0 {
		return append(*embedInput, *in...)
	}

	return *embedInput
}
