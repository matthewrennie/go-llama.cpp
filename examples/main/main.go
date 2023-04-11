package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/alexflint/go-arg"
	"github.com/fatih/color"
	. "github.com/matthewrennie/go-llama.cpp"
	"github.com/matthewrennie/go-llama.cpp/inference"
)

var (
	llama       *Llama
	params      ExampleParams
	chInterject chan bool = make(chan bool, 1)
)

type ExampleParams struct {
	LLamaParams
	Color          bool     `arg:"--color" help:"colorise output to distinguish prompt and user input from generations"`
	Interactive    bool     `arg:"-i, --interactive" help:"run in interactive mode"`
	ReversePrompts []string `arg:"-r, --reverse-prompt,separate" help:"run in interactive mode and poll user input upon seeing PROMPT (can be
                        specified more than once for multiple prompts)."`
	VerbosePrompt bool `arg:"--verbose-prompt" help:"print prompt before generation"`
}

func printPrompt(prompt string, tokens []Token) {
	fmt.Fprintf(os.Stderr, "\nprompt: '%s'\n", prompt)
	fmt.Fprintf(os.Stderr, "number of tokens in prompt = %d\n", len(tokens))

	for _, t := range tokens {
		fmt.Fprintf(os.Stderr, "%6d -> '%s'\n", t, llama.TokenToString(t))
	}

	fmt.Fprintf(os.Stderr, "\n\n")
}

func printPromptBasedOnKeep(prompt string, tokens []Token, keep int) {
	fmt.Fprintf(os.Stderr, "static prompt based on n_keep:")
	for i := 0; i < keep; i++ {
		fmt.Fprintf(os.Stderr, "%s", llama.TokenToString(tokens[i]))
	}
	fmt.Fprintf(os.Stderr, "\n\n")
}

func printConfig(nCtx int) {
	fmt.Fprintf(os.Stderr, "sampling: temp = %f, top_k = %d, top_p = %f, repeat_last_n = %d, repeat_penalty = %f\n",
		params.Temp, params.TopK, params.TopP, params.RepeatLastN, params.RepeatPenalty)

	fmt.Fprintf(os.Stderr, "generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n",
		nCtx, params.NBatch, params.NPredict, params.NKeep)

	fmt.Fprintf(os.Stderr, "\n\n")
}

func tokensEndWithUserPrompt(lastNTokens *[]Token) bool {
	if lastNTokens == nil {
		return false
	}

	lastOutput := ""
	for _, t := range *lastNTokens {
		lastOutput += llama.TokenToString(t)
	}

	for _, r := range params.ReversePrompts {
		if strings.HasSuffix(lastOutput, r) {
			return true
		}
	}

	return false
}

func scanIn(msg chan string, quit chan bool) {
	reader := bufio.NewReader(os.Stdin)
	for {
		var s string
		s, _ = reader.ReadString('\n')
		msg <- s

		mayBeQuit := <-quit
		if mayBeQuit {
			return
		}
	}
}

func userPrompt(lastNTokens *[]Token, interject bool) *[]Token {
	if lastNTokens == nil {
		return nil
	}

	if !interject && !tokensEndWithUserPrompt(lastNTokens) {
		empty := []Token{}
		return &empty
	}

	text := ""
	if interject && len(params.ReversePrompts) > 0 {
		text = params.ReversePrompts[0]
		fmt.Printf("\n%s", text)
	}

	if params.Color {
		defer color.Unset()
		color.Set(color.FgGreen, color.Bold)
	}

	signal.Reset(syscall.SIGINT, syscall.SIGTERM)
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)
	defer catchInferenceInterject()

	quit := make(chan bool, 1)

	msg := make(chan string, 1)
	go scanIn(msg, quit)

loop:
	for {
		select {
		case <-sigs:
			return nil
		case s := <-msg:
			switch {
			case strings.HasSuffix(s, "\\\n"):
				lastIndex := strings.LastIndex(s, "\\")
				text += s[:lastIndex] + s[lastIndex+1:]
				quit <- false
			case strings.HasSuffix(s, "\n"):
				text += s
				quit <- true
				break loop
			default:
				text += s
			}
		}
	}

	tokens, err := llama.Tokenize(text, false)
	if err != nil {
		panic(err)
	}

	return &tokens
}

func display(lastToken Token) {
	fmt.Printf("%s", llama.TokenToString(lastToken))
}

func catchInferenceInterject() {
	signal.Reset(syscall.SIGINT, syscall.SIGTERM)
	c := make(chan os.Signal)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		select {
		case <-c:
			chInterject <- true
			signal.Reset(syscall.SIGINT, syscall.SIGTERM)
			return
		}
	}()
}

func main() {
	arg.MustParse(&params)

	var err error
	llama, err = Init(params.LLamaParams)
	if err != nil {
		panic(err)
	}

	fmt.Fprintf(os.Stderr, "\nsystem_info: n_threads = %d | %s\n", params.NThreads.Threads, llama.SystemInfo())

	if params.Interactive {
		fmt.Fprintf(os.Stderr, "\n== Running in interactive mode. ==\n")
		fmt.Fprintf(os.Stderr, " - Press Ctrl+C to interject at any time.\n")
		fmt.Fprintf(os.Stderr, " - Press Return to return control to LLaMa.\n")
		fmt.Fprintf(os.Stderr, "If you want to submit another line, end your input in '\\'.\n\n")
	}

	// Add a space in front of the first character to match OG llama tokenizer behavior
	prompt := fmt.Sprintf(" %s", params.Prompt)

	embedInput, err := llama.Tokenize(prompt, true)
	if err != nil {
		panic(err)
	}

	if params.VerbosePrompt {
		printPrompt(prompt, embedInput)

		params.NKeep = int(math.Min(float64(params.NKeep), float64(len(embedInput))))
		if params.NKeep > 0 {
			printPromptBasedOnKeep(prompt, embedInput, params.NKeep)
		}
	}

	nCtx := llama.ContextSize()
	printConfig(nCtx)

	if params.Color {
		defer color.Unset()
		color.Set(color.FgYellow)
	}
	fmt.Printf(params.Prompt)

	llamaInf := inference.NewLlamaInference(llama)

	switch {
	case params.Interactive:
		if err := llamaInf.InferInteractive(embedInput, userPrompt, display, llamaInf.DefaultSwapContext, chInterject); err != nil {
			panic(err)
		}
	default:
		if err := llamaInf.InferPromptOnly(embedInput, display, llamaInf.DefaultSwapContext, chInterject); err != nil {
			panic(err)
		}
	}

	color.Unset()
	llama.PrintTimings()
	llama.Free()
}
