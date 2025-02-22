from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache
import mlx.core as mx

class Chat:
    def __init__(self, model_path="mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit", think='</think>', max_think=5000, max_kv_size=None):
        self.model, self.tokenizer = load(model_path)
        self.think = think
        self.max_think = max_think
        self.reset(max_kv_size=max_kv_size)
    def reset(self, max_kv_size=None, system=''):
        self.prompt_cache = make_prompt_cache(self.model, max_kv_size=max_kv_size)
        self.stop = None
        self.toks = None
        self.ongoing = None
    def get_ntok(self, s):
        return len(self.tokenizer.encode(s))
    def resume(self, max_new=500, verbose=False, stream=None):
        if self.stop == 'stop' or self.ongoing == None:
            return dict(text='', outputs=[''], benchmark='n/a', stop=self.stop)
        return self.generate(inputs='', toks=self.toks, max_new=max_new, verbose=verbose, stream=stream)
    def __call__(self, inputs, max_new=500, chat_fmt=True, verbose=False, stream=None):
        if isinstance(inputs, str):
            prompt = inputs
        else:
            prompt = inputs[0]
        if self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            toks = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
        self.ongoing = stream_generate(self.model, self.tokenizer, prompt=toks, max_tokens=self.max_think, prompt_cache=self.prompt_cache)
        return self.generate(inputs=prompt, toks=toks, max_new=max_new, verbose=verbose, stream=stream) 
    def generate(self, inputs, toks, max_new, verbose, stream):
        outputs = ''
        f = None
        if isinstance(stream, str):
            try:
                f = open(stream, 'a', encoding='utf-8')
            except:
                f = None
        for response in self.ongoing:
            frag = response.text
            if stream:
                if f:
                    f.write(frag)
                    f.flush()
                else:
                    print(frag, flush=True, end='')
            outputs += frag
            if self.think in outputs:
                max_new -= 1
            if max_new < 0:
                break
        if stream:
            if f:
                f.close()
            else:
                print()
        self.stop = response.finish_reason
        self.toks = mx.array([response.token])
        text = outputs.split(self.think)[-1].strip()
        benchmark = f"Prompt: {response.prompt_tokens} tokens, {response.prompt_tps:.3f} tokens-per-sec\nGeneration: {response.generation_tokens} tokens, {response.generation_tps:.3f} tokens-per-sec"
        if verbose:
            print(f'{inputs=}\n\n{text=}\n\n{benchmark=}')
        return dict(text=text, outputs=outputs, benchmark=benchmark, stop=self.stop)

if __name__ == '__main__':
    test()
