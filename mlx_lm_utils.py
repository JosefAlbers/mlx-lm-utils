from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache

class Chat:
    def __init__(self, model_path="mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit"):
        self.model, self.tokenizer = load(model_path)
        self.think = '</think>'
        self.reset()
    def reset(self, system=''):
        self.prompt_cache = make_prompt_cache(self.model)
    def get_ntok(self, s):
        return len(self.tokenizer.encode(s))
    def __call__(self, inputs, max_new, chat_fmt=True, verbose=False, stream=None):
        if isinstance(inputs, str):
            prompt = inputs
        else:
            prompt = inputs[0]
        if isinstance(stream, str):
            try:
                f = open(stream, 'a', encoding='utf-8')
            except:
                f = None
        if self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
        outputs = ''
        for response in stream_generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_new, prompt_cache=self.prompt_cache):
            frag = response.text
            outputs += frag
            if stream:
                if f:
                    f.write(frag)
                    f.flush()
                else:
                    print(frag, flush=True, end='')
        if stream:
            if f:
                f.close()
            else:
                print()
        stop = response.finish_reason
        # R1 <think> ... </think> {
        text = outputs[outputs.rfind(self.think)+len(self.think):].strip()
        # }
        benchmark = f"Prompt: {response.prompt_tokens} tokens, {response.prompt_tps:.3f} tokens-per-sec\nGeneration: {response.generation_tokens} tokens, {response.generation_tps:.3f} tokens-per-sec"
        return dict(text=text, outputs=outputs, benchmark=benchmark, stop=stop)
