# Copyright 2025 Josef Albers
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from mlx_lm import load, generate, stream_generate
from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache
import mlx.core as mx
import time
import os
import copy
import re

class Chat:
    def __init__(self, model_path="mlx-community/Qwen2.5-Coder-32B-4bit", fim_template="{context}<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>", think=('<think>', '</think>'), max_kv_size=None, dict_repo=None, cache_dir=''):
        self.model_path = model_path
        self.model, self.tokenizer = load(model_path)
        self.think_start, self.think_end = think
        self.fim_template = fim_template
        self.set_cache_repo(dict_repo, cache_dir=cache_dir, max_kv_size=max_kv_size)
        self.reset(max_kv_size=max_kv_size)
    def fim(self, prefix, suffix, max_new=500, current_path=None):
        cache, context = self.get_cache_repo(current_path)
        prompt = self.fim_template.format(context=context, prefix=prefix, suffix=suffix.rstrip())
        autocomplete = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_new, verbose=False, prompt_cache=cache)
        return dict(autocomplete=autocomplete, prompt=prompt, prefix=prefix, suffix=suffix, dict_repo=self.dict_repo)
    def set_cache_repo(self, dict_repo, cache_dir='', max_kv_size=None):
        cache_path = os.path.join(cache_dir, re.sub(r'[^a-zA-Z0-9]', '_', self.model_path)+'_repo.safetensors')
        self.dict_repo = dict_repo
        self.repo_cache = None
        if not dict_repo:
            return
        try:
            loaded_cache, loaded_dict = load_prompt_cache(cache_path, True)
            loaded_dict = eval(loaded_dict)
            if max(loaded_dict['list_mtime'])+1 >= max(dict_repo['list_mtime']) and set(loaded_dict['repo_files']) == set(dict_repo['repo_files']):
                mx.eval(loaded_cache)
                self.repo_cache = loaded_cache
                return
            else:
                del loaded_cache
                del loaded_dict
        except:
            pass
        self.repo_cache = make_prompt_cache(self.model, max_kv_size=max_kv_size)
        generate(self.model, self.tokenizer, prompt=''.join(dict_repo['list_content']), max_tokens=1, verbose=False, prompt_cache=self.repo_cache)
        save_prompt_cache(cache_path, self.repo_cache, str(dict_repo))
    def get_cache_repo(self, current_path): 
        cache = None
        context = ''
        if not self.dict_repo:
            return cache, context
        for p, n in zip(self.dict_repo['rest_files'], self.dict_repo['rest_names']):
            if p == current_path:
                continue
            try:
                with open(p, 'r') as f:
                    context += f'<|file_sep|>{n}\n{f.read().rstrip()}\n'
            except:
                pass
        if len(self.dict_repo['repo_files']) + len(context) == 0:
            return cache, context
        if self.repo_cache:
            cache = copy.deepcopy(self.repo_cache)
        else:
            context = self.dict_repo['list_content'][0] + context
        n = os.path.relpath(current_path, self.dict_repo['repo_path'])
        context += f'<|file_sep|>{n}\n'
        return cache, context
    def reset(self, system='', max_kv_size=None):
        self.prompt_cache = make_prompt_cache(self.model, max_kv_size=max_kv_size)
        self.stop = None
        self.toks = None
        self.ongoing = None
        self.output_toks = None
        self.hx_toks = []
    def get_ntok(self, s):
        return len(self.tokenizer.encode(s))
    def resume(self, max_new=500, verbose=False, stream=None):
        if self.stop == 'stop' or self.ongoing == None:
            return dict(text='', output='', hx='', benchmark='n/a', stop=self.stop)
        if self.stop == 'length':
            self.ongoing = stream_generate(self.model, self.tokenizer, prompt=self.toks, max_tokens=max_new, prompt_cache=self.prompt_cache) 
        return self.generate(inputs='', toks=self.toks, max_new=max_new, verbose=verbose, stream=stream)
    def __call__(self, inputs, max_new=500, chat_fmt=True, verbose=False, stream=None):
        self.output_toks = []
        if isinstance(inputs, str):
            prompt = inputs
        else:
            prompt = inputs[0]
        if self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            toks = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
        self.hx_toks += toks
        self.ongoing = stream_generate(self.model, self.tokenizer, prompt=toks, max_tokens=max_new, prompt_cache=self.prompt_cache)
        return self.generate(inputs=prompt, toks=toks, max_new=max_new, verbose=verbose, stream=stream) 
    def generate(self, inputs, toks, max_new, verbose, stream):
        f = None
        if isinstance(stream, str):
            try:
                f = open(stream, 'a', encoding='utf-8')
            except:
                f = None
        output_toks = []
        for response in self.ongoing:
            if stream:
                frag = response.text
                if f:
                    f.write(frag)
                    f.flush()
                else:
                    print(frag, flush=True, end='')
            output_toks.append(response.token)
        output_toks = output_toks[:-1]
        self.output_toks += output_toks
        self.hx_toks += output_toks
        self.stop = response.finish_reason
        self.toks = mx.array([response.token])
        output = self.tokenizer.decode(self.output_toks)
        text = output.split(self.think_end)[-1].strip()
        hx = self.tokenizer.decode(self.hx_toks)
        benchmark = f"Prompt: {response.prompt_tokens} tokens, {response.prompt_tps:.3f} tokens-per-sec\nGeneration: {response.generation_tokens} tokens, {response.generation_tps:.3f} tokens-per-sec"
        if verbose:
            verbosity = f'\n\n**INPUT**\n{inputs}\n\n**OUTPUT**\n{text}\n\n**BENCHMARK**\n{benchmark}'
            if f:
                f = f.write(verbosity)
                f.flush()
            else:
                print(f'\033[34m{verbosity}\033[0m')
        if stream:
            if f:
                f.close()
            else:
                print()
        return dict(text=text, output=output, hx=hx, benchmark=benchmark, stop=self.stop)

