from pathlib import Path

import tiktoken
from tiktoken.load import load_tiktoken_bpe
import torch 
import os 



# just get this from meta's stuff 
class Tokenizer:
    """Thin wrapper around tiktoken that keeps track of Llama-3 special IDs."""
    def __init__(self, model_path):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)

        mergeable = load_tiktoken_bpe(model_path)

        self.special = {
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }
        self.special.update({f"<|reserved_{i}|>": 128002 + i
                             for i in range(256)
                             if 128002 + i not in self.special.values()})

        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
                    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
                    r"|\p{N}{1,3}"
                    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
                    r"|\s*[\r\n]+"
                    r"|\s+(?!\S)"
                    r"|\s+",
            mergeable_ranks=mergeable,
            special_tokens=self.special,
        )

    def encode(self, text, bos=False, eos=False):
        ids = ([self.special["<|begin_of_text|>"]] if bos else []) \
              + self.model.encode(text)
        if eos:
            ids.append(self.special["<|end_of_text|>"])
        return ids

    def decode(self, ids):
        return self.model.decode(ids)
    


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    for _ in range(max_new_tokens): 
        idx_cond = idx[:, -context_size:] # what does this do? 
        with torch.no_grad(): 
            logits = model(idx_cond) 

        logits = logits[:, -1, :]

        if top_k is not None: 
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, -1]] = -float('inf')

        if temperature > 0.0: 
            logits = logits / temperature 
            probs = torch.softmax(logits, dim = -1) 


            idx_next = torch.multinomial(probs, num_samples = 1)
        else: 
            # if zero temperature just do argmax 
            idx_next = torch.argmax(logits, dim = -1, keepdim = True)
        if idx_next == eos_id: 
            break 

        idx = torch.cat((idx, idx_next), dim = 1)

    return idx 


def text_to_idxs(text, tokenizer): 
    encoded = tokenizer.encode(text, bos = True)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) 
    return encoded_tensor 


def idxs_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())