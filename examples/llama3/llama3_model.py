# native torch implementation of llama 3

import torch 
import torch.nn as nn

class RMSNorm(nn.Module): 
    def __init__(self, dim: int, eps: float = 1e-8): 
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        output = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return output * self.weight
    

# test 
vec = torch.tensor([1.0, 2.0, 3.0])
norm = RMSNorm(3)(vec)
actual = nn.RMSNorm(3, eps = 1e-8)(vec)

print(norm)
print(actual)

# test 2 
vec = torch.tensor([1.0, 2.0, 3.0])
norm = RMSNorm(3)(vec)
actual = nn.RMSNorm(3, eps = 1e-8)(vec)


# Instead of Gelu, use SwiGLU

class SiLU(nn.Module): 
    def __init__(self): 
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return x * torch.sigmoid(x)
    

# test 
vec = torch.tensor([1.0, 2.0, 3.0])
silu = SiLU()(vec)
actual = torch.nn.SiLU()(vec)

print(silu, actual)

class FeedForward(nn.Module): 
    def __init__(self, embed_dim, hidden_dim, dtype): 
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim, dtype = dtype, bias = False)
        self.linear2 = nn.Linear(embed_dim, hidden_dim, dtype = dtype, bias = False)
        self.linear_out = nn.Linear(hidden_dim, embed_dim, dtype = dtype, bias = False)
        self.silu = SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        out = self.silu(x1) * x2 
        return self.linear_out(out)
    
    

# Rope 

# ok so ihave tried FIFTY different versions of rope and this is the one that seems to wokr, at this point im not gonna try to implement it myself 
# and have this guy's code -- i thought you have to interleave the cos and sin for adjacent head dim components, but i guess it's instead 
# first half of the head dim is cos, second half is sin 
# i jacked this from some guy's 3.2 code, honestly i have no idea what weird smoothing stuff they r doing  for 3.2 
# # Frequency adjustments
def rope_params(head_dim, exp_base=10_000, context_len=4096, freq_config=None):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (exp_base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))

    
    
    
    if freq_config is not None:
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

        wavelen = 2 * torch.pi / inv_freq

        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        smoothed_inv_freq = (
            (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        )

        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama


    # Generate position indices
    positions = torch.arange(context_len)

    # Compute the angles – one value for every pair (2 dims) in each head
    # Shape: (context_length, head_dim // 2)
    angles = positions[:, None] * inv_freq[None, :]

    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Pre-compute sine and cosine for the half-dimension; they will be
    # broadcast to the full head inside `apply_rope`.
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

# test 
head_dim = 4096
freqs_cis = rope_params(head_dim)
print(freqs_cis)


def apply_rope(x, cos_terms, sin_terms): 
    # assert head_dim % 2 == 0, "head dim must be even for rope" 

    # # Split into even / odd components
    # x_even = x[..., ::2]   # (B, H, S, D/2)
    # x_odd  = x[..., 1::2]

    # cos_terms = cos_terms[:seq_len].unsqueeze(0).unsqueeze(0)  # (1,1,S,D/2)
    # sin_terms = sin_terms[:seq_len].unsqueeze(0).unsqueeze(0)

    # # Apply the rotation without in-place clobbering
    # x_rot_even = x_even * cos_terms - x_odd * sin_terms
    # x_rot_odd  = x_even * sin_terms + x_odd * cos_terms

    # # Interleave back to (B, H, S, D)
    # x_out = torch.empty_like(x)
    # x_out[..., ::2] = x_rot_even
    # x_out[..., 1::2] = x_rot_odd
    # return x_out
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos_terms = cos_terms[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin_terms = sin_terms[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos_terms) + (rotated * sin_terms)

    return x_rotated.to(dtype=x.dtype)


print("rope test")
batch_size = 2
context_len = 5
num_heads = 4
head_dim = 16

# Instantiate RoPE parameters
cos, sin = rope_params(head_dim=head_dim, context_len=context_len)

# Dummy query and key tensors
torch.manual_seed(123)
queries = torch.randn(batch_size, num_heads, context_len, head_dim)
keys = torch.randn(batch_size, num_heads, context_len, head_dim)

# Apply rotary position embeddings
queries_rot = apply_rope(queries, cos, sin)
keys_rot = apply_rope(keys, cos, sin)

# the rope params are of shape (seq__len, head_dim//2), because for each token embedding, 
# we have a vector of size head_dim // 2 which represents the cosine (or sine) component of that angle 

# then, we take an x of shape (batch, num_heads, seq_len, head_dim) and split it into 
# (batch, num_heads, seq_len, head_dim // 2) 
# now we notice that we can just apply the per token rope to our x vector 
print(queries_rot.shape, keys_rot.shape)



# next step: grouped query attention MHA 

class GroupedQueryAttention(nn.Module): 
    def __init__(self, embed_dim, out_dim, num_heads, num_kv_groups, dtype, device): 
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.num_kv_groups = num_kv_groups
        self.out_dim = out_dim
        self.device = device
        # one key is shared across num_kv_groups queries 
        # Previously, we would have out_dim = num_heads * head_dim
        # we could think of having "num_heads" amount of keys/values each with size head_dim, but now instead we just have num_kv_groups maount of these head_dim vectors 


        self.Wk = nn.Linear(embed_dim, num_kv_groups * self.head_dim, dtype = dtype, bias = False)
        self.Wq = nn.Linear(embed_dim, num_heads * self.head_dim, dtype = dtype, bias = False)
        self.Wv = nn.Linear(embed_dim, num_kv_groups * self.head_dim, dtype = dtype, bias = False)



        self.out = nn.Linear(num_heads * self.head_dim, out_dim, dtype = dtype, bias = False)

        #self.norm = RMSNorm(embed_dim)
        #self.rope = rope_params(embed_dim)

    def forward(self, x: torch.Tensor, mask, cos, sin) -> torch.Tensor: 
        batch_size, seq_len, embed_dim = x.shape 

        queries = self.Wq(x)
        keys = self.Wk(x)
        values = self.Wv(x)

        # split into heads 
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, seq_len, self.num_kv_groups, self.head_dim)
        values = values.view(batch_size, seq_len, self.num_kv_groups, self.head_dim)



        # we want the inner dimensions to be (seq_len, head_dim) so that we can do this BMM for every head 

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2) 
        values = values.transpose(1, 2) 

        # now, we want to repeat every key and value self.num_heads/self.num_kv_groups times 
        
        # next, lert us apply rope 

        # important! we have to do ROPE before we do the repetition for the keys and values, because we dont want the repliated key/value 
        # in a group to have different rope angles 

        if cos is not None: 
            queries = apply_rope(queries, cos, sin)
            keys = apply_rope(keys, cos, sin) # we dont apply rope to the values 


        keys = keys.repeat_interleave(self.num_heads // self.num_kv_groups, dim = 1)
        values = values.repeat_interleave(self.num_heads // self.num_kv_groups, dim = 1)
        
        # ok now we want to do the BMM of shape (batch_size, num_heads, seq_len, head_dim) by (batch_size, num_heads, head_dim, seq_len) 
        s = queries @ keys.transpose(-2, -1) 
        # OK BUT WE WANT CAUSAL ATTENTION so this means that any q_i k_j dot product should be 0 if j > i 
        # in other words, keys at positions higher than a query's position cannot possibly attend to it \

        # so we want to make the upper triangular part of the s matrix to be -inf , EXCLUDE THE DIAGONAL 
        if mask is None: 
            mask = torch.triu(torch.ones(seq_len, seq_len, dtype = torch.bool, device = self.device ), diagonal = 1)

        s.masked_fill_(mask, -torch.inf) 
        # now we can do the softmax 
        p = torch.softmax(s / self.head_dim ** 0.5, dim = -1) # we do the softmax ACROSS ROWS

        # now, our p "scores" matrix is of size (seq_len, seq_len) 
        # and our values is of size (batch_size, num_heads, seq_len, head_dim) 
        # matmul these to get our weighted output tokens of size (batch_size, num_heads, seq_len, head_dim) 

        o = p @ values 
        # but now we need to reflip the num_heads and seq_len dimensions to be what they were before, so that we can concat the heads 
        # I think of our current "o" matrix as the sequence of per-head vectors across all heads. 
        # So for every head \in num_head, we have seq_len number of head_dim sized vectors. 
        # This transpose will basically collect all those per-head vectors across the seq_len 
        # so now for every position in our sequence length, we will have a num_heads * head_dim "matrix" that represents our output 
        # and then we just flatten those two dimensions to get our final output! 

        assert self.out_dim == self.num_heads * self.head_dim, "out_dim must be num_heads * head_dim" 
        output = o.transpose(1, 2).reshape(batch_size, seq_len, self.out_dim) 

        # lastly, return the out-projected output, of size (batch_size, seq_len, out_dim) # WAHOO 
        return self.out(output), p






        

        # apply mask 
        if mask is not None: 
            mask = mask.view(batch_size, 1, 1, seq_len)
            queries = queries.masked_fill(~mask, float("-inf"))
            keys = keys.masked_fill(~mask, float("-inf"))
            values = values.masked_fill(~mask, float("-inf"))
            


# ok lets test 
batch_size = 1
context_len = 3000
max_context_len = 8192
embed_dim = 4096
num_heads = 32

dummy_batch = torch.randn(batch_size, context_len, embed_dim) 

gqa = GroupedQueryAttention(embed_dim, embed_dim, num_heads, num_kv_groups = 8, dtype = torch.float32, device = "cpu")

gqa(dummy_batch, None, None, None)



# ok bruh finally we can do the transformer block 

class TransformerBlock(nn.Module): 
    def __init__(self, embed_dim, hidden_dim, num_heads, num_kv_groups, dtype, device): 
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.dtype = dtype
        self.device = device

        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        self.gqa = GroupedQueryAttention(embed_dim, embed_dim, num_heads, num_kv_groups, dtype, device)
        self.ffn = FeedForward(embed_dim, hidden_dim, dtype)

    def forward(self, x: torch.Tensor, mask = None, cos = None, sin = None) -> torch.Tensor:
        attn_out = self.gqa(self.norm1(x), mask, cos, sin) 
        skipped = x + attn_out 
        ffn_out = self.ffn(self.norm2(skipped))

        return ffn_out + skipped
    


default_llama_config = {
    "vocab_size": 32000, 
    "embed_dim": 4096, 
    "num_heads": 32, 
    "num_kv_groups": 8, 
    "num_layers": 32, 
    "rope_base": 10000,
    "max_context_len": 8192, 
    "dtype": torch.bfloat16, 
    "device": "cpu"
}
class Llama3(nn.Module): 
    def __init__(self, config): 
        super().__init__()
        self.emb_table = nn.Embedding(config["vocab_size"], config["embed_dim"])
        self.layers = nn.Sequential(*[TransformerBlock(config["embed_dim"], config["hidden_dim"], config["num_heads"], config["num_kv_groups"], config["dtype"], config["device"]) for _ in range(config["num_layers"])])
        self.norm = RMSNorm(config["embed_dim"], eps = 1e-5)
        self.device = config["device"]
        # OUTPUT HEAD 
        self.lm_head = nn.Linear(config["embed_dim"], config["vocab_size"], bias = False, dtype = config["dtype"])

        # precompute rope stuff 
        self.cos, self.sin = rope_params(config["embed_dim"]//config["num_heads"], exp_base=config["rope_base"], context_len = config["context_length"], freq_config = config["rope_freq"])


        self.config = config 


    def forward(self, x: torch.Tensor, mask = None, cos = None, sin = None) -> torch.Tensor: 
        x = self.emb_table(x)
        
        seq_len = x.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype = torch.bool, device = self.device), diagonal = 1)

        self.cos = self.cos.to(self.device)
        self.sin = self.sin.to(self.device)

        for layer in self.layers: 
            x = layer(x, mask, self.cos, self.sin)

        x = self.norm(x) 
        logits = self.lm_head(x.to(self.config["dtype"]))

        return logits 
    

default_llama_config = {
    "vocab_size": 32000, 
    "embed_dim": 4096, 
    "num_heads": 32, 
    "num_kv_groups": 8, 
    "num_layers": 32, 
    "rope_base": 10000,
    "max_context_len": 8192, 
    "dtype": torch.bfloat16, 
    "device": "cpu"
}

LLAMA3_CONFIG_8B = {
    "vocab_size": 128_256,   # NEW: Larger vocabulary size
    "context_length": 8192,  # NEW: Larger context length
    "embed_dim": 4096,         # Embedding dimension
    "num_heads": 32,           # Number of attention heads
    "num_layers": 32,          # Number of layers
    "hidden_dim": 14_336,    # NEW: Larger size of the intermediate dimension in FeedForward
    "num_kv_groups": 8,        # NEW: Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,  # NEW: The base in RoPE's "theta" was increased to 500_000
    "rope_freq": None,       # NEW: Additional configuration for adjusting the RoPE frequencies
    "device": torch.device("mps"),
    "dtype": torch.bfloat16  # Lower-precision dtype to reduce memory usage
}

LLAMA32_CONFIG = {
    "vocab_size": 128_256,           # Vocabulary size
    "context_length": 131_072,       # Context length that was used to train the model
    "embed_dim": 2048,                 # Embedding dimension
    "num_heads": 32,                   # Number of attention heads
    "num_layers": 16,                  # Number of layers
    "hidden_dim": 8192,              # Size of the intermediate dimension in FeedForward
    "num_kv_groups": 8,                # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,          # The base in RoPE's "theta"
    "device": torch.device("mps"),
    "dtype": torch.bfloat16,         # Lower-precision dtype to reduce memory usage
    "rope_freq": {                   # RoPE frequency scaling
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}



def assign (model_tensor, loaded_tensor, tensor_name="null"): 
    assert model_tensor.shape == loaded_tensor.shape, f"Shape mismatch for {model_tensor} and {loaded_tensor}"
    if isinstance(loaded_tensor, torch.Tensor): 
        return torch.nn.Parameter(loaded_tensor.clone().detach()) 
    else: 
        return torch.nn.Parameter(torch.tensor(loaded_tensor))
    


def load_weights_into_llama(model, param_config, params):
    model.emb_table.weight = assign(model.emb_table.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")

    # for every transformer block, we load in 
    # Wq, Wk, Wv, Wout
    # Then, the weights for RMSnorm1 (the one before attention) 
    # then the weights for RMSnorm2 (the one after attention) 
    # then the feed forward is with swiglu, so we have 
    # one linear layer taking to hidden dim 
    # one linear layer taking to hidden dim of the sigmoid 
    # one linear layer being the out projection 

    for l in range(param_config["num_layers"]):

        # Load attention weights
        model.layers[l].gqa.Wq.weight = assign(
            model.layers[l].gqa.Wq.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        model.layers[l].gqa.Wk.weight = assign(
            model.layers[l].gqa.Wk.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        model.layers[l].gqa.Wv.weight = assign(
            model.layers[l].gqa.Wv.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )
        model.layers[l].gqa.out.weight = assign(
            model.layers[l].gqa.out.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight"
        )
        model.layers[l].norm1.weight = assign(
            model.layers[l].norm1.weight,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )

        # Load FeedForward weights
        model.layers[l].ffn.linear1.weight = assign(
            model.layers[l].ffn.linear1.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight"
        )
        model.layers[l].ffn.linear2.weight = assign(
            model.layers[l].ffn.linear2.weight,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight"
        )
        model.layers[l].ffn.linear_out.weight = assign(
            model.layers[l].ffn.linear_out.weight,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight"
        )
        model.layers[l].norm2.weight = assign(
            model.layers[l].norm2.weight,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )

    # Load output layer weights
    model.norm.weight = assign(model.norm.weight, params["model.norm.weight"], "model.norm.weight")

    if "lm_head.weight" in params.keys():
        model.lm_head.weight = assign(model.lm_head.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        model.lm_head.weight = assign(model.lm_head.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")
        print("Model uses weight tying.")


load_weights_into_llama(model, LLAMA3_CONFIG_8B, combined_weights)

print("done model loading")
model.to(device);
del combined_weights 


token_ids = generate(
    model=model,
    idx=text_to_idxs("Bello all! My name is ", tokenizer).to(device),
    max_new_tokens=30,
    context_size=LLAMA3_CONFIG_8B["context_length"],
    top_k=5,
    temperature=0.
)

print("Output text:\n", idxs_to_text(token_ids, tokenizer))