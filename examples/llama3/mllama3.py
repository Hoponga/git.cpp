# multimodal llama 3.2 

# im gonna try to keep this as close to the base llama 3 implementation as possible 

# ALSO MY OTHER GOAL IS TO NOT PASS IN RANDOM CONFIGS EVERYWHERE except to the high level vision/text models 

import torch 
import torch.nn as nn 
from llama3_model import Llama3, TransformerBlock, RMSNorm, rope_params, GroupedQueryAttention, apply_rope, FeedForward
from mllama_image_utils import MllamaPrecomputedAspectRatioEmbedding, MllamaPrecomputedPositionEmbedding, _prepare_aspect_ratio_attention_mask, _prepare_cross_attention_mask

import torch.nn.functional as F 

# TODO: Replace this with own image processor later 
from transformers.models.mllama.processing_mllama import MllamaProcessor 
from PIL import Image
import requests

# processor = MllamaProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision")

# url = "https://cdn.pixabay.com/photo/2017/03/07/22/17/cabin-2125387_1280.jpg"
# #local_path ="/root/bird.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# print(processor(
#     images=image,
#     text=["<|image|>If I had to write a haiku for this one"],
#     images_kwargs = {"size": {"height": 448, "width": 448}},
#     text_kwargs = {"padding": "right"},
#     common_kwargs = {"return_tensors": "pt"},
# ))


# Architecture guide: https://j-qi.medium.com/inside-mllama-3-2-understanding-metas-vision-language-model-architecture-ae12ad24dcbf 


# single hidden layer with GELU activation 
class VisionFeedForward(nn.Module): 
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim, bias = True)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, embed_dim, bias = True)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.linear1(x)
        x = self.gelu(x)
        return self.linear2(x)

class MMTransformerBlock(nn.Module):
    """Transformer block that can operate in either self-attention or cross-attention mode."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_kv_groups: int,
        dtype: torch.dtype,
        device: str | torch.device,
        *,
        cross_attn: bool = False,
    ) -> None:
        super().__init__()
        self.cross_attn = cross_attn
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)

        if cross_attn:
            self.attn = CrossGroupedQueryAttention(
                embed_dim,
                embed_dim,
                num_heads,
                num_kv_groups,
                dtype,
                device,
            )
        else:
            # Fallback to regular self-attention GQA
            self.attn = GroupedQueryAttention(
                embed_dim,
                embed_dim,
                num_heads,
                num_kv_groups,
                dtype,
                device,
            )
        self.ffn = FeedForward(embed_dim, hidden_dim, dtype)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        *,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        

        if self.cross_attn:
            if context is None:
                raise ValueError("context must be provided when cross_attn is True")
            attn_out = self.attn(self.norm1(x), context, mask, cos, sin)
        else:
            attn_out = self.attn(self.norm1(x), mask, cos, sin)

        skipped = x + attn_out
        ffn_out = self.ffn(self.norm2(skipped))
        return ffn_out + skipped

# ok this is the super cooked part 
# so we have the vision encoder which is a 32 layer transformer and the global transformer which is an 8 layer transformer 

class MLLama3VisionModel(nn.Module): 
    def __init__(self, config): 
        super().__init__()

        # image pre-processing/patching taken from source code: 
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.max_num_tiles = config["max_num_tiles"]
        self.hidden_size = config["vision_embed_dim"]
        self.num_channels = config["num_channels"]
        self.intermediate_layers_indices = config["intermediate_layers_indices"] # which layers to collect intermediate states from for concatenation

        self.num_patches = (self.image_size // self.patch_size) ** 2 + 1
        self.scale = config["vision_embed_dim"]**-0.5

        self.patch_embedding = nn.Conv2d(
            in_channels=config["num_channels"],
            out_channels=config["vision_embed_dim"],
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
            bias=False,
        )

        # output shape of this convolution: (batch_size, num_patches, hidden_size)
        # todo: positional embedding stuff: 

        self.norm1 = nn.LayerNorm(config["vision_embed_dim"], eps = config["norm_eps"])
        self.norm2 = nn.LayerNorm(config["vision_embed_dim"], eps = config["norm_eps"])


        # the encoder is not gated, the global transformer is 
        self.encoder = MLLama3VisionEncoder(config["vision_embed_dim"], config["vision_hidden_dim"], config["vision_num_heads"], config["dtype"], config["device"], config["norm_eps"], False, config["vision_encoder_num_layers"])
        self.global_transformer = MLLama3VisionEncoder(config["vision_embed_dim"], config["vision_hidden_dim"], config["vision_num_heads"], config["dtype"], config["device"], config["norm_eps"], True, config["global_encoder_num_layers"])


    # this is the function signature from the source code 
    def forward(
        self, pixel_values: torch.Tensor, aspect_ratio_ids: torch.Tensor, aspect_ratio_mask: torch.Tensor, output_attentions = False, output_hidden_states = False, return_dict = False): 

        # todo: bunch of pre-processing 
        # pixel values shape follows (batch_size, num_concurrent_media, num_tiles, num_channels, height, width)

        B, M, T, C, H, W = pixel_values.shape 

        pixel_values = pixel_values.reshape(B * M * T, C, H, W)
        aspect_ratio_ids = aspect_ratio_ids.reshape(B * M, -1)

  
        target_dtype = self.patch_embedding.weight.dtype
        target_device = self.patch_embedding.weight.device
        patch_embeds = self.patch_embedding(pixel_values.to(target_device, target_dtype))
        hidden_state = patch_embeds.flatten(2).transpose(1, 2)


        _, num_patches, dim = hidden_state.shape
        hidden_state = hidden_state.reshape(B * M * T, -1, dim)
        hidden_state = self.pre_tile_positional_embedding(hidden_state, aspect_ratio_ids)


        hidden_state = hidden_state.reshape(B * M * T, num_patches, dim)
        hidden_state = self.apply_class_embedding(hidden_state)
        num_patches += 1

        # Position embeddings
        hidden_state = hidden_state.reshape(B * M * T, num_patches, dim)
        hidden_state = self.gated_positional_embedding(hidden_state, aspect_ratio_ids)

        hidden_state = self.norm1(hidden_state)

        # Compute the number of tokens to pad
        num_padding_patches = (8 - (hidden_state.shape[-2] % 8)) % 8
        # Compute padding tuple for pad function
        padding = (0, 0, 0, num_padding_patches)  # (pad_left, pad_right, pad_left for dim -2, pad_right for dim -2)
        # Pad the tensor
        hidden_state = F.pad(hidden_state, padding, mode="constant", value=0)
        slice_index = -num_padding_patches if num_padding_patches > 0 else None

        # Prepare attention mask
        attention_mask = aspect_ratio_mask.reshape(B * M, -1)
        attention_mask = _prepare_aspect_ratio_attention_mask(
            aspect_ratio_mask=attention_mask,
            num_patches=self.num_patches,
            target_length=hidden_state.shape[2],
            dtype=self.dtype,
        )
        hidden_state = hidden_state.view(B * M * T, -1, dim)

        # the 32 layer encoder returns the final hidden state as well as all the intermediate states 
        hidden_state, intermediate_states = self.encoder(hidden_state, mask=attention_mask)

        hidden_state = self.norm2(hidden_state)

        # Apply global encoder
        hidden_state = hidden_state.reshape(
            B * M * T, num_patches + num_padding_patches, dim
        )
        hidden_state = self.post_tile_positional_embedding(hidden_state, aspect_ratio_ids)
        hidden_state = hidden_state.reshape(
            B * M * T, (num_patches + num_padding_patches), dim
        )


        hidden_state = self.global_transformer(hidden_state, mask=attention_mask)

        # Remove padding form hidden state
        hidden_state = hidden_state.reshape(
            B * M * T, num_patches + num_padding_patches, dim
        )
        hidden_state = hidden_state[:, :, :slice_index]
        hidden_state = hidden_state.reshape(B, M, num_patches, dim)

        # Collect intermediate layer outputs from encoder output
        all_intermediate_hidden_states = [intermediate_states[i] for i in self.intermediate_layers_indices]
        intermediate_hidden_states = torch.stack(all_intermediate_hidden_states, dim=-1)

        # Remove padding from intermediate hidden states
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            B * M * T, num_patches + num_padding_patches, -1
        )
        intermediate_hidden_states = intermediate_hidden_states[:, :, :slice_index]
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            B, M, num_patches, -1
        )

        # Concatenate final hidden state and intermediate hidden states
        hidden_state = torch.cat([hidden_state, intermediate_hidden_states], dim=-1)

        return hidden_state 
        




# vision encoder: 
# 32 layer transformer followed by 8 layer global encoder 
class MLLama3VisionEncoder(nn.Module): 
    def __init__(self, embed_dim, hidden_dim, num_heads, dtype, device, norm_eps, is_gated, num_layers): 
        super().__init__()
        self.layers = nn.Sequential(*[MLLama3VisionEncoderBlock(embed_dim, hidden_dim, num_heads, dtype, device, norm_eps, is_gated) for _ in range(num_layers)])
        self.device = device


    # ok for whatever magical reason to be determined later, we collect all the encoder intermediate states 

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor: 
        intermediate_states = [] 
        
        for layer in self.layers: 
            x = layer(x, mask, None, None)
            intermediate_states.append(x)

        return x, intermediate_states 


# VisionEncoderBlock ~= TransformerBlock
# Differences: 
# from looking at the source code, it looks like the "vision attention" is just MHA rather than GQA in the text decoder
# then, here the norms are LayerNorms instead if RMSNorms
class MLLama3VisionEncoderBlock(nn.Module): 
    def __init__(self, embed_dim, hidden_dim, num_heads, dtype, device, norm_eps, is_gated): 
        super().__init__()
        self.is_gated = is_gated

        self.norm1 = nn.LayerNorm(embed_dim, eps = norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps = norm_eps)

        # mha = gqa with num_heads # of groups 
        self.attn = GroupedQueryAttention(embed_dim, embed_dim, num_heads, num_heads, dtype, device)
        self.ffn = VisionFeedForward(embed_dim, hidden_dim)

        # i just copy pasted this from the source code 
        # idk why ML people use such fancy words like "GATE" but i guess these are just scalar valued weights for 
        # attention output and ffn output going back into residual stream
        if self.is_gated: 
            self.gate_attn = nn.Parameter(torch.ones(1) * torch.pi / 4)
            self.gate_ffn = nn.Parameter(torch.ones(1) * torch.pi / 4)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, cos: torch.Tensor | None = None, sin: torch.Tensor | None = None) -> torch.Tensor: 
        attn_out, attn_weights = self.attn(self.norm1(x), mask, cos, sin)
        if self.is_gated: 
            attn_out = attn_out * self.gate_attn.tanh() 
        
        skipped = x + attn_out 

        ffn_out = self.ffn(self.norm2(skipped))
        if self.is_gated: 
            ffn_out = ffn_out * self.gate_ffn.tanh() 


        # todo: add attention weights outputting logic 

        return ffn_out + skipped 


# ok so the actual decoder model is (in the source code it's called "TextModel" but decoder lowk makes more sense)
# firs the normal embedding table 
# then we have alternating 
# x3 normal selfattention blocks 
# x1 cross attention block with vision input 
# and this alternates for num layers 
# and then the final rms norm & lm head to get token logits 

# POSITION EMBEDDDINGS DETAILS: 
class MLLama3Decoder(nn.Module): 
    def __init__(self, config): 
        super().__init__()
        # self.emb_table = nn.Embedding(config["vocab_size"], config["embed_dim"])
        # self.layers = nn.Sequential(*[TransformerBlock(config["embed_dim"], config["hidden_dim"], config["num_heads"], config["num_kv_groups"], config["dtype"], config["device"]) for _ in range(config["num_layers"])])
        # self.norm = RMSNorm(config["embed_dim"], eps = 1e-5)
        # self.device = config["device"]
        # # OUTPUT HEAD 
        # self.lm_head = nn.Linear(config["embed_dim"], config["vocab_size"], bias = False, dtype = config["dtype"])

        # # precompute rope stuff 
        # self.cos, self.sin = rope_params(config["embed_dim"]//config["num_heads"], exp_base=config["rope_base"], context_len = config["context_length"], freq_config = config["rope_freq"])


        # self.config = config 

        self.cross_attn_layers = config["cross_attn_layers"]

        self.emb_table = nn.Embedding(config["vocab_size"], config["embed_dim"])
        layers = [] 
        # initialize the interleaved cross attention layers, everything else is the exact same as the llama3 decoder 
        for i in range(config["num_layers"]): 
            if i in self.cross_attn_layers: 
                layers.append(MMTransformerBlock(config["embed_dim"], config["hidden_dim"], config["num_heads"], config["num_kv_groups"], config["dtype"], config["device"], cross_attn = True))
            else: 
                layers.append(MMTransformerBlock(config["embed_dim"], config["hidden_dim"], config["num_heads"], config["num_kv_groups"], config["dtype"], config["device"], cross_attn = False))
        self.layers = nn.Sequential(*layers)
        self.norm = RMSNorm(config["embed_dim"], eps = 1e-5)
        self.lm_head = nn.Linear(config["embed_dim"], config["vocab_size"], bias = False, dtype = config["dtype"])

        self.cos, self.sin = rope_params(config["embed_dim"]//config["num_heads"], exp_base=config["rope_base"], context_len = config["context_length"], freq_config = config["rope_freq"])
        self.device = config["device"]


    def forward(self, x: torch.Tensor, mask = None, cos = None, sin = None, vision_context = None) -> torch.Tensor: 
        x = self.emb_table(x)
        
        seq_len = x.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype = torch.bool, device = self.device), diagonal = 1)

        self.cos = self.cos.to(self.device)
        self.sin = self.sin.to(self.device)

        for layer in self.layers: 
            # if the layer is a cross attention layer, we pass in the vision context but no causal mask 
            if layer.cross_attn: 
                x = layer(x, None, self.cos, self.sin, context = vision_context)
            else: 
                x = layer(x, mask, self.cos, self.sin)

        x = self.norm(x) 
        logits = self.lm_head(x.to(self.config["dtype"]))

        return logits 


class CrossGroupedQueryAttention(nn.Module):

    def __init__(self, embed_dim, out_dim, num_heads, num_kv_groups, dtype, device):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.num_kv_groups = num_kv_groups
        self.out_dim = out_dim
        self.device = device



        if self.num_heads % self.num_kv_groups != 0:
            raise ValueError("num_heads must be divisible by num_kv_groups")

        self.Wq = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False, dtype=dtype)
        self.Wk = nn.Linear(embed_dim, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.Wv = nn.Linear(embed_dim, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.out = nn.Linear(num_heads * self.head_dim, out_dim, bias=False, dtype=dtype)


    # in theory the cross attention is just an exact rewrite of self attention but with a separate input for x_kv 
    # todo: check the transformers library implementation for if this is true 

    # i use the proxy kernels notation here: 
    # s: pre-softmax qk^t matrix 
    # p: post-softmax qk^t matrix 
    # o: output p @ v matrix 
    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        mask: torch.Tensor | None = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len_q, _ = x_q.shape
        seq_len_kv = x_kv.shape[1]

        # only diff: use x_kv for k and v 
        q = self.Wq(x_q).view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        k = self.Wk(x_kv).view(batch_size, seq_len_kv, self.num_kv_groups, self.head_dim)
        v = self.Wv(x_kv).view(batch_size, seq_len_kv, self.num_kv_groups, self.head_dim)


        q = q.transpose(1, 2) 
        k = k.transpose(1, 2)  
        v = v.transpose(1, 2)


        if cos is not None and sin is not None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        repeat_factor = self.num_heads // self.num_kv_groups
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

        # don't need mask because image conditioning is non-causal?

    
        s = q @ k.transpose(-2, -1)  
        s = s / (self.head_dim ** 0.5)



        if mask is not None:
            s = s.masked_fill(mask, -torch.inf)

        p = torch.softmax(s, dim=-1)
        o = p @ v  


        o = o.transpose(1, 2).reshape(batch_size, seq_len_q, self.out_dim)
        return self.out(o)



class MLLama3(nn.Module): 
    def __init__(self, config): 
        super().__init__()
        self.vocab_size = config["text_config"]["vocab_size"]
        self.hidden_size = config["text_config"]["hidden_size"]
        self.max_num_tiles = config["vision_config"]["max_num_tiles"]
        self.vision_output_dim = config["vision_config"]["vision_output_dim"]
        self.pad_token_id = config["pad_token_id"] if config["pad_token_id"] is not None else -1

        self.vision_model = MLLama3VisionModel(config["vision_config"])
        self.language_model = MLLama3Decoder(config["text_config"])
        self.multi_modal_projector = nn.Linear(
            config["vision_config"]["vision_output_dim"],
            config["text_config"]["hidden_size"],
            bias=True,
        )