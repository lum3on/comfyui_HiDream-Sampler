from typing import Optional
import torch
from .attention import HiDreamAttention

USE_FLASH_ATTN = False
USE_FLASH_ATTN3 = False


# Copied from https://github.com/black-forest-labs/flux/blob/main/src/flux/math.py
def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    # Check if we've already determined the attention type
    if not hasattr(attention, "_attention_type"):
        # Determine which attention implementation to use
        attention_type = "sdpa"  # default
        
        try:
            import importlib.util
            if importlib.util.find_spec("sageattention"):
                attention_type = "SageAttention"
            elif importlib.util.find_spec("flash_attn"):
                attention_type = "FlashAttention2"
            elif importlib.util.find_spec("flash_attn_interface"):
                attention_type = "FlashAttention3"
        except:
            pass
        
        # Cache the result
        attention._attention_type = attention_type
        print(f"using {attention_type}")
    
    # Get the cached attention type
    attention_type = attention._attention_type

    
    # Execute the appropriate attention implementation
    if attention_type == "SageAttention":
        from sageattention import sageattn
        hidden_states = sageattn(query, key, value, tensor_layout="NHD", is_causal=False)
    elif attention_type == "FlashAttention2":
        from flash_attn import flash_attn_func
        hidden_states = flash_attn_func(query, key, value, dropout_p=0., causal=False)
    elif attention_type == "FlashAttention3":
        from flash_attn_interface import flash_attn_func
        hidden_states = flash_attn_func(query, key, value, causal=False, deterministic=False)[0]
    elif attention_type == "sdpa":
        q = query.transpose(1, 2)  
        k = key.transpose(1, 2)    
        v = value.transpose(1, 2)
        
        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2)
    else:
        raise ValueError("Invalid attention implementation")

    hidden_states = hidden_states.flatten(-2)
    hidden_states = hidden_states.to(query.dtype)
    return hidden_states

class HiDreamAttnProcessor_flashattn:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __call__(
        self,
        attn: HiDreamAttention,
        image_tokens: torch.FloatTensor,
        image_tokens_masks: Optional[torch.FloatTensor] = None,
        text_tokens: Optional[torch.FloatTensor] = None,
        rope: torch.FloatTensor = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        dtype = image_tokens.dtype
        batch_size = image_tokens.shape[0]

        query_i = attn.q_rms_norm(attn.to_q(image_tokens)).to(dtype=dtype)
        key_i = attn.k_rms_norm(attn.to_k(image_tokens)).to(dtype=dtype)
        value_i = attn.to_v(image_tokens)

        inner_dim = key_i.shape[-1]
        head_dim = inner_dim // attn.heads

        query_i = query_i.view(batch_size, -1, attn.heads, head_dim)
        key_i = key_i.view(batch_size, -1, attn.heads, head_dim)
        value_i = value_i.view(batch_size, -1, attn.heads, head_dim)
        if image_tokens_masks is not None:
            key_i = key_i * image_tokens_masks.view(batch_size, -1, 1, 1)

        if not attn.single:
            query_t = attn.q_rms_norm_t(attn.to_q_t(text_tokens)).to(dtype=dtype)
            key_t = attn.k_rms_norm_t(attn.to_k_t(text_tokens)).to(dtype=dtype)
            value_t = attn.to_v_t(text_tokens)

            query_t = query_t.view(batch_size, -1, attn.heads, head_dim)
            key_t = key_t.view(batch_size, -1, attn.heads, head_dim)
            value_t = value_t.view(batch_size, -1, attn.heads, head_dim)

            num_image_tokens = query_i.shape[1]
            num_text_tokens = query_t.shape[1]
            query = torch.cat([query_i, query_t], dim=1)
            key = torch.cat([key_i, key_t], dim=1)
            value = torch.cat([value_i, value_t], dim=1)
        else:
            query = query_i
            key = key_i
            value = value_i
        
        if query.shape[-1] == rope.shape[-3] * 2:
            query, key = apply_rope(query, key, rope)
        else:
            query_1, query_2 = query.chunk(2, dim=-1)
            key_1, key_2 = key.chunk(2, dim=-1)
            query_1, key_1 = apply_rope(query_1, key_1, rope)
            query = torch.cat([query_1, query_2], dim=-1)
            key = torch.cat([key_1, key_2], dim=-1)

        hidden_states = attention(query, key, value)

        if not attn.single:
            hidden_states_i, hidden_states_t = torch.split(hidden_states, [num_image_tokens, num_text_tokens], dim=1)
            hidden_states_i = attn.to_out(hidden_states_i)
            hidden_states_t = attn.to_out_t(hidden_states_t)
            return hidden_states_i, hidden_states_t
        else:
            hidden_states = attn.to_out(hidden_states)
            return hidden_states