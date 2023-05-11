import os
import numpy as np
import math

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.linen.dtypes import promote_dtype

from typing import Optional, Any


Dtype = Any

def scaled_dot_attention(query : jnp.ndarray,
                         key : jnp.ndarray,
                         value : jnp.ndarray,
                         mask : Optional[jnp.ndarray] = None,
                         dtype : Optional[Dtype] = None
                       ):


    """
    Compute scaled attention given query, key and value from the paper 'Attention is All You Need'
    This method is for multi-head attention

    Args :

        query : b x ... x num_heads x seq_len x d_k_per_head
        key   : b x ... x num_heads x seq_len x d_k_per_head
        value : b x ... x num_heads x seq_len x d_k_per_head
        mask  : b x ... x num_heads x seq_len x d_k_per_head

    """

    # for q, k and v check the number of dimensions.

    assert key.ndim == query.ndim == value.ndim, 'query, key and value must have same rank'
    assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], 'query, key and value batch dimensions must match'
    assert query.shape[-3] == key.shape[-3] == value.shape[-3], 'query, key and value must have same number of heads (-2 dimension)'
    assert query.shape[-2] == key.shape[-2] == value.shape[-2], 'query, key and value must have the same sequence length dimension (-3 dimension)'
    assert query.shape[-1] == key.shape[-1] == value.shape[-1], 'query, key and value must have the same token dimension d_k (-1 dimension)'

    # make sure that q, k and v are can be broadcasted to the smallest common dtype
    query, key, value = promote_dtype(query, key, value, dtype = dtype)
    dtype = query.dtype

    d_k = query.shape[-1]
    attn_logits = jnp.matmul(query, jnp.swapaxes(key, -2, -1)) # Q.K^T, b x seq_len x num_heads x num_heads

    attn_logits = attn_logits / jnp.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask ==0, -9e15, attn_logits)

    attention = nn.softmax(attn_logits, axis = -1) # weights for the values
    values = jnp.matmul(attention, value) # b x seq_len x num_heads x d_k

    return values






