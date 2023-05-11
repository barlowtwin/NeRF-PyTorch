# testing modules independently
import numpy as np
import jax.numpy as jnp
from attention import scaled_dot_attention


# testing scaled dot attention for multi head sqeuences

q = np.random.rand(10,10, 5, 5)
k = np.random.rand(10, 10, 5, 5)
v = np.random.rand(10, 10, 5, 5)

q = jnp.asarray(q)
k = jnp.asarray(k)
v = jnp.asarray(v)

ans = scaled_dot_attention(q, k, v)
print(ans.shape)
