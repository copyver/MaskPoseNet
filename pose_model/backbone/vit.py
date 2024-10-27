import tensorflow as tf
import tensorflow.keras.layers as KL
from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange
from tensorflow import einsum
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer

from pose_model.utils.utils import get_chosen_pixel_feats, sample_pts_feats


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def masked_fill(tensor, mask, value):
    return tf.where(mask, tf.fill(tf.shape(tensor), value), tensor)


def _check_block_type(block):
    if block not in ['self', 'cross']:
        raise ValueError('Unsupported block type "{}".'.format(block))


class PreNorm(Layer):
    def __init__(self, fn, **kwargs):
        super(PreNorm, self).__init__(**kwargs)

        self.norm = KL.LayerNormalization()
        self.fn = fn

    def call(self, x):
        return self.fn(self.norm(x))


class MLP(Layer):
    def __init__(self, dim, hidden_dim, dropout=0.0, **kwargs):
        super(MLP, self).__init__(**kwargs)

        def GELU():
            def gelu(x, approximate=False):
                if approximate:
                    coeff = tf.cast(0.044715, x.dtype)
                    return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * (x + coeff * tf.pow(x, 3))))
                else:
                    return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))

            return KL.Activation(gelu)

        self.net = Sequential([
            KL.Dense(units=hidden_dim),
            GELU(),
            KL.Dropout(rate=dropout),
            KL.Dense(units=dim),
            KL.Dropout(rate=dropout)
        ])

    def call(self, x, training=None):
        return self.net(x, training=training)


class Attention(Layer):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, **kwargs):
        super(Attention, self).__init__(**kwargs)
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = KL.Softmax()
        self.to_qkv = KL.Dense(units=inner_dim * 3, use_bias=False)

        if project_out:
            self.to_out = [
                KL.Dense(units=dim),
                KL.Dropout(rate=dropout)
            ]
        else:
            self.to_out = []

        self.to_out = Sequential(self.to_out)

    def call(self, x, training=None):
        qkv = self.to_qkv(x)
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # dots = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2])) * self.scale
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)

        # x = tf.matmul(attn, v)
        x = einsum('b h i j, b h j d -> b h i d', attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.to_out(x, training=training)

        return x


class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads, dropout=None, name=None):
        super(MultiHeadAttention, self).__init__(name=name)
        if d_model % num_heads != 0:
            raise ValueError('`d_model` must be a multiple of `num_heads`.')

        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = KL.Dense(d_model, name=name+"_dense_q")
        self.proj_k = KL.Dense(d_model, name=name+"_dense_k")
        self.proj_v = KL.Dense(d_model, name=name+"_dense_v")

        self.dropout = KL.Dropout(rate=dropout) if dropout else None

    def call(self, input_q, input_k, input_v, key_weights=None, key_masks=None, attention_factors=None,
             attention_masks=None):
        """
        Args:
            input_q (Tensor): input tensor for query (B, N, C)
            input_k (Tensor): input tensor for key (B, M, C)
            input_v (Tensor): input tensor for value (B, M, C)
            key_weights (Tensor): soft masks for the keys (B, M)
            key_masks (BoolTensor): True if ignored, False if preserved (B, M)
            attention_factors (Tensor): factors for attention matrix (B, N, M)
            attention_masks (BoolTensor): True if ignored, False if preserved (B, N, M)

        Returns:
            hidden_states: torch.Tensor (B, C, N)
            attention_scores: intermediate values
            'attention_scores': torch.Tensor (B, H, N, M), attention scores before dropout
        """
        q = rearrange(self.proj_q(input_q), 'b n (h c) -> b h n c', h=self.num_heads)
        k = rearrange(self.proj_k(input_k), 'b m (h c) -> b h m c', h=self.num_heads)
        v = rearrange(self.proj_v(input_v), 'b m (h c) -> b h m c', h=self.num_heads)
        attention_scores = einsum('bhnc,bhmc->bhnm', q, k) / self.d_model_per_head ** 0.5

        if attention_factors is not None:
            attention_scores *= tf.expand_dims(attention_factors, 1)
        if key_weights is not None:
            attention_scores *= tf.expand_dims(tf.expand_dims(key_weights, 1), 1)
        if key_masks is not None:
            key_masks_expanded = tf.expand_dims(tf.expand_dims(key_masks, 1), 1)
            attention_scores = masked_fill(attention_scores, key_masks_expanded, float('-inf'))
        if attention_masks is not None:
            attention_scores = masked_fill(attention_scores, attention_masks, float('-inf'))

        attention_scores = tf.nn.softmax(attention_scores, axis=-1)

        if self.dropout:
            attention_scores = self.dropout(attention_scores)

        hidden_states = tf.matmul(attention_scores, v)
        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')

        return hidden_states, attention_scores


class RPEMultiHeadAttention(Layer):
    """
    使用了多头注意力机制，并且集成了相对位置编码（RPE）
    """

    def __init__(self, d_model, num_heads, dropout=None, name=None):
        super(RPEMultiHeadAttention, self).__init__(name=name)
        if d_model % num_heads != 0:
            raise ValueError('`d_model` must be a multiple of `num_heads`.')

        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = KL.Dense(d_model, name=name+"_dense_q")
        self.proj_k = KL.Dense(d_model, name=name+"_dense_k")
        self.proj_v = KL.Dense(d_model, name=name+"_dense_v")
        self.proj_p = KL.Dense(d_model, name=name+"_dense_p")

        self.dropout = KL.Dropout(rate=dropout) if dropout else None

    def call(self, input_q, input_k, input_v, embed_qk, key_weights=None, key_masks=None, attention_factors=None):
        q = rearrange(self.proj_q(input_q), 'b n (h c) -> b h n c', h=self.num_heads)
        k = rearrange(self.proj_k(input_k), 'b m (h c) -> b h m c', h=self.num_heads)
        v = rearrange(self.proj_v(input_v), 'b m (h c) -> b h m c', h=self.num_heads)
        p = rearrange(self.proj_p(embed_qk), 'b n m (h c) -> b h n m c', h=self.num_heads)

        attention_scores_p = einsum('bhnc,bhnmc->bhnm', q, p)
        attention_scores_e = einsum('bhnc,bhmc->bhnm', q, k)
        attention_scores = (attention_scores_e + attention_scores_p) / (self.d_model_per_head ** 0.5)

        if attention_factors is not None:
            attention_scores *= tf.expand_dims(attention_factors, 1)
        if key_weights is not None:
            attention_scores *= tf.expand_dims(tf.expand_dims(key_weights, 1), 1)
        if key_masks is not None:
            key_masks_expanded = tf.expand_dims(tf.expand_dims(key_masks, 1), 1)
            attention_scores = masked_fill(attention_scores, key_masks_expanded, float('-inf'))

        attention_scores = tf.nn.softmax(attention_scores, axis=-1)
        if self.dropout:
            attention_scores = self.dropout(attention_scores)

        hidden_states = tf.matmul(attention_scores, v)
        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')

        return hidden_states, attention_scores


class LinearAttention(Layer):
    def __init__(self, d_model, num_heads, focusing_factor=3, name=None):
        super(LinearAttention, self).__init__(name=name)
        if d_model % num_heads != 0:
            raise ValueError('`d_model` must be a multiple of `num_heads`.')
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads
        self.focusing_factor = focusing_factor
        self.kernel_function = tf.nn.relu

        self.proj_q = KL.Dense(self.d_model, name=name+"_dense_q")
        self.proj_k = KL.Dense(self.d_model, name=name+"_dense_k")
        self.proj_v = KL.Dense(self.d_model, name=name+"_dense_v")
        self.scale = self.add_weight(shape=(1, 1, self.d_model), initializer="zeros", trainable=True,
                                     name=name+"_scale")
        self.softplus = tf.nn.softplus

    def call(self, input_q, input_k, input_v):
        q = self.proj_q(input_q)
        k = self.proj_k(input_k)
        v = self.proj_v(input_v)
        scale = self.softplus(self.scale)

        q = self.kernel_function(q) + 1e-6
        k = self.kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = tf.norm(q, axis=-1, keepdims=True)
        k_norm = tf.norm(k, axis=-1, keepdims=True)
        q = q ** self.focusing_factor
        k = k ** self.focusing_factor
        q = (q / tf.norm(q, axis=-1, keepdims=True)) * q_norm
        k = (k / tf.norm(k, axis=-1, keepdims=True)) * k_norm

        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        # i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]
        i, j, c, d = tf.shape(q)[-2], tf.shape(k)[-2], tf.shape(k)[-1], tf.shape(v)[-1]

        z = 1 / (einsum("b i c, b c -> b i", q, tf.reduce_sum(k, axis=1)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = einsum("b j c, b j d -> b c d", k, v)
            x = einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = einsum("b i c, b j c -> b i j", q, k)
            x = einsum("b i j, b j d, b i -> b i d", qk, v, z)
        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)

        return x


class AttentionLayer(Layer):
    def __init__(self, d_model, num_heads, dropout=None, name=None):
        super(AttentionLayer, self).__init__(name=name)
        self.attention = MultiHeadAttention(d_model, num_heads, name=name+"_multihead")
        self.linear = KL.Dense(d_model, name=name+"_dense")
        self.dropout = KL.Dropout(dropout) if dropout else None
        self.norm = KL.LayerNormalization(epsilon=1e-6, name=name+"_layer_norm")

    def call(self, input_states, memory_states, memory_weights=None,
             memory_masks=None, attention_factors=None, attention_masks=None, training=None):
        hidden_states, attention_scores = self.attention(input_states,
                                                         memory_states,
                                                         memory_states,
                                                         key_weights=memory_weights,
                                                         key_masks=memory_masks,
                                                         attention_factors=attention_factors,
                                                         attention_masks=attention_masks
                                                         )
        hidden_states = self.linear(hidden_states)
        if self.dropout:
            hidden_states = self.dropout(hidden_states, training=training)
        output_states = self.norm(hidden_states + input_states)
        return output_states, attention_scores


class RPEAttentionLayer(Layer):
    """
    用于实现带有相对位置编码的多头注意力机制。
    该层利用了先前定义的 RPEMultiHeadAttention 类，并在此基础上添加了线性层、丢弃层和层归一化，以完成一个完整的注意力模块
    """

    def __init__(self, d_model, num_heads, dropout=None, name=None):
        super(RPEAttentionLayer, self).__init__(name=name)
        self.attention = RPEMultiHeadAttention(d_model, num_heads, dropout=dropout, name=name+"_multihead_attention")
        self.linear = KL.Dense(d_model, name=name+"_dense")
        self.dropout = KL.Dropout(dropout) if dropout else None
        self.norm = KL.LayerNormalization(epsilon=1e-6, name=name+"_layer_norm")

    def call(self, input_states, memory_states, position_states, memory_weights=None,
             memory_masks=None, attention_factors=None, training=None):
        hidden_states, attention_scores = self.attention(input_states,
                                                         memory_states,
                                                         memory_states,
                                                         position_states,
                                                         key_weights=memory_weights,
                                                         key_masks=memory_masks,
                                                         attention_factors=attention_factors)
        hidden_states = self.linear(hidden_states)
        if self.dropout:
            hidden_states = self.dropout(hidden_states, training=training)
        output_states = self.norm(hidden_states + input_states)
        return output_states, attention_scores


class LinearAttentionLayer(Layer):
    def __init__(self, d_model, num_heads, dropout=False, focusing_factor=3, name=None):
        super(LinearAttentionLayer, self).__init__(name=name)
        self.attention = LinearAttention(d_model, num_heads, focusing_factor, name=name+"_attention")
        self.linear = KL.Dense(d_model, name=name+"_dense")
        self.dropout = KL.Dropout(dropout) if dropout else None
        self.norm = KL.LayerNormalization(epsilon=1e-6, name=name+"_layer_norm")

    def call(self, input_states, memory_states, training=None):
        hidden_states = self.attention(input_states, memory_states, memory_states)
        hidden_states = self.linear(hidden_states)
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states, training=training)
        output_states = self.norm(hidden_states + input_states)
        return output_states


class AttentionOutput(Layer):
    def __init__(self, d_model, dropout=None, activation_fn='ReLU', name=None):
        super(AttentionOutput, self).__init__(name=name)
        self.expand = KL.Dense(d_model * 2, name=name+"_expand")
        self.activation = KL.Activation(activation_fn, name=name+"_activation")
        self.squeeze = KL.Dense(d_model, name=name+"_squeeze")
        self.dropout = KL.Dropout(rate=dropout) if dropout else None
        self.norm = KL.LayerNormalization(epsilon=1e-6,name=name+"_layer_norm")

    def call(self, input_states, training=None):
        hidden_states = self.expand(input_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.squeeze(hidden_states)
        if self.dropout:
            hidden_states = self.dropout(hidden_states, training=training)
        output_states = self.norm(input_states + hidden_states)
        return output_states


class TransformerLayer(Layer):
    def __init__(self, d_model, num_heads, dropout=None, activation_fn='ReLU', name=None):
        super(TransformerLayer, self).__init__(name=name)
        self.attention = AttentionLayer(d_model, num_heads, dropout=dropout, name=name+"_attention")
        self.attn_output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn,
                                           name=name+"_attention_output")

    def call(self, input_states, memory_states, memory_weights=None, memory_masks=None, attention_factors=None,
             attention_masks=None):
        hidden_states, attention_scores = self.attention(input_states,
                                                         memory_states,
                                                         memory_weights=memory_weights,
                                                         memory_masks=memory_masks,
                                                         attention_factors=attention_factors,
                                                         attention_masks=attention_masks
                                                         )
        output_states = self.attn_output(hidden_states)
        return output_states, attention_scores


class RPETransformerLayer(Layer):
    def __init__(self, d_model, num_heads, dropout=None, activation_fn='ReLU', name=None):
        super(RPETransformerLayer, self).__init__(name=name)
        self.attention = RPEAttentionLayer(d_model, num_heads, dropout=dropout, name=name+"_attention_layer")
        self.attn_output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn,
                                           name=name+"_attention_output")

    def call(self, input_states, memory_states, position_states, memory_weights=None, memory_masks=None,
             attention_factors=None):
        hidden_states, attention_scores = self.attention(input_states=input_states,
                                                         memory_states=memory_states,
                                                         position_states=position_states,
                                                         memory_weights=memory_weights,
                                                         memory_masks=memory_masks,
                                                         attention_factors=attention_factors
                                                         )
        output_states = self.attn_output(hidden_states)
        return output_states, attention_scores


class LinearTransformerLayer(Layer):
    def __init__(self, d_model, num_heads, dropout=None, activation_fn='ReLU', focusing_factor=3,name=None):
        super(LinearTransformerLayer, self).__init__(name=name)
        self.attention = LinearAttentionLayer(d_model, num_heads, dropout=dropout, focusing_factor=focusing_factor,
                                              name=name+"_attention")
        self.attn_output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn,
                                           name=name+"_attention_output")

    def call(self, input_states, memory_states):
        hidden_states = self.attention(input_states, memory_states)
        output_states = self.attn_output(hidden_states)
        return output_states


class GeometricTransformer(Layer):
    def __init__(
            self,
            blocks,
            d_model,
            num_heads,
            dropout=None,
            activation_fn='ReLU',
            return_attention_scores=False,
            parallel=False,
            name=None
    ):
        super(GeometricTransformer, self).__init__(name=name)
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(RPETransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn,
                                                  name=name+"_rpe"))
            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn,
                                               name=name+"_transformer"))
        self.layers = layers
        self.return_attention_scores = return_attention_scores
        self.parallel = parallel

    def call(self, feats0, embeddings0, feats1, embeddings1, masks0=None, masks1=None):
        """
        Args:
            feats0, feats1: 输入特征
            embeddings0, embeddings1: 与特征相关的位置嵌入
            masks0, masks1: 可选的遮罩，用于影响注意力分数
        """
        attention_scores = []
        for i, block in enumerate(self.blocks):
            layer = self.layers[i]
            if block == 'self':
                feats0, scores0 = layer(feats0, feats0, embeddings0, memory_masks=masks0)
                feats1, scores1 = layer(feats1, feats1, embeddings1, memory_masks=masks1)
            else:
                if self.parallel:
                    new_feats0, scores0 = layer(feats0, feats1, memory_masks=masks1)
                    new_feats1, scores1 = layer(feats1, feats0, memory_masks=masks0)
                    feats0 = new_feats0
                    feats1 = new_feats1
                else:
                    feats0, scores0 = layer(feats0, feats1, memory_masks=masks1)
                    feats1, scores1 = layer(feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1


class SparseToDenseTransformer(tf.keras.layers.Layer):
    """
    对输入特征进行采样，获取背景 token（如果存在）和通过远点采样索引获得的特征。
    使用 sparse_layer 处理这些特征和相应的嵌入。
    使用 dense_layer 将处理过的特征整合回原始的密集特征表示
    """

    def __init__(self, d_model, sparse_blocks, num_heads=4, dropout=None, activation='relu',
                 parallel=False, focusing_factor=3, with_bg_token=True, replace_bg_token=True, name=None):
        super().__init__(name=name)
        self.with_bg_token = with_bg_token
        self.replace_bg_token = replace_bg_token
        self.sparse_layer = GeometricTransformer(blocks=sparse_blocks, d_model=d_model, num_heads=num_heads,
                                                 dropout=dropout, activation_fn=activation, parallel=parallel,
                                                 return_attention_scores=False,name=name+"_geo_transformer")
        self.dense_layer = LinearTransformerLayer(d_model, num_heads, focusing_factor=focusing_factor,
                                                  name=name+"_linear_transformer")

    def call(self, dense_feats0, embeddings0, fps_idx0, dense_feats1, embeddings1, fps_idx1, masks0=None, masks1=None):
        feats0 = self._sample_feats(dense_feats0, fps_idx0)
        feats1 = self._sample_feats(dense_feats1, fps_idx1)
        feats0, feats1 = self.sparse_layer(feats0, embeddings0, feats1, embeddings1, masks0, masks1)

        dense_feats0 = self._get_dense_feats(dense_feats0, feats0)
        dense_feats1 = self._get_dense_feats(dense_feats1, feats1)
        return dense_feats0, dense_feats1

    def _sample_feats(self, dense_feats, fps_idx):
        if self.with_bg_token:
            bg_token = dense_feats[:, :1, :] if self.with_bg_token else None
        feats = tf.gather(dense_feats, fps_idx, batch_dims=1)

        if self.with_bg_token:
            feats = tf.concat([bg_token, feats], axis=1)

        return feats

    def _get_dense_feats(self, dense_feats, feats):
        if self.with_bg_token and self.replace_bg_token:
            bg_token = feats[:, 0:1, :]
            feats = self.dense_layer(dense_feats[:, 1:, :], feats[:, 1:, :])
            return tf.concat([bg_token, feats], axis=1)
        else:
            return self.dense_layer(dense_feats, feats)


class Transformer(Layer):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0,**kwargs):
        super(Transformer, self).__init__(**kwargs)

        self.layers = []

        for idx in range(depth):
            self.layers.append([
                PreNorm(Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout,
                                  name="ViT_attention_{}".format(idx)),
                        name="ViT_PreNorm_Attention_{}".format(idx)),
                PreNorm(MLP(dim, mlp_dim, dropout=dropout, name="ViT_mlp_{}".format(idx)),
                        name="ViT_PreNorm_mlp_{}".format(idx)),
            ])

    def call(self, x, training=None):
        for attn, mlp in self.layers:
            x = attn(x) + x
            x = mlp(x) + x

        return x


class ViT(Layer):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool='cls', dim_head=64, dropout=0.0, emb_dropout=0.0, **kwargs):
        """
            image_size: int.
            -> Image size. If you have rectangular images, make sure your image size is the maximum of the width and height
            patch_size: int.
            -> Number of patches. image_size must be divisible by patch_size.
            -> The number of patches is: n = (image_size // patch_size) ** 2 and n must be greater than 16.
            num_classes: int.
            -> Number of classes to classify.
            dim: int.
            -> Last dimension of attn_output tensor after linear transformation KL.Linear(..., dim).
            depth: int.
            -> Number of Transformer blocks.
            heads: int.
            -> Number of heads in Multi-head Attention layer.
            mlp_dim: int.
            -> Dimension of the MLP (FeedForward) layer.
            dropout: float between [0, 1], default 0..
            -> Dropout rate.
            emb_dropout: float between [0, 1], default 0.
            -> Embedding dropout rate.
            pool: string, either cls token pooling or mean pooling
        """
        super(ViT, self).__init__(**kwargs)

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_embedding = Sequential([
            Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            KL.Dense(units=dim, name="ViT_patch_embedding_dense"),
        ], name='ViT_patch_embedding')

        self.pos_embedding = tf.Variable(initial_value=tf.random.normal([1, num_patches + 1, dim]), name="ViT_pos_embedding")
        self.cls_token = tf.Variable(initial_value=tf.random.normal([1, 1, dim]), name="ViT_cls_token")
        self.dropout = KL.Dropout(rate=emb_dropout, name="ViT_dropout")

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, name="ViT_transformer")

        self.pool = pool

        if num_classes is not None:
            self.mlp_head = Sequential([
                KL.LayerNormalization(name="ViT_mlp_head_layer_norm"),
                KL.Dense(units=num_classes, name="ViT_mlp_head_dense")
            ], name='ViT_mlp_head')

    def call(self, img, training=None, **kwargs):
        x = self.patch_embedding(img)
        b, n, d = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = tf.concat([cls_tokens, x], axis=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x, training=training)

        x = self.transformer(x, training=training)

        if self.pool == 'mean':
            x = tf.reduce_mean(x, axis=1)
        else:
            x = x[:, 0]

        x = self.mlp_head(x)

        return x


class ViTFpn(ViT):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim,
                 pool='cls', dim_head=64, dropout=0.0, emb_dropout=0.0, **kwargs):
        super(ViTFpn, self).__init__(image_size, patch_size, None, dim, depth, heads, mlp_dim,
                                     pool=pool, dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout, **kwargs)

    @tf.function
    def call(self, x, **kwargs):
        x = self.patch_embedding(x)
        shape = tf.shape(x)
        b, n, d = shape[0], shape[1], shape[2]

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = tf.concat([cls_tokens, x], axis=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        out = []
        d = len(self.transformer.layers)
        n = d // 4
        idx_nblock = [d - 1, d - n - 1, d - 2 * n - 1, d - 3 * n - 1]

        for idx, (attn, mlp) in enumerate(self.transformer.layers):
            x = attn(x) + x
            x = mlp(x) + x
            if idx in idx_nblock:
                out.append(x)

        return out


class ViTAE(Layer):
    """
    input: [B, h, w, C]
    attn_output: [B, h, w, out_dim]
    """

    def __init__(self, config, **kwargs):
        super(ViTAE, self).__init__(**kwargs)
        self.config = config
        self.vit = ViTFpn(
            image_size=config['image_size'],
            patch_size=config['patch_size'],
            dim=config['embed_dim'],
            depth=config['depth'],
            heads=config['heads'],
            mlp_dim=config['mlp_dim'],
            pool=config['pool'],
            dim_head=config['dim_head'],
            dropout=config['dropout'],
            emb_dropout=config['emb_dropout'],
            name="ViT_FPN"
        )
        self.len_patches = config['image_size'] // config['patch_size']
        self.use_pyramid_feat = config['use_pyramid_feat']
        self.up_type = config['up_type']
        self.out_dim = config['out_dim']

        if self.up_type == 'linear':
            self.output_upscaling = KL.Dense(16 * self.out_dim, use_bias=True, activation='linear', name="ViTAE_upscaling")
        elif self.up_type == 'deconv':
            self.output_upscaling = tf.keras.Sequential([
                KL.Conv2DTranspose(self.out_dim * 2, kernel_size=2, strides=2, name="ViTAE_conv2d_transpose_1"),
                KL.LayerNormalization(epsilon=1e-6, name="ViTAE_layer_norm"),
                tf.keras.activations.gelu,
                KL.Conv2DTranspose(self.out_dim, kernel_size=2, strides=2, name="ViTAE_conv2d_transpose_2"),
            ])
        else:
            raise ValueError("Unsupported upscaling type")

    @tf.function
    def call(self, inputs):
        input_shape = tf.shape(inputs)
        h = input_shape[1]
        w = input_shape[2]
        size = tf.stack([h, w])
        vit_outs = self.vit(inputs)  # vit_outs = [B, num_patches+1, embed_dim]
        cls_tokens = vit_outs[-1][:, 0, :]  # cls_tokens = [B, embed_dim]
        vit_outs = [l[:, 1:, :] for l in vit_outs]  # vit_outs = [B, num_patches, embed_dim]

        if self.use_pyramid_feat:
            x = KL.Concatenate(axis=-1)(vit_outs)  # vit_outs_concat = [B, num_patches, embed_dim*4]
        else:
            x = vit_outs[-1]

        if self.up_type == 'linear':
            x = self.output_upscaling(x)  # vit_upscale = [B, num_patches, out_dim * 16]
            x = KL.Reshape((self.len_patches, self.len_patches, 4, 4, self.out_dim))(x)
            x = KL.Permute((1, 3, 2, 4, 5))(x)
            x = KL.Reshape((self.len_patches * 4, self.len_patches * 4, -1))(x)
            x = tf.image.resize(x, size)  # vit_upscale = [B, h, w, out_dim]
        elif self.up_type == 'deconv':
            x = KL.Permute((2, 1))(x)
            x = KL.Reshape((16, 16, -1))(x)
            x = self.output_upscaling(x)
            x = tf.image.resize(x, size)

        return x, cls_tokens


@keras.utils.register_keras_serializable()
class ViTEncoder(Layer):
    def __init__(self, config, npoint=2048, **kwargs):
        super(ViTEncoder, self).__init__(**kwargs)
        self.npoint = npoint
        self.config = config
        self.rgb_net = ViTAE(self.config, name="ViTAE_rgb_net")

    def call(self, rgb, rgb_choose, pts, tem1_rgb, tem1_choose, tem1_pts, tem2_rgb, tem2_choose, tem2_pts):
        dense_fm = self.get_img_feats(rgb, rgb_choose)
        dense_pm = pts
        dense_po = tf.concat([tem1_pts, tem2_pts], axis=1)
        radius = tf.reduce_max(tf.norm(dense_po, axis=2), axis=1)
        dense_pm = dense_pm / (tf.reshape(radius, (-1, 1, 1)) + 1e-6)
        tem1_pts = tem1_pts / (tf.reshape(radius, (-1, 1, 1)) + 1e-6)
        tem2_pts = tem2_pts / (tf.reshape(radius, (-1, 1, 1)) + 1e-6)

        dense_po, dense_fo = self.get_obj_feats(
            [tem1_rgb, tem2_rgb],
            [tem1_pts, tem2_pts],
            [tem1_choose, tem2_choose]
        )

        dense_pm = tf.cast(dense_pm, tf.float32)
        dense_fm = tf.cast(dense_fm, tf.float32)
        dense_po = tf.cast(dense_po, tf.float32)
        dense_fo = tf.cast(dense_fo, tf.float32)

        return dense_pm, dense_fm, dense_po, dense_fo, radius

    def get_img_feats(self, img, choose):
        return get_chosen_pixel_feats(self.rgb_net(img)[0], choose)

    def get_obj_feats(self, tem_rgb_list, tem_pts_list, tem_choose_list, npoint=None):
        if npoint is None:
            npoint = self.npoint

        tem_feat_list = []
        for tem, tem_choose in zip(tem_rgb_list, tem_choose_list):
            tem_feat_list.append(self.get_img_feats(tem, tem_choose))

        tem_pts = tf.concat(tem_pts_list, axis=1)
        tem_feat = tf.concat(tem_feat_list, axis=1)

        return sample_pts_feats(tem_pts, tem_feat, npoint)

    def get_config(self):
        config = super(ViTEncoder, self).get_config()
        config.update({
            'npoint': self.npoint,
            'config': self.config
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
