"""Tests for LTX-2 MLX Transformer components."""

import mlx.core as mx


def test_rope():
    """Test Rotary Position Embedding components."""
    from LTX_2_MLX.model.transformer import (
        precompute_freqs_cis,
        apply_rotary_emb,
        create_position_grid,
        LTXRopeType,
    )

    # Test position grid creation
    grid = create_position_grid(batch_size=1, frames=4, height=8, width=8)
    num_tokens = 4 * 8 * 8  # F * H * W
    assert grid.shape == (1, 3, num_tokens), f"Got {grid.shape}"
    print("  create_position_grid: OK")

    # Test freqs_cis precomputation with position grid
    dim = 128
    freqs = precompute_freqs_cis(
        indices_grid=grid,
        dim=dim,
        theta=10000.0,
        num_attention_heads=4,
        rope_type=LTXRopeType.INTERLEAVED,
    )
    cos_freqs, sin_freqs = freqs
    # For interleaved, output is (B, T, dim)
    assert cos_freqs.shape == (1, num_tokens, dim), f"Got cos shape {cos_freqs.shape}"
    assert sin_freqs.shape == (1, num_tokens, dim), f"Got sin shape {sin_freqs.shape}"
    print("  precompute_freqs_cis: OK")

    # Test rotary embedding application
    # Input for interleaved: (..., dim)
    x = mx.random.normal(shape=(1, num_tokens, dim))
    x_rotated = apply_rotary_emb(x, freqs, LTXRopeType.INTERLEAVED)
    assert x_rotated.shape == x.shape, f"Expected {x.shape}, got {x_rotated.shape}"
    print("  apply_rotary_emb: OK")


def test_timestep_embedding():
    """Test timestep embedding components."""
    from LTX_2_MLX.model.transformer import (
        get_timestep_embedding,
        Timesteps,
        TimestepEmbedding,
    )

    # Test basic timestep embedding
    timesteps = mx.array([0.0, 0.5, 1.0])
    emb = get_timestep_embedding(timesteps, embedding_dim=256)
    assert emb.shape == (3, 256), f"Expected (3, 256), got {emb.shape}"
    print("  get_timestep_embedding: OK")

    # Test Timesteps module
    timesteps_module = Timesteps(num_channels=256)
    emb = timesteps_module(mx.array([0.5]))
    assert emb.shape == (1, 256), f"Expected (1, 256), got {emb.shape}"
    print("  Timesteps: OK")

    # Test TimestepEmbedding
    ts_embed = TimestepEmbedding(in_channels=256, time_embed_dim=512)
    x = mx.random.normal(shape=(1, 256))
    out = ts_embed(x)
    assert out.shape == (1, 512), f"Expected (1, 512), got {out.shape}"
    print("  TimestepEmbedding: OK")


def test_rms_norm():
    """Test RMSNorm."""
    from LTX_2_MLX.model.transformer import RMSNorm, rms_norm

    # Test functional rms_norm
    x = mx.random.normal(shape=(2, 10, 64))
    weight = mx.ones((64,))
    normed = rms_norm(x, weight)
    assert normed.shape == x.shape
    print("  rms_norm (functional): OK")

    # Test RMSNorm module
    norm = RMSNorm(dims=64)
    normed = norm(x)
    assert normed.shape == x.shape
    print("  RMSNorm (module): OK")


def test_feed_forward():
    """Test FeedForward network."""
    from LTX_2_MLX.model.transformer import FeedForward

    ff = FeedForward(dim=256, dim_out=256, mult=4)  # hidden = dim * mult = 1024

    x = mx.random.normal(shape=(1, 100, 256))
    out = ff(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print("  FeedForward: OK")


def test_attention():
    """Test Attention modules."""
    from LTX_2_MLX.model.transformer import Attention, SelfAttention, CrossAttention

    # Test basic Attention (self-attention mode)
    attn = Attention(
        query_dim=256,
        heads=4,
        dim_head=64,
    )
    q = mx.random.normal(shape=(1, 100, 256))
    out = attn(q)  # context=None means self-attention
    assert out.shape == q.shape, f"Expected {q.shape}, got {out.shape}"
    print("  Attention (self): OK")

    # Test SelfAttention
    self_attn = SelfAttention(
        dim=256,  # SelfAttention takes 'dim' not 'query_dim'
        heads=4,
        dim_head=64,
    )
    out = self_attn(q)
    assert out.shape == q.shape, f"Expected {q.shape}, got {out.shape}"
    print("  SelfAttention: OK")

    # Test CrossAttention
    cross_attn = CrossAttention(
        query_dim=256,
        context_dim=512,  # CrossAttention uses 'context_dim' not 'kv_dim'
        heads=4,
        dim_head=64,
    )
    context = mx.random.normal(shape=(1, 50, 512))
    out = cross_attn(q, context)
    assert out.shape == q.shape, f"Expected {q.shape}, got {out.shape}"
    print("  CrossAttention: OK")


def test_adaln():
    """Test Adaptive Layer Norm."""
    from LTX_2_MLX.model.transformer import AdaLayerNormSingle

    # AdaLN with timestep conditioning
    # num_embeddings=6 means: scale, shift, gate for self-attn and cross-attn (or FFN)
    adaln = AdaLayerNormSingle(embedding_dim=512, num_embeddings=6)

    # Timestep values (not embedding)
    timestep = mx.array([0.5])

    # Get AdaLN parameters
    out = adaln(timestep)
    # Output shape: (batch, num_embeddings * embedding_dim)
    assert out.shape == (1, 6 * 512), f"Expected (1, 3072), got {out.shape}"
    print("  AdaLayerNormSingle: OK")


def test_transformer_block():
    """Test BasicTransformerBlock."""
    from LTX_2_MLX.model.transformer import (
        BasicTransformerBlock,
        TransformerArgs,
        precompute_freqs_cis,
        create_position_grid,
    )

    # Create block with matching dimensions
    dim = 256
    num_heads = 4
    head_dim = 64  # inner_dim = num_heads * head_dim = 256
    context_dim = 512

    block = BasicTransformerBlock(
        dim=dim,
        num_heads=num_heads,
        head_dim=head_dim,
        context_dim=context_dim,
    )

    # Create inputs
    seq_len = 64
    hidden_states = mx.random.normal(shape=(1, seq_len, dim))
    context = mx.random.normal(shape=(1, 20, context_dim))

    # Create position embeddings
    grid = create_position_grid(batch_size=1, frames=4, height=4, width=4)  # 64 tokens
    freqs_cis = precompute_freqs_cis(
        indices_grid=grid,
        dim=dim,
        num_attention_heads=num_heads,
    )

    # Timestep embedding shape: (B, T, 6, dim)
    # 6 = scale, shift, gate for self-attn + scale, shift, gate for FFN
    timesteps = mx.random.normal(shape=(1, seq_len, 6, dim))

    # Create TransformerArgs
    args = TransformerArgs(
        x=hidden_states,
        context=context,
        timesteps=timesteps,
        positional_embeddings=freqs_cis,
    )

    out = block(args)
    assert out.x.shape == hidden_states.shape, f"Expected {hidden_states.shape}, got {out.x.shape}"
    print("  BasicTransformerBlock: OK")


def test_ltx_model_minimal():
    """Test LTXModel with minimal configuration."""
    from LTX_2_MLX.model.transformer import LTXModel, LTXModelType, Modality, create_position_grid

    # Create a minimal model (fewer layers for testing)
    # inner_dim = num_heads * head_dim must equal cross_attention_dim
    num_heads = 4
    head_dim = 64
    inner_dim = num_heads * head_dim  # 256
    caption_channels = 128  # Input caption dimension (like Gemma output)

    model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        in_channels=128,
        out_channels=128,
        num_layers=2,  # Minimal for testing
        cross_attention_dim=inner_dim,  # Must equal num_heads * head_dim
        caption_channels=caption_channels,
    )

    # Create input modality
    batch_size = 1
    frames, height, width = 2, 4, 4  # Small grid for testing
    seq_len = frames * height * width  # 32 tokens

    latent = mx.random.normal(shape=(batch_size, seq_len, 128))
    context = mx.random.normal(shape=(batch_size, 20, caption_channels))  # 20 text tokens
    context_mask = mx.ones((batch_size, 20))
    timestep = mx.array([0.5])

    # Create position grid: (B, 3, T, 2) - with start/end bounds
    # The model expects positions with shape (B, n_dims, T, 2)
    grid = create_position_grid(batch_size, frames, height, width)  # (B, 3, T)
    # Add bounds dimension - each position has (start, end) = (pos, pos+1)
    grid_start = grid[..., None]  # (B, 3, T, 1)
    grid_end = grid_start + 1
    positions = mx.concatenate([grid_start, grid_end], axis=-1)  # (B, 3, T, 2)

    modality = Modality(
        latent=latent,
        context=context,
        context_mask=context_mask,
        timesteps=timestep,
        positions=positions,
        enabled=True,
    )

    # Forward pass
    output = model(modality)
    assert output.shape == latent.shape, f"Expected {latent.shape}, got {output.shape}"
    print("  LTXModel forward: OK")


def run_transformer_tests():
    """Run all transformer tests."""
    print("\n=== Transformer Tests ===\n")

    print("Testing RoPE...")
    test_rope()

    print("\nTesting timestep embeddings...")
    test_timestep_embedding()

    print("\nTesting RMSNorm...")
    test_rms_norm()

    print("\nTesting FeedForward...")
    test_feed_forward()

    print("\nTesting Attention...")
    test_attention()

    print("\nTesting AdaLN...")
    test_adaln()

    print("\nTesting TransformerBlock...")
    test_transformer_block()

    print("\nTesting LTXModel (minimal)...")
    test_ltx_model_minimal()

    print("\n=== All Transformer Tests Passed! ===\n")


if __name__ == "__main__":
    run_transformer_tests()
