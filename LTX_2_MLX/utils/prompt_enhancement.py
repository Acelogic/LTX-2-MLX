"""Prompt enhancement utilities for LTX-2 MLX.

Uses Gemma 3 to enhance user prompts for better video generation.
"""

import re
from typing import Optional

import mlx.core as mx
import numpy as np
from PIL import Image


# Default system prompts for prompt enhancement
T2V_SYSTEM_PROMPT = """You are a highly skilled video production expert tasked with transforming simple user prompts into rich, cinematic video descriptions. Your goal is to take the user's basic idea and expand it into a detailed, visually compelling description that would guide a state-of-the-art AI video generator.

When enhancing prompts, consider:
1. Camera work: angles, movements (pan, tilt, zoom, tracking shots)
2. Lighting: natural, artificial, dramatic, soft, golden hour
3. Environment and setting details
4. Subject actions and movements
5. Color palette and visual mood
6. Temporal progression within the scene

Keep your response focused on visual description only. Do not include dialogue, sound effects, or music descriptions. Output only the enhanced prompt, nothing else."""

I2V_SYSTEM_PROMPT = """You are a highly skilled video production expert. Given an image and a user prompt, create a detailed video description that:
1. Accurately describes the key visual elements in the image
2. Incorporates the user's requested action or scene development
3. Adds cinematic details like camera movement, lighting changes, and temporal progression

Describe the video as a continuous scene starting from the provided image. Focus on visual elements only - no dialogue, sound effects, or music. Output only the enhanced prompt, nothing else."""


def clean_response(response: str) -> str:
    """
    Clean up a generated response.

    - Remove curly quotes and replace with straight quotes
    - Remove leading characters like dashes, asterisks, colons
    - Strip whitespace
    """
    # Replace curly quotes with straight quotes
    response = response.replace(""", '"').replace(""", '"')
    response = response.replace("'", "'").replace("'", "'")

    # Remove common leading characters
    response = response.lstrip("-*:> ")

    # Strip whitespace
    response = response.strip()

    return response


def resize_aspect_ratio_preserving(
    image: np.ndarray,
    long_side: int,
) -> np.ndarray:
    """Resize image preserving aspect ratio, scaling the long side."""
    h, w = image.shape[:2]
    if h > w:
        new_h = long_side
        new_w = int(w * long_side / h)
    else:
        new_w = long_side
        new_h = int(h * long_side / w)

    # Use PIL for resizing
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.array(pil_image)


def load_image_for_enhancement(
    image_path: str,
    long_side: int = 896,
) -> np.ndarray:
    """Load an image and resize for prompt enhancement."""
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    return resize_aspect_ratio_preserving(image_np, long_side)


def create_t2v_chat_prompt(
    user_prompt: str,
    system_prompt: Optional[str] = None,
) -> str:
    """Create a chat prompt for T2V prompt enhancement."""
    system_prompt = system_prompt or T2V_SYSTEM_PROMPT
    # Gemma 3 chat format
    chat = (
        f"<bos><start_of_turn>user\n"
        f"{system_prompt}\n\n"
        f"User prompt: {user_prompt}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    return chat


def create_i2v_chat_prompt(
    user_prompt: str,
    system_prompt: Optional[str] = None,
) -> str:
    """Create a chat prompt for I2V prompt enhancement (text only, image handled separately)."""
    system_prompt = system_prompt or I2V_SYSTEM_PROMPT
    # Gemma 3 chat format
    chat = (
        f"<bos><start_of_turn>user\n"
        f"{system_prompt}\n\n"
        f"[Image provided]\n"
        f"User prompt: {user_prompt}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    return chat


def enhance_prompt_t2v(
    user_prompt: str,
    gemma_model,
    tokenizer,
    max_new_tokens: int = 256,
    seed: int = 42,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Enhance a text-to-video prompt using Gemma.

    Args:
        user_prompt: User's original prompt.
        gemma_model: Loaded Gemma 3 model.
        tokenizer: Gemma tokenizer.
        max_new_tokens: Maximum tokens to generate.
        seed: Random seed for generation.
        system_prompt: Optional custom system prompt.

    Returns:
        Enhanced prompt string.
    """
    mx.random.seed(seed)

    # Create chat prompt
    chat_prompt = create_t2v_chat_prompt(user_prompt, system_prompt)

    # Tokenize
    encoding = tokenizer(
        chat_prompt,
        return_tensors="np",
        padding=False,
        truncation=True,
        max_length=2048,
    )
    input_ids = mx.array(encoding["input_ids"])

    # Generate with sampling
    generated = generate_text(
        gemma_model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
    )

    # Decode
    generated_ids = generated[0].tolist()
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Extract only the generated part (after model turn)
    if "<start_of_turn>model" in response:
        response = response.split("<start_of_turn>model")[-1]

    return clean_response(response)


def enhance_prompt_i2v(
    user_prompt: str,
    image_path: str,
    gemma_model,
    tokenizer,
    max_new_tokens: int = 256,
    seed: int = 42,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Enhance an image-to-video prompt using Gemma.

    Note: This version creates a text-based enhancement since Gemma 3 text-only
    models don't support image input. For full I2V enhancement with vision,
    use a vision-language model variant.

    Args:
        user_prompt: User's original prompt.
        image_path: Path to the conditioning image.
        gemma_model: Loaded Gemma 3 model.
        tokenizer: Gemma tokenizer.
        max_new_tokens: Maximum tokens to generate.
        seed: Random seed for generation.
        system_prompt: Optional custom system prompt.

    Returns:
        Enhanced prompt string.
    """
    mx.random.seed(seed)

    # Create chat prompt (text-only version)
    chat_prompt = create_i2v_chat_prompt(user_prompt, system_prompt)

    # Tokenize
    encoding = tokenizer(
        chat_prompt,
        return_tensors="np",
        padding=False,
        truncation=True,
        max_length=2048,
    )
    input_ids = mx.array(encoding["input_ids"])

    # Generate
    generated = generate_text(
        gemma_model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
    )

    # Decode
    generated_ids = generated[0].tolist()
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Extract only the generated part
    if "<start_of_turn>model" in response:
        response = response.split("<start_of_turn>model")[-1]

    return clean_response(response)


def generate_text(
    model,
    input_ids: mx.array,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    eos_token_id: int = 107,  # Gemma end-of-turn token
) -> mx.array:
    """
    Generate text using autoregressive sampling.

    Args:
        model: Gemma model with __call__ method.
        input_ids: Input token IDs [1, seq_len].
        max_new_tokens: Maximum new tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        eos_token_id: End of sequence token ID.

    Returns:
        Generated token IDs [1, seq_len + new_tokens].
    """
    generated = input_ids

    for _ in range(max_new_tokens):
        # Get logits for next token
        # Model forward: returns (logits, hidden_states) or just logits
        output = model(generated)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        # Get last token logits
        next_token_logits = logits[:, -1, :]

        # Apply temperature
        if temperature > 0:
            next_token_logits = next_token_logits / temperature

        # Sample with top-p (nucleus sampling)
        next_token = sample_top_p(next_token_logits, top_p)

        # Append to sequence
        generated = mx.concatenate([generated, next_token[:, None]], axis=1)
        mx.eval(generated)

        # Check for EOS
        if int(next_token[0]) == eos_token_id:
            break

    return generated


def sample_top_p(logits: mx.array, p: float) -> mx.array:
    """
    Sample from logits using nucleus (top-p) sampling.

    Args:
        logits: Logits tensor [batch, vocab].
        p: Probability threshold for nucleus sampling.

    Returns:
        Sampled token IDs [batch].
    """
    # Convert to probabilities
    probs = mx.softmax(logits, axis=-1)

    # Sort probabilities descending
    sorted_indices = mx.argsort(-probs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

    # Compute cumulative probabilities
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    # Create mask for tokens to keep (cumsum <= p, plus first token always)
    # Shift by 1 to include the token that crosses the threshold
    mask = cumulative_probs - sorted_probs <= p

    # Zero out probabilities for tokens outside nucleus
    sorted_probs = mx.where(mask, sorted_probs, mx.zeros_like(sorted_probs))

    # Renormalize
    sorted_probs = sorted_probs / mx.sum(sorted_probs, axis=-1, keepdims=True)

    # Sample from sorted distribution
    # Use gumbel-max trick for sampling
    u = mx.random.uniform(shape=sorted_probs.shape)
    gumbel = -mx.log(-mx.log(u + 1e-10) + 1e-10)
    scores = mx.log(sorted_probs + 1e-10) + gumbel
    sampled_idx = mx.argmax(scores, axis=-1)

    # Map back to original indices
    batch_size = logits.shape[0]
    sampled_tokens = mx.array([
        sorted_indices[b, sampled_idx[b]] for b in range(batch_size)
    ])

    return sampled_tokens


def generate_enhanced_prompt(
    gemma_model,
    tokenizer,
    prompt: str,
    image_path: Optional[str] = None,
    max_new_tokens: int = 256,
    seed: int = 42,
) -> str:
    """
    Generate an enhanced prompt using Gemma.

    This is the main entry point for prompt enhancement.

    Args:
        gemma_model: Loaded Gemma 3 model.
        tokenizer: Gemma tokenizer.
        prompt: User's original prompt.
        image_path: Optional path to conditioning image (for I2V).
        max_new_tokens: Maximum tokens to generate.
        seed: Random seed.

    Returns:
        Enhanced prompt string.
    """
    if image_path:
        return enhance_prompt_i2v(
            user_prompt=prompt,
            image_path=image_path,
            gemma_model=gemma_model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
    else:
        return enhance_prompt_t2v(
            user_prompt=prompt,
            gemma_model=gemma_model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
