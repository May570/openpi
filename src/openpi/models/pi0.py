import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

from jax.experimental import io_callback
import openpi.shared.rtc_store as rtc_store

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        self.action_horizon = 50
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
    
    def rtc_sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        first_call: bool,
        last_actions: jax.Array,
        count: int,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        guide_beta: float = 2.0,
    ) -> tuple[_model.Actions, bool, jax.Array, int]:
        infer_delay_length = 8
        sleep_delay_length = 8
        self.action_horizon = 50
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        rtc_mask = make_rtc_mask((batch_size, self.action_horizon, self.action_dim), total=infer_delay_length)

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        if last_actions is None:
            last_actions = jnp.zeros((batch_size, self.action_horizon, self.action_dim))

        def step_noguide(carry):
            x_t, time, first_call = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], 
                mask=full_attn_mask, 
                positions=positions, 
                kv_cache=kv_cache, 
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt, first_call

        def step_guided(carry):
            x_t, time, first_call = carry
                    
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_ = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_, suffix_attn_mask], axis=-1)
            positions_ = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], 
                mask=full_attn_mask, 
                positions=positions_, 
                kv_cache=kv_cache, 
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            tau = time
            r2 = ((1.0 - tau) ** 2) / (tau ** 2 + (1.0 - tau) ** 2 + 1e-8)
            weight = jnp.minimum(guide_beta, ((1.0 - tau) / (tau + 1e-8)) * r2)
            A_hat = x_t + (1.0 - tau) * v_t
            def loss(x):
                return jnp.sum(rtc_mask * (A_hat - last_actions) ** 2) / (jnp.sum(rtc_mask) + 1e-8)
            g = jax.grad(loss)(x_t)
            v_t = v_t - weight * g

            return x_t + dt * v_t, time + dt, first_call

        def step(carry):
            return jax.lax.cond(
                carry[2],
                step_noguide,
                step_guided,
                carry
            )

        def cond(carry):
            x_t, time, first_call = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _, first_call = jax.lax.while_loop(cond, step, (noise, 1.0, first_call))

        def no_change(carry):
            first_call, last_actions = carry
            return False, last_actions

        def modify(carry):
            first_call, last_actions = carry
            def clip_and_pad(x):
                cut = x[:, sleep_delay_length:, :]
                zeros = jnp.zeros((cut.shape[0], sleep_delay_length, cut.shape[2]), dtype=cut.dtype)
                return jnp.concatenate([cut, zeros], axis=1)
            return first_call, clip_and_pad(last_actions)

        first_call_new, last_actions_new = jax.lax.cond(first_call, no_change, modify, (first_call, x_0))

        return x_0, first_call_new, last_actions_new, count

    def rtc_original(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        first_call: bool,
        last_actions: jax.Array,
        count: int,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        guide_beta: float = 2.0,
    ) -> tuple[_model.Actions, bool, jax.Array, int]:
        infer_delay_length = 3
        sleep_delay_length = 7
        self.action_horizon = 50
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        rtc_mask = make_rtc_mask((batch_size, self.action_horizon, self.action_dim), total=infer_delay_length)

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        if last_actions is None:
            last_actions = jnp.zeros((batch_size, self.action_horizon, self.action_dim))

        def step_noguide(carry):
            x_t, time, first_call = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], 
                mask=full_attn_mask, 
                positions=positions, 
                kv_cache=kv_cache, 
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt, first_call

        def step_guided(carry):
            x_t, time, first_call = carry

            def denoiser(x_t):
                suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                    observation, x_t, jnp.broadcast_to(time, batch_size)
                )
                suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
                prefix_attn_mask_ = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
                full_attn_mask = jnp.concatenate([prefix_attn_mask_, suffix_attn_mask], axis=-1)
                positions_ = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

                (_, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens], 
                    mask=full_attn_mask, 
                    positions=positions_, 
                    kv_cache=kv_cache, 
                    adarms_cond=[None, adarms_cond],
                )
                v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
                return x_t - time * v_t, v_t
            
            x_1, vjp_fun, v_t = jax.vjp(denoiser, x_t, has_aux=True)
            error = (last_actions - x_1) * rtc_mask
            pinv_correction = vjp_fun(error)[0]

            t_prime = 1.0 - time  # 把 1→0 的 time 映射到 0→1
            inv_r2 = (t_prime**2 + (1 - t_prime)**2) / ((1 - t_prime)**2 + 1e-8)
            c = (1 - t_prime) / (t_prime + 1e-8)
            guidance_weight = jnp.minimum(c * inv_r2, guide_beta)

            v_t = v_t + guidance_weight * pinv_correction

            return x_t + dt * v_t, time + dt, first_call

        def step(carry):
            return jax.lax.cond(
                carry[2],
                step_noguide,
                step_guided,
                carry
            )

        def cond(carry):
            x_t, time, first_call = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _, first_call = jax.lax.while_loop(cond, step, (noise, 1.0, first_call))

        def no_change(carry):
            first_call, last_actions = carry
            def clip_and_pad(x):
                cut = x[:, sleep_delay_length:, :]
                zeros = jnp.zeros((cut.shape[0], sleep_delay_length, cut.shape[2]), dtype=cut.dtype)
                return jnp.concatenate([cut, zeros], axis=1)
            return first_call, clip_and_pad(last_actions)
            # return False, last_actions

        def modify(carry):
            first_call, last_actions = carry
            def clip_and_pad(x):
                cut = x[:, infer_delay_length + sleep_delay_length:, :]
                zeros = jnp.zeros((cut.shape[0], infer_delay_length + sleep_delay_length, cut.shape[2]), dtype=cut.dtype)
                return jnp.concatenate([cut, zeros], axis=1)
            return first_call, clip_and_pad(last_actions)

        first_call_new, last_actions_new = jax.lax.cond(first_call, no_change, modify, (first_call, x_0))

        return x_0, first_call_new, last_actions_new, count
    
    def rtc_new_obs_sample_actions_1(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        first_call: bool,
        last_actions: jax.Array,
        count: int,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        guide_beta: float = 2.0,
    ) -> tuple[_model.Actions, bool, jax.Array, int]:
        infer_delay_length = 10
        sleep_delay_length = 0
        self.action_horizon = 60
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        rtc_mask = make_rtc_mask((batch_size, self.action_horizon, self.action_dim), total=infer_delay_length)
        # flatten 保存结构模板
        flat_obs, tree_def = jax.tree_util.tree_flatten(observation)
        shape_dtype_structs = tuple(jax.ShapeDtypeStruct(arr.shape, arr.dtype) for arr in flat_obs)

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        if last_actions is None:
            last_actions = jnp.zeros((batch_size, self.action_horizon, self.action_dim))

        while rtc_store.RTC_STORE[rtc_store.GLOBAL_KEY] is None:
            pass

        def step_noguide(carry):
            x_t, time, first_call = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], 
                mask=full_attn_mask, 
                positions=positions, 
                kv_cache=kv_cache, 
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt, first_call

        def step_guided(carry):
            x_t, time, first_call = carry

            def get_obs(_):
                with rtc_store.RTC_STORE_LOCK:
                    new_obs_struct = rtc_store.RTC_STORE.get(rtc_store.GLOBAL_KEY, None)
                new_obs_struct = _model.preprocess_observation(None, new_obs_struct, train=False)
                flat_new_obs, _ = jax.tree_util.tree_flatten(new_obs_struct)
                # 确认和模板一致
                # for i, (tpl, val) in enumerate(zip(flat_obs, flat_new_obs)):
                #     print(f"[{i}] 模板: dtype={tpl.dtype} shape={tpl.shape} 真实: dtype={val.dtype} shape={val.shape}")
                return tuple(flat_new_obs)

            flat_new_obs = io_callback(get_obs, shape_dtype_structs, None)
            new_obs = jax.tree_util.tree_unflatten(tree_def, flat_new_obs)

            def denoiser(x_t):
                suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                    new_obs, x_t, jnp.broadcast_to(time, batch_size)
                )
                suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
                prefix_attn_mask_ = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
                full_attn_mask = jnp.concatenate([prefix_attn_mask_, suffix_attn_mask], axis=-1)
                positions_ = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

                (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens], 
                    mask=full_attn_mask, 
                    positions=positions_, 
                    kv_cache=kv_cache, 
                    adarms_cond=[None, adarms_cond],
                )
                v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
                return x_t - time * v_t, v_t

            x_1, vjp_fun, v_t = jax.vjp(denoiser, x_t, has_aux=True)
            error = (last_actions - x_1) * rtc_mask
            pinv_correction = vjp_fun(error)[0]

            t_prime = 1.0 - time
            inv_r2 = (t_prime**2 + (1 - t_prime)**2) / ((1 - t_prime)**2 + 1e-8)
            c = (1 - t_prime) / (t_prime + 1e-8)
            guidance_weight = jnp.minimum(c * inv_r2, guide_beta)

            v_t = v_t + guidance_weight * pinv_correction

            return x_t + dt * v_t, time + dt, first_call

        def step(carry):
            return jax.lax.cond(
                carry[2],
                step_noguide,
                step_guided,
                carry
            )

        def cond(carry):
            x_t, time, first_call= carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _, first_call = jax.lax.while_loop(cond, step, (noise, 1.0, first_call))

        def no_change(carry):
            first_call, last_actions = carry
            return False, last_actions

        def modify(carry):
            first_call, last_actions = carry
            def clip_and_pad(x):
                cut = x[:, infer_delay_length + sleep_delay_length:, :]
                zeros = jnp.zeros((cut.shape[0], infer_delay_length + sleep_delay_length, cut.shape[2]), dtype=cut.dtype)
                return jnp.concatenate([cut, zeros], axis=1)
            return first_call, clip_and_pad(last_actions)

        first_call_new, last_actions_new = jax.lax.cond(first_call, no_change, modify, (first_call, x_0))

        return x_0, first_call_new, last_actions_new, count
    
    def rtc_new_obs_sample_actions_2(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        first_call: bool,
        last_actions: jax.Array,
        count: int,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        guide_beta: float = 2.0,
    ) -> tuple[_model.Actions, bool, jax.Array, int]:
        infer_delay_length = 10
        self.action_horizon = 60  # horizon 50 不够用
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        rtc_mask = make_rtc_mask((batch_size, self.action_horizon, self.action_dim), total=infer_delay_length)
        # flatten 保存结构模板
        flat_obs, tree_def = jax.tree_util.tree_flatten(observation)
        shape_dtype_structs = tuple(jax.ShapeDtypeStruct(arr.shape, arr.dtype) for arr in flat_obs)

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        if last_actions is None:
            last_actions = jnp.zeros((batch_size, self.action_horizon, self.action_dim))

        while rtc_store.RTC_STORE[rtc_store.GLOBAL_KEY] is None:
            pass

        def step_noguide(carry):
            x_t, time, first_call = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], 
                mask=full_attn_mask, 
                positions=positions, 
                kv_cache=kv_cache, 
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt, first_call

        def step_guided(carry):
            x_t, time, first_call = carry

            def get_obs(_):
                with rtc_store.RTC_STORE_LOCK:
                    new_obs_struct = rtc_store.RTC_STORE.get(rtc_store.GLOBAL_KEY, None)
                new_obs_struct = _model.preprocess_observation(None, new_obs_struct, train=False)
                flat_new_obs, _ = jax.tree_util.tree_flatten(new_obs_struct)
                # 确认和模板一致
                # for i, (tpl, val) in enumerate(zip(flat_obs, flat_new_obs)):
                #     print(f"[{i}] 模板: dtype={tpl.dtype} shape={tpl.shape} 真实: dtype={val.dtype} shape={val.shape}")
                return tuple(flat_new_obs)

            flat_new_obs = io_callback(get_obs, shape_dtype_structs, None)
            new_obs = jax.tree_util.tree_unflatten(tree_def, flat_new_obs)

            # 用 new_obs 重建 prefix 与 kv_cache
            prefix_tokens_new, prefix_mask_new, prefix_ar_mask_new = self.embed_prefix(new_obs)
            prefix_attn_mask_new = make_attn_mask(prefix_mask_new, prefix_ar_mask_new)
            positions_prefix_new = jnp.cumsum(prefix_mask_new, axis=1) - 1
            _, kv_cache_new = self.PaliGemma.llm(
                [prefix_tokens_new, None],
                mask=prefix_attn_mask_new,
                positions=positions_prefix_new
            )

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                new_obs, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_ = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_, suffix_attn_mask], axis=-1)
            positions_ = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], 
                mask=full_attn_mask, 
                positions=positions_, 
                kv_cache=kv_cache_new,    # 有了 kv_cache 重建就改成 kv_cache_new
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            tau = time
            r2 = ((1.0 - tau) ** 2) / (tau ** 2 + (1.0 - tau) ** 2 + 1e-8)
            weight = jnp.minimum(guide_beta, ((1.0 - tau) / (tau + 1e-8)) * r2)
            A_hat = x_t + (1.0 - tau) * v_t
            def loss(x):
                return jnp.sum(rtc_mask * (A_hat - last_actions) ** 2) / (jnp.sum(rtc_mask) + 1e-8)
            g = jax.grad(loss)(x_t)
            v_t = v_t - weight * g
            
            return x_t + dt * v_t, time + dt, first_call

        def step(carry):
            return jax.lax.cond(
                carry[2],
                step_noguide,
                step_guided,
                carry
            )

        def cond(carry):
            x_t, time, first_call = carry
            return time >= -dt / 2

        x_0, _, first_call = jax.lax.while_loop(cond, step, (noise, 1.0, first_call))

        def no_change(carry):
            first_call, last_actions = carry
            return False, last_actions

        def modify(carry):
            first_call, last_actions = carry
            def clip_and_pad(x):
                cut = x[:, infer_delay_length:, :]
                zeros = jnp.zeros((cut.shape[0], infer_delay_length, cut.shape[2]), dtype=cut.dtype)
                return jnp.concatenate([cut, zeros], axis=1)
            return first_call, clip_and_pad(last_actions)

        first_call_new, last_actions_new = jax.lax.cond(first_call, no_change, modify, (first_call, x_0))

        return x_0, first_call_new, last_actions_new, count
    
    def rtc_new_obs_sample_actions_3(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        first_call: bool,
        last_actions: jax.Array,
        count: int,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        guide_beta: float = 2.0,
    ) -> tuple[_model.Actions, bool, jax.Array, int]:
        infer_delay_length = 5
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        rtc_mask = make_rtc_mask((batch_size, self.action_horizon, self.action_dim), total=infer_delay_length)
        # flatten 保存结构模板
        flat_obs, tree_def = jax.tree_util.tree_flatten(observation)
        shape_dtype_structs = tuple(jax.ShapeDtypeStruct(arr.shape, arr.dtype) for arr in flat_obs)

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        if last_actions is None:
            last_actions = jnp.zeros((batch_size, self.action_horizon, self.action_dim))

        while rtc_store.RTC_STORE[rtc_store.GLOBAL_KEY] is None:
            pass

        def step_noguide(carry):
            x_t, time, first_call, observation, has_update1, has_update2, kv_cache = carry

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], 
                mask=full_attn_mask, 
                positions=positions, 
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond]
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt, first_call, observation, has_update1, has_update2, kv_cache

        def step_guided(carry):
            x_t, time, first_call, observation, has_update1, has_update2, kv_cache = carry

            def update1_branch(carry):
                x_t, time, first_call, last_used_obs, has_update1, has_update2, kv_cache = carry
                def get_obs(_):
                    with rtc_store.RTC_STORE_LOCK:
                        new_obs_struct = rtc_store.RTC_STORE.get(rtc_store.GLOBAL_KEY, None)
                    new_obs_struct = _model.preprocess_observation(None, new_obs_struct, train=False)
                    flat_new_obs, _ = jax.tree_util.tree_flatten(new_obs_struct)
                    return tuple(flat_new_obs)
                flat_new_obs = io_callback(get_obs, shape_dtype_structs, None)
                new_obs = jax.tree_util.tree_unflatten(tree_def, flat_new_obs)

                # 用 new_obs 重建 prefix 与 kv_cache
                prefix_tokens_new, prefix_mask_new, prefix_ar_mask_new = self.embed_prefix(new_obs)
                prefix_attn_mask_new = make_attn_mask(prefix_mask_new, prefix_ar_mask_new)
                positions_prefix_new = jnp.cumsum(prefix_mask_new, axis=1) - 1
                _, kv_cache_new = self.PaliGemma.llm(
                    [prefix_tokens_new, None],
                    mask=prefix_attn_mask_new,
                    positions=positions_prefix_new
                )

                return x_t, time, first_call, new_obs, jnp.array(True), has_update2, kv_cache_new
            
            def update2_branch(carry):
                x_t, time, first_call, last_used_obs, has_update1, has_update2, kv_cache= carry
                def get_obs(_):
                    with rtc_store.RTC_STORE_LOCK:
                        new_obs_struct = rtc_store.RTC_STORE.get(rtc_store.GLOBAL_KEY, None)
                    new_obs_struct = _model.preprocess_observation(None, new_obs_struct, train=False)
                    flat_new_obs, _ = jax.tree_util.tree_flatten(new_obs_struct)
                    return tuple(flat_new_obs)
                flat_new_obs = io_callback(get_obs, shape_dtype_structs, None)
                new_obs = jax.tree_util.tree_unflatten(tree_def, flat_new_obs)

                # 用 new_obs 重建 prefix 与 kv_cache
                prefix_tokens_new, prefix_mask_new, prefix_ar_mask_new = self.embed_prefix(new_obs)
                prefix_attn_mask_new = make_attn_mask(prefix_mask_new, prefix_ar_mask_new)
                positions_prefix_new = jnp.cumsum(prefix_mask_new, axis=1) - 1
                _, kv_cache_new = self.PaliGemma.llm(
                    [prefix_tokens_new, None],
                    mask=prefix_attn_mask_new,
                    positions=positions_prefix_new
                )

                return x_t, time, first_call, new_obs, has_update1, jnp.array(True), kv_cache_new
            
            def do_nothing_branch(carry):
                return carry
            
            # 第一次更新
            carry = jax.lax.cond(
                (time < 2 / 3) & (~has_update1),
                update1_branch,
                do_nothing_branch,
                carry
            )
            x_t, time, first_call, new_obs, has_update1, has_update2, kv_cache_new = carry

            # 第二次更新
            carry = jax.lax.cond(
                (time < 1 / 3) & (~has_update2),
                update2_branch,
                do_nothing_branch,
                carry
            )
            x_t, time, first_call, new_obs, has_update1, has_update2, kv_cache_new = carry

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                new_obs, x_t, jnp.broadcast_to(time, batch_size)
            )
            
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_ = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_, suffix_attn_mask], axis=-1)
            positions_ = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], 
                mask=full_attn_mask, 
                positions=positions_, 
                kv_cache=kv_cache_new,  # 有了 kv_cache 重建就改成 kv_cache_new
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            tau = time
            r2 = ((1.0 - tau) ** 2) / (tau ** 2 + (1.0 - tau) ** 2 + 1e-8)
            weight = jnp.minimum(guide_beta, ((1.0 - tau) / (tau + 1e-8)) * r2)
            A_hat = x_t + (1.0 - tau) * v_t
            def loss(x):
                return jnp.sum(rtc_mask * (A_hat - last_actions) ** 2) / (jnp.sum(rtc_mask) + 1e-8)
            g = jax.grad(loss)(x_t)
            v_t = v_t - weight * g

            return x_t + dt * v_t, time + dt, first_call, new_obs, has_update1, has_update2, kv_cache_new

        def step(carry):
            return jax.lax.cond(
                carry[2],
                step_noguide,
                step_guided,
                carry
            )

        def cond(carry):
            x_t, time, first_call, observation, has_update1, has_update2, kv_cache = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _, first_call, _, _, _, _ = jax.lax.while_loop(cond, step, (noise, 1.0, first_call, observation, False, False, kv_cache))

        def no_change(carry):
            first_call, last_actions = carry
            return False, last_actions

        def modify(carry):
            first_call, last_actions = carry
            def clip_and_pad(x):
                cut = x[:, infer_delay_length:, :]
                zeros = jnp.zeros((cut.shape[0], infer_delay_length, cut.shape[2]), dtype=cut.dtype)
                return jnp.concatenate([cut, zeros], axis=1)
            return first_call, clip_and_pad(last_actions)

        first_call_new, last_actions_new = jax.lax.cond(first_call, no_change, modify, (first_call, x_0))

        return x_0, first_call_new, last_actions_new, count
    
    def rtc_new_obs_sample_actions_4(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        first_call: bool,
        last_actions: jax.Array,
        count: int,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        guide_beta: float = 2.0,
    ) -> tuple[_model.Actions, bool, jax.Array, int]:
        infer_delay_length = 10
        sleep_delay_length = 0
        self.action_horizon = 60
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        rtc_mask = make_rtc_mask((batch_size, self.action_horizon, self.action_dim), total=infer_delay_length)
        # flatten 保存结构模板
        flat_obs, tree_def = jax.tree_util.tree_flatten(observation)
        shape_dtype_structs = tuple(jax.ShapeDtypeStruct(arr.shape, arr.dtype) for arr in flat_obs)

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        if last_actions is None:
            last_actions = jnp.zeros((batch_size, self.action_horizon, self.action_dim))

        while rtc_store.RTC_STORE[rtc_store.GLOBAL_KEY] is None:
            pass

        def step_noguide(carry):
            x_t, time, first_call, observation, has_update1, has_update2 = carry

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], 
                mask=full_attn_mask, 
                positions=positions, 
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond]
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt, first_call, observation, has_update1, has_update2 

        def step_guided(carry):
            x_t, time, first_call, observation, has_update1, has_update2 = carry

            def update1_branch(carry):
                x_t, time, first_call, last_used_obs, has_update1, has_update2 = carry
                def get_obs(_):
                    with rtc_store.RTC_STORE_LOCK:
                        new_obs_struct = rtc_store.RTC_STORE.get(rtc_store.GLOBAL_KEY, None)
                    new_obs_struct = _model.preprocess_observation(None, new_obs_struct, train=False)
                    flat_new_obs, _ = jax.tree_util.tree_flatten(new_obs_struct)
                    return tuple(flat_new_obs)
                flat_new_obs = io_callback(get_obs, shape_dtype_structs, None)
                new_obs = jax.tree_util.tree_unflatten(tree_def, flat_new_obs)

                return x_t, time, first_call, new_obs, jnp.array(True), has_update2
            
            def update2_branch(carry):
                x_t, time, first_call, last_used_obs, has_update1, has_update2 = carry
                def get_obs(_):
                    with rtc_store.RTC_STORE_LOCK:
                        new_obs_struct = rtc_store.RTC_STORE.get(rtc_store.GLOBAL_KEY, None)
                    new_obs_struct = _model.preprocess_observation(None, new_obs_struct, train=False)
                    flat_new_obs, _ = jax.tree_util.tree_flatten(new_obs_struct)
                    return tuple(flat_new_obs)
                flat_new_obs = io_callback(get_obs, shape_dtype_structs, None)
                new_obs = jax.tree_util.tree_unflatten(tree_def, flat_new_obs)

                return x_t, time, first_call, new_obs, has_update1, jnp.array(True)
            
            def do_nothing_branch(carry):
                return carry
            
            # 第一次更新
            carry = jax.lax.cond(
                (time < 2 / 3) & (~has_update1),
                update1_branch,
                do_nothing_branch,
                carry
            )
            x_t, time, first_call, new_obs, has_update1, has_update2 = carry

            # 第二次更新
            carry = jax.lax.cond(
                (time < 1 / 3) & (~has_update2),
                update2_branch,
                do_nothing_branch,
                carry
            )
            x_t, time, first_call, new_obs, has_update1, has_update2 = carry

            def denoiser(x_t):
                suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                    new_obs, x_t, jnp.broadcast_to(time, batch_size)
                )
                suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
                prefix_attn_mask_ = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
                full_attn_mask = jnp.concatenate([prefix_attn_mask_, suffix_attn_mask], axis=-1)
                positions_ = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

                (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens], 
                    mask=full_attn_mask, 
                    positions=positions_, 
                    kv_cache=kv_cache, 
                    adarms_cond=[None, adarms_cond],
                )
                v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
                return x_t - time * v_t, v_t

            x_1, vjp_fun, v_t = jax.vjp(denoiser, x_t, has_aux=True)
            error = (last_actions - x_1) * rtc_mask
            pinv_correction = vjp_fun(error)[0]

            t_prime = 1.0 - time
            inv_r2 = (t_prime**2 + (1 - t_prime)**2) / ((1 - t_prime)**2 + 1e-8)
            c = (1 - t_prime) / (t_prime + 1e-8)
            guidance_weight = jnp.minimum(c * inv_r2, guide_beta)

            v_t = v_t + guidance_weight * pinv_correction

            return x_t + dt * v_t, time + dt, first_call, new_obs, has_update1, has_update2

        def step(carry):
            return jax.lax.cond(
                carry[2],
                step_noguide,
                step_guided,
                carry
            )

        def cond(carry):
            x_t, time, first_call, observation, has_update1, has_update2 = carry
            # robust to floating-point error
            return time >= -dt / 2 * 5

        x_0, _, first_call, _, _, _ = jax.lax.while_loop(cond, step, (noise, 1.0, first_call, observation, False, False))

        def no_change(carry):
            first_call, last_actions = carry
            return False, last_actions

        def modify(carry):
            first_call, last_actions = carry
            def clip_and_pad(x):
                cut = x[:, infer_delay_length:, :]
                zeros = jnp.zeros((cut.shape[0], infer_delay_length, cut.shape[2]), dtype=cut.dtype)
                return jnp.concatenate([cut, zeros], axis=1)
            return first_call, clip_and_pad(last_actions)

        first_call_new, last_actions_new = jax.lax.cond(first_call, no_change, modify, (first_call, x_0))

        return x_0, first_call_new, last_actions_new, count

def make_rtc_mask(shape, total=5, ones_steps=3, decay_steps=5):
    # shape: (b, h, d)
    b, h, d = shape
    zeros1 = jnp.zeros(total - ones_steps)  # 前面的0
    ones = jnp.ones(ones_steps)             # ones_steps个1
    decay = jnp.linspace(1, 0, decay_steps + 1)[1:] if decay_steps > 0 else jnp.array([])
    zeros2 = jnp.zeros(h - total - decay_steps) if h > total + decay_steps else jnp.array([])

    mask_1d = jnp.concatenate([zeros1, ones, decay, zeros2])
    # 防止长度超出/不够，截断或补0
    mask_1d = jnp.pad(mask_1d[:h], (0, max(0, h - mask_1d.shape[0])))
    mask = mask_1d[None, :, None]
    mask = jnp.broadcast_to(mask, shape)
    return mask
