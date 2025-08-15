import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

from jax.experimental import io_callback
import time
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


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class Pi0(_model.BaseModel):
    def __init__(self, config: Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")
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
        self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

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
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # add a single state token
        state_token = self.state_proj(obs.state)[:, None, :]
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        # image/language inputs do not attend to state or actions
        ar_mask += [True]

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        # mix timestep + action information using an MLP
        action_tokens = self.action_in_proj(noisy_actions)
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)
        tokens.append(action_time_tokens)
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

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
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions
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
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
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
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
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
    
    def rit_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        first_call: bool,
        last_actions: jax.Array,
        count: int,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        rti_steps: int | at.Int[at.Array, ""] = 2,
        rti_t0: float = 1.0,
    ) -> tuple[_model.Actions, bool, jax.Array, int]:
        self.action_horizon = 10
        delay_length = 3
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        if last_actions is None:
            last_actions = jnp.zeros((batch_size, self.action_horizon, self.action_dim))

        first_call_b = jnp.asarray(first_call, dtype=bool)
        time0 = jnp.where(first_call_b, jnp.array(1.0, jnp.float32), jnp.array(rti_t0, jnp.float32))
        steps0 = jnp.where(
            first_call_b,
            jnp.array(num_steps, jnp.float32),
            jnp.array(rti_steps, jnp.float32),
        )
        dt = -(time0 - 0.0) / steps0
        x_init = jnp.where(first_call_b, noise, last_actions)

        def step_noguide(carry):
            x_t, time, is_first = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
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
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt, False

        def step_guided(carry):
            x_t, time, is_first = carry
                    
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_ = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_, suffix_attn_mask], axis=-1)
            positions_ = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions_, kv_cache=kv_cache
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt, False

        def step(carry):
            return jax.lax.cond(
                carry[2],
                step_noguide,
                step_guided,
                carry
            )

        def cond(carry):
            x_t, time, is_first = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _, first_call_new = jax.lax.while_loop(cond, step, (x_init, time0, first_call_b))

        # 裁剪 + 复制尾部填充
        def clip_and_pad(x):
            cut = x[:, delay_length:, :]
            tail = x[:, -1:, :]
            tail_rep = jnp.repeat(tail, repeats=delay_length, axis=1)
            return jnp.concatenate([cut, tail_rep], axis=1) 

        last_actions_new = clip_and_pad(x_0)

        return x_0, first_call_new, last_actions_new, count
    
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
        delay_length = 26
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        rtc_mask = make_rtc_mask((batch_size, self.action_horizon, self.action_dim), total=delay_length)
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
            # print(f"4.[标记] 线程:{threading.current_thread().name} | self id={id(self)}, id(RTC_STORE)={id(rtc_store.RTC_STORE)}")
            time.sleep(0.001)

        def step_noguide(carry):
            x_t, time, is_first, has_update1, has_update2, last_actions = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
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
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            x_e = x_t + dt * v_t
            last_actions = x_e

            return x_e, time + dt, False, has_update1, has_update2, last_actions

        def step_guided(carry):
            x_t, time, is_first, has_update1, has_update2, last_actions = carry

            def get_obs(_):
                # print(f"3.[标记] 线程:{threading.current_thread().name} | self id={id(self)}, id(RTC_STORE)={id(rtc_store.RTC_STORE)}")
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

            # def update1_branch(carry):
            #     x_t, time, is_first, last_used_obs, has_update1, has_update2 = carry
            #     def get_obs(_):
            #         with rtc_store.RTC_STORE_LOCK:
            #             new_obs_struct = rtc_store.RTC_STORE.get(rtc_store.GLOBAL_KEY, None)
            #         new_obs_struct = _model.preprocess_observation(None, new_obs_struct, train=False)
            #         flat_new_obs, _ = jax.tree_util.tree_flatten(new_obs_struct)
            #         return tuple(flat_new_obs)
            #     flat_new_obs = io_callback(get_obs, shape_dtype_structs, None)
            #     new_obs = jax.tree_util.tree_unflatten(tree_def, flat_new_obs)
            #     return x_t, time, is_first, new_obs, jnp.array(True), has_update2
            # def update2_branch(carry):
            #     x_t, time, is_first, last_used_obs, has_update1, has_update2 = carry
            #     def get_obs(_):
            #         with rtc_store.RTC_STORE_LOCK:
            #             new_obs_struct = rtc_store.RTC_STORE.get(rtc_store.GLOBAL_KEY, None)
            #         new_obs_struct = _model.preprocess_observation(None, new_obs_struct, train=False)
            #         flat_new_obs, _ = jax.tree_util.tree_flatten(new_obs_struct)
            #         return tuple(flat_new_obs)
            #     flat_new_obs = io_callback(get_obs, shape_dtype_structs, None)
            #     new_obs = jax.tree_util.tree_unflatten(tree_def, flat_new_obs)
            #     return x_t, time, is_first, new_obs, has_update1, jnp.array(True)
            # def do_nothing_branch(carry):
            #     return carry
            # # 第一次更新
            # carry = jax.lax.cond(
            #     (time < 2 / 3) & (~has_update1),
            #     update1_branch,
            #     do_nothing_branch,
            #     carry
            # )
            # x_t, time, is_first, last_used_obs, has_update1, has_update2 = carry
            # # 第二次更新
            # carry = jax.lax.cond(
            #     (time < 1 / 3) & (~has_update2),
            #     update2_branch,
            #     do_nothing_branch,
            #     carry
            # )
            # x_t, time, is_first, last_used_obs, has_update1, has_update2 = carry
            # new_obs = last_used_obs

            # 用 new_obs 重建 prefix 与 kv_cache
            prefix_tokens_new, prefix_mask_new, prefix_ar_mask_new = self.embed_prefix(new_obs)
            prefix_attn_mask_new = make_attn_mask(prefix_mask_new, prefix_ar_mask_new)
            positions_prefix_new = jnp.cumsum(prefix_mask_new, axis=1) - 1
            _, kv_cache_new = self.PaliGemma.llm(
                [prefix_tokens_new, None],
                mask=prefix_attn_mask_new,
                positions=positions_prefix_new
            )

            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                new_obs, x_t, jnp.broadcast_to(time, batch_size)
            )
            
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_ = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_, suffix_attn_mask], axis=-1)
            positions_ = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions_, kv_cache=kv_cache_new  # 有了 kv_cache a重进就改成 kv_cache_new
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            tau = time
            r2 = ((1.0 - tau) ** 2) / (tau ** 2 + (1.0 - tau) ** 2 + 1e-8)
            weight = jnp.minimum(guide_beta, ((1.0 - tau) / (tau + 1e-8)) * r2)
            # weight = guide_beta * tau   # 越往后越重视obs
            A_hat = x_t + (1.0 - tau) * v_t
            def loss(x):
                return jnp.sum(rtc_mask * (A_hat - last_actions) ** 2) / (jnp.sum(rtc_mask) + 1e-8)
            g = jax.grad(loss)(x_t)
            v_t = v_t - weight * g
            x_e = x_t + dt * v_t

            def clip_and_pad(x):
                cut = x[:, delay_length:, :]
                zeros = jnp.zeros((cut.shape[0], delay_length, cut.shape[2]), dtype=cut.dtype)
                return jnp.concatenate([cut, zeros], axis=1)
            last_actions = clip_and_pad(x_e)

            return x_e, time + dt, False, has_update1, has_update2, last_actions

        def step(carry):
            return jax.lax.cond(
                carry[2],
                step_noguide,
                step_guided,
                carry
            )

        def cond(carry):
            x_t, time, is_first, has_update1, has_update2, last_actions = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _, first_call_new, _, _, last_actions_new = jax.lax.while_loop(cond, step, (noise, 1.0, first_call, False, False, last_actions))

        # # 裁剪+补零
        # def clip_and_pad(x):
        #     cut = x[:, delay_length:, :]
        #     zeros = jnp.zeros((cut.shape[0], delay_length, cut.shape[2]), dtype=cut.dtype)
        #     return jnp.concatenate([cut, zeros], axis=1)

        # last_actions_new = clip_and_pad(x_0)

        return x_0, first_call_new, last_actions_new, count
    
    def rtc_only_actions(
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
        total_delay_length = 4
        infer_delay_length = 4
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        rtc_mask = make_rtc_mask((batch_size, self.action_horizon, self.action_dim), total=infer_delay_length, ones_steps=infer_delay_length)

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        if last_actions is None:
            last_actions = jnp.zeros((batch_size, self.action_horizon, self.action_dim))

        def step_noguide(carry):
            x_t, time, is_first = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
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
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt, False

        def step_guided(carry):
            x_t, time, is_first = carry
                    
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_ = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_, suffix_attn_mask], axis=-1)
            positions_ = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions_, kv_cache=kv_cache
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            tau = time
            r2 = ((1.0 - tau) ** 2) / (tau ** 2 + (1.0 - tau) ** 2 + 1e-8)
            weight = jnp.minimum(guide_beta, ((1.0 - tau) / (tau + 1e-8)) * r2)
            # weight = guide_beta * tau   # 越往后越重视obs
            A_hat = x_t + (1.0 - tau) * v_t
            def loss(x):
                return jnp.sum(rtc_mask * (A_hat - last_actions) ** 2) / (jnp.sum(rtc_mask) + 1e-8)
            g = jax.grad(loss)(x_t)
            v_t = v_t - weight * g

            return x_t + dt * v_t, time + dt, False

        def step(carry):
            return jax.lax.cond(
                carry[2],
                step_noguide,
                step_guided,
                carry
            )

        def cond(carry):
            x_t, time, is_first = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _, first_call_new = jax.lax.while_loop(cond, step, (noise, 1.0, first_call))

        # 裁剪+补零
        def clip_and_pad(x):
            cut = x[:, total_delay_length:, :]
            zeros = jnp.zeros((cut.shape[0], total_delay_length, cut.shape[2]), dtype=cut.dtype)
            return jnp.concatenate([cut, zeros], axis=1)

        last_actions_new = clip_and_pad(x_0)

        return x_0, first_call_new, last_actions_new, count

def make_rtc_mask(shape, total=3, ones_steps=3, decay_steps=3):
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
