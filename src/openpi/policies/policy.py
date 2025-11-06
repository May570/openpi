from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
import openpi.shared.rtc_store as rtc_store

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._meanflow_sample_actions = nnx_utils.module_jit(model.meanflow_sample_actions)
            self._rtc_original = nnx_utils.module_jit(model.rtc_original)
            # self._rtc_sample_actions = nnx_utils.module_jit(model.rtc_sample_actions)                      # rtc 复现
            self._rtc_new_obs_sample_actions_1 = nnx_utils.module_jit(model.rtc_new_obs_sample_actions_1)  # 只有状态更新，只有后缀每次都前向
            # self._rtc_new_obs_sample_actions_2 = nnx_utils.module_jit(model.rtc_new_obs_sample_actions_2)  # 图像也更新，前缀和后缀每次都重新前向
            # self._rtc_new_obs_sample_actions_3 = nnx_utils.module_jit(model.rtc_new_obs_sample_actions_3)  # 图像也更新，但是只更新两次
            self._rtc_new_obs_sample_actions_4 = nnx_utils.module_jit(model.rtc_new_obs_sample_actions_4)  # 只有状态更新，但是只更新两次
            self.first_call = True
            self.last_actions = None
            self.debug_counter = 0
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()

        if not self._is_pytorch_model:
            actions, self.first_call, self.last_actions, self.debug_counter = self._rtc_original(
                sample_rng_or_pytorch_device,
                observation,
                self.first_call,
                self.last_actions,
                self.debug_counter,
                **sample_kwargs
            )
            outputs = {
                "state": inputs["state"],
                "actions": actions,
            }

            # outputs = {
            #     "state": inputs["state"],
            #     "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
            # }
            
            # Unbatch and convert to np.ndarray.
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        else:
            outputs = {
                "state": inputs["state"],
                "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
            }
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        
        model_time = time.monotonic() - start_time

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
    
    @override
    def reset(self) -> None:
        """Reset the policy to its initial state."""
        self.first_call = True
        self.last_actions = None
        self.debug_counter = 0
        rtc_store.RTC_STORE[rtc_store.GLOBAL_KEY] = None
        logging.info("Policy has been reset.")

    def update_obs(self, new_obs: dict) -> None:
        inputs = jax.tree.map(lambda x: x, new_obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        with rtc_store.RTC_STORE_LOCK:
            rtc_store.RTC_STORE[rtc_store.GLOBAL_KEY] = _model.Observation.from_dict(inputs)


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
