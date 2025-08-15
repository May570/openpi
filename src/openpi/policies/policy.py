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
    ):
        # self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._rtc_new_obs_sample_actions_1 = nnx_utils.module_jit(model.rtc_new_obs_sample_actions_1)  # 只有状态更新，只有后缀每次都前向
        self._rtc_new_obs_sample_actions_2 = nnx_utils.module_jit(model.rtc_new_obs_sample_actions_2)  # 图像也更新，前缀和后缀每次都重新前向
        self._rtc_new_obs_sample_actions_3 = nnx_utils.module_jit(model.rtc_new_obs_sample_actions_3)  # 图像也更新，但是只更新两次
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self.first_call = True
        self.last_actions = None
        self.debug_counter = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        start_time = time.monotonic()
        self._rng, sample_rng = jax.random.split(self._rng)

        actions, self.first_call, self.last_actions, self.debug_counter = self._rtc_new_obs_sample_actions_3(
            sample_rng,
            _model.Observation.from_dict(inputs),
            self.first_call,
            self.last_actions,
            self.debug_counter,
            **self._sample_kwargs
        )
        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        # outputs = {
        #     "state": inputs["state"],
        #     "actions": self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **self._sample_kwargs),
        # }
        # Unbatch and convert to np.ndarray.        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
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
