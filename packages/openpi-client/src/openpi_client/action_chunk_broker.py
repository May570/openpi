from typing import Dict

import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy

import threading
import time


class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int, use_rtc: bool = False):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None

        self._use_rtc = use_rtc
        self._last_obs = None
        self._buffer: Dict[str, np.ndarray] | None = None
        self._rtc_current_step: int = 0
        self._stop_event = None
        self._thread = None
        self._is_first = True

    def _start_infer_thread(self):
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_infer_loop)
        self._thread.start()

    def _run_infer_loop(self):
        while not self._stop_event.is_set():
            if self._last_obs is not None:
                # print(f"-Running inference for action chunk broker with last observation--")
                self._buffer = self._policy.infer(self._last_obs)
                # print(f"---------Finished inference for action chunk broker--------")

                if self._is_first:
                    self._rtc_current_step = 0
                    self._is_first = False
                else:
                    self._rtc_current_step = 3

                time.sleep(0.5)  # Sleep to avoid busy waiting

    def close(self):
        # print(f"---------Closing ActionChunkBroker--------")
        if self._thread is not None and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join()
        self._thread = None
        self._stop_event = None
        self._buffer = None

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        self._last_obs = obs

        if self._use_rtc:
            if self._thread is None or not self._thread.is_alive():
                self._start_infer_thread()
                
            # if self._rtc_current_step >= self._action_horizon:
            #     self._buffer = None

            while self._buffer is None or self._rtc_current_step >= self._buffer["actions"].shape[0]:
                pass

            def slicer(x):
                if isinstance(x, np.ndarray):
                    return x[self._rtc_current_step, ...]
                else:
                    return x
            results = tree.map_structure(slicer, self._buffer)

            self._rtc_current_step += 1

        else:
            if self._last_results is None:
                self._last_results = self._policy.infer(obs)
                self._cur_step = 0

            def slicer(x):
                if isinstance(x, np.ndarray):
                    return x[self._cur_step, ...]
                else:
                    return x

            results = tree.map_structure(slicer, self._last_results)
            self._cur_step += 1

            if self._cur_step >= self._action_horizon:
                self._last_results = None

        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0
        self._last_obs = None
        self._rtc_current_step = 0
