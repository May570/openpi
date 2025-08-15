import logging
import threading
import time

from openpi_client.runtime import agent as _agent
from openpi_client.runtime import environment as _environment
from openpi_client.runtime import subscriber as _subscriber

import sys

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(levelname)s %(asctime)s %(message)s",
    datefmt="%H:%M:%S"
)


class Runtime:
    """The core module orchestrating interactions between key components of the system."""

    def __init__(
        self,
        environment: _environment.Environment,
        agent: _agent.Agent,
        subscribers: list[_subscriber.Subscriber],
        max_hz: float = 0,
        num_episodes: int = 1,
        max_episode_steps: int = 0,
        use_rtc: bool = False,
    ) -> None:
        self._environment = environment
        self._agent = agent
        self._subscribers = subscribers
        self._max_hz = max_hz
        self._num_episodes = num_episodes
        self._max_episode_steps = max_episode_steps

        self._in_episode = False
        self._episode_steps = 0
        self._steps_list = []
        self._episode_durations = []
        self._episode_successes = []
        self._use_rtc = use_rtc

    def run(self) -> None:
        """Runs the runtime loop continuously until stop() is called or the environment is done."""
        for _ in range(self._num_episodes):
            self._run_episode()

        # Final reset, this is important for real environments to move the robot to its home position.
        self._environment.reset()

        # 排除 episode 1，单独打印
        for idx, duration in enumerate(self._episode_durations):
            steps = self._steps_list[idx]
            if idx == 0:
                logging.info(f"Episode {idx + 1}: duration = {duration:.2f} s, steps = {steps}, {self._episode_successes[idx]} (warmup, excluded from average)")
                continue
            hz = steps / duration if duration > 0 else 0
            logging.info(f"Episode {idx + 1}: duration = {duration:.2f} s, steps = {steps}, {self._episode_successes[idx]}, approx {hz:.2f} Hz")

        # 平均统计不含 warmup
        durations = self._episode_durations[1:]
        steps_list = self._steps_list[1:]
        if durations:
            avg_duration = sum(durations) / len(durations)
            avg_steps = sum(steps_list) / len(steps_list)
            avg_hz = avg_steps / avg_duration
            logging.info(f"[Summary] (Excluding warmup) Average duration: {avg_duration:.2f} s, Average frequency: {avg_hz:.2f} Hz")


    def run_in_new_thread(self) -> threading.Thread:
        """Runs the runtime loop in a new thread."""
        thread = threading.Thread(target=self.run)
        thread.start()
        return thread

    def mark_episode_complete(self) -> None:
        """Marks the end of an episode."""
        self._in_episode = False

    def _run_episode(self) -> None:
        """Runs a single episode."""
        logging.info("Starting episode...")
        start_time = time.time()

        self._environment.reset()
        self._agent.reset()
        for subscriber in self._subscribers:
            subscriber.on_episode_start()

        self._in_episode = True
        self._episode_steps = 0
        step_time = 1 / self._max_hz if self._max_hz > 0 else 0
        last_step_time = time.time()

        while self._in_episode:
            self._step()
            self._episode_steps += 1

            # Sleep to maintain the desired frame rate
            now = time.time()
            dt = now - last_step_time
            if dt < step_time:
                time.sleep(step_time - dt)
                last_step_time = time.time()
            else:
                last_step_time = now
        if self._use_rtc:
            self._agent.close()

        duration = time.time() - start_time
        self._episode_durations.append(duration)
        self._steps_list.append(self._episode_steps)

        info = self._environment.get_info()
        # logging.info(f"info: {info}")
        success = info.get("is_success", False)
        self._episode_successes.append(success)
        status = "SUCCESS ✅" if success else "FAIL ❌"
        logging.info(f"Episode {len(self._episode_successes)} result: {status}")

        logging.info(f"Episode completed in {duration:.2f} seconds.")
        for subscriber in self._subscribers:
            subscriber.on_episode_end()

    def _step(self) -> None:
        """A single step of the runtime loop."""
        observation = self._environment.get_observation()
        action = self._agent.get_action(observation)
        self._environment.apply_action(action)
        # print(f"-----------------------apply_action----------------------------")

        for subscriber in self._subscribers:
            subscriber.on_step(observation, action)

        if self._environment.is_episode_complete() or (
            self._max_episode_steps > 0 and self._episode_steps >= self._max_episode_steps
        ):
            self.mark_episode_complete()
