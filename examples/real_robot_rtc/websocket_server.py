import asyncio
import http
import logging
import time
import traceback
import threading
import base64
import io

from torchvision import transforms
from PIL import Image
import numpy as np
from openpi_client import base_policy as _base_policy
import msgpack_numpy
import websockets
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)
to_tensor = transforms.ToTensor()

class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}

        self._latest_raw = None
        self._latest_obs = None
        # 推理线程
        self._infer_thread = None
        self._stop_infer_event = None
        # 重构/更新线程
        self._obs_thread = None
        self._stop_obs_event = None

        self._loop = None
        self._ws = None
        self._lock = threading.Lock()

        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    def decode_image_base64(self, image_base64):
        """解码base64图片"""
        try:
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            image = to_tensor(image)
            return image
        except Exception as e:
            logger.error(f"图片解码失败: {e}")
            raise ValueError(f"图片解码失败: {e}")
        
    def decode_image_bytes(self, image_bytes: bytes):
        """解码 JPEG/PNG bytes -> Tensor[C,H,W]"""
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image = to_tensor(image)
            return image
        except Exception as e:
            logger.error(f"图片解码失败: {e}")
            raise ValueError(f"图片解码失败: {e}")

    # def process_images(self, images_dict):
    #     """处理图片列表，适配openpi的图片格式"""
    #     try:
    #         sample_dict = {}
    #         # 根据openpi的配置调整图片键名
    #         for k in ['cam_high', 'cam_left_wrist', 'cam_right_wrist']:
    #             if k in images_dict:
    #                 # sample_dict[k] = self.decode_image_base64(images_dict[k])
    #                 sample_dict[k] = self.decode_image_bytes(images_dict[k])
    #             else:
    #                 logger.warning(f"缺少图片: {k}")

    #     except Exception as e:
    #         logger.error(f"处理图片失败: {e}")
    #         raise ValueError(f"处理图片失败: {e}")

    #     return sample_dict

    def process_images(self, images_dict):
        """处理图片列表，适配openpi的图片格式"""
        try:
            sample_dict = {}
            for k in ['cam_high', 'cam_left_wrist', 'cam_right_wrist']:
                if k in images_dict:
                    val = images_dict[k]
                    if isinstance(val, str):
                        # test 客户端发来的 base64 str
                        image_bytes = base64.b64decode(val)
                        sample_dict[k] = self.decode_image_bytes(image_bytes)
                    elif isinstance(val, (bytes, bytearray)):
                        # 直接是 bytes
                        sample_dict[k] = self.decode_image_bytes(val)
                    else:
                        logger.error(f"不支持的图片类型: {type(val)} for key={k}")
                else:
                    logger.warning(f"缺少图片: {k}")

        except Exception as e:
            logger.error(f"处理图片失败: {e}")
            raise ValueError(f"处理图片失败: {e}")

        return sample_dict

    def _start_infer_thread(self):
        self._stop_infer_thread()
        self._stop_infer_event = threading.Event()
        self._infer_thread = threading.Thread(target=self._run_infer_loop, daemon=True)
        self._infer_thread.start()

    def _start_obs_thread(self):
        self._stop_obs_thread()
        self._stop_obs_event = threading.Event()
        self._obs_thread = threading.Thread(target=self._run_obs_loop, daemon=True)
        self._obs_thread.start()

    def _stop_infer_thread(self):
        if self._stop_infer_event:
            self._stop_infer_event.set()
        if self._infer_thread and self._infer_thread.is_alive():
            try:
                self._infer_thread.join(timeout=2.0)
            except Exception:
                logging.exception("Join infer thread failed")
        self._infer_thread = None
        self._stop_infer_event = None
        logging.info("Infer thread stopped")

    def _stop_obs_thread(self):
        if self._stop_obs_event:
            self._stop_obs_event.set()
        if self._obs_thread and self._obs_thread.is_alive():
            try:
                self._obs_thread.join(timeout=2.0)
            except Exception:
                logging.exception("Join obs thread failed")
        self._obs_thread = None
        self._stop_obs_event = None
        logging.info("Obs thread stopped")

    def _run_infer_loop(self):
        while not self._stop_infer_event.is_set():
            if self._latest_obs is not None:
                with self._lock:
                    observation = self._latest_obs
                    self._latest_obs = None
                try:
                    result = self._policy.infer(observation)
                    logging.info(f"Infer result type={type(result)}, keys={list(result.keys()) if isinstance(result, dict) else result}")
                except Exception as e:
                    logging.error(f"Infer failed: {e}")
                    break
                if self._ws and self._loop:
                    logging.info(f"Sending inference result to client")
                    send_coro = self._ws.send(msgpack_numpy.packb(result))
                    asyncio.run_coroutine_threadsafe(send_coro, self._loop)
                    time.sleep(1)

    def _run_obs_loop(self):
        while not self._stop_obs_event.is_set():
            if self._latest_raw is not None:
                with self._lock:
                    raw_data = self._latest_raw
                    self._latest_raw = None
                images = raw_data.get('images')
                state = raw_data.get('state')
                images_tensor = self.process_images(images)
                obs = {
                    "images": {
                        "cam_high": images_tensor['cam_high'],
                        "cam_left_wrist": images_tensor['cam_left_wrist'],
                        "cam_right_wrist": images_tensor['cam_right_wrist'],
                    },
                    "state": np.array(state[0]).astype(np.float32),
                    "prompt": "Pick up the fruit and place it into the bowl.",
                }
                with self._lock:
                    self._latest_obs = obs
                    logging.info("Observation reconstructed")
                # self._policy.update_obs(obs)
                # logging.info("Observation updated")

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            # compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")

        self._loop = asyncio.get_running_loop()
        self._ws = websocket
        self._policy.reset()

        self._start_infer_thread()
        self._start_obs_thread()
        self._latest_raw = None
        self._latest_obs = None

        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                data = msgpack_numpy.unpackb(await websocket.recv())
                with self._lock:
                    self._latest_raw = data

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                self._policy.reset()
                self._stop_infer_thread()
                self._stop_obs_thread()
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
