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
from openpi_client import msgpack_numpy
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

        self._latest_obs = None
        self._thread = None
        self._stop_event = None

        self._loop = None
        self._ws = None
        
        self._update = False

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

    def process_images(self, images_dict):
        """处理图片列表，适配openpi的图片格式"""
        try:
            sample_dict = {}
            # 根据openpi的配置调整图片键名
            for k in ['cam_high', 'cam_left_wrist', 'cam_right_wrist']:
                if k in images_dict:
                    sample_dict[k] = self.decode_image_base64(images_dict[k])
                else:
                    logger.warning(f"缺少图片: {k}")

        except Exception as e:
            logger.error(f"处理图片失败: {e}")
            raise ValueError(f"处理图片失败: {e}")

        return sample_dict

    def _start_infer_thread(self):
        self._stop_infer_thread()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_infer_loop, daemon=True)
        self._thread.start()

    def _stop_infer_thread(self):
        if self._stop_event:
            self._stop_event.set()
        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=2.0)
            except Exception:
                logging.exception("Join infer thread failed")
        self._thread = None
        self._stop_event = None

    def _run_infer_loop(self):
        while not self._stop_event.is_set():
            if self._latest_obs is not None and self._update:
                result = self._policy.infer(self._latest_obs)
                if self._ws and self._loop:
                    # 将推理结果返回给 WebSocket 客户端
                    logging.info(f"Sending inference result to client")
                    send_coro = self._ws.send(msgpack_numpy.packb(result))
                    asyncio.run_coroutine_threadsafe(send_coro, self._loop)
                    self._update = False
                    time.sleep(2)  # Sleep to avoid busy waiting

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")

        self._stop_infer_thread()
        self._latest_obs = None
        self._loop = asyncio.get_running_loop()
        self._ws = websocket

        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                obs_data = msgpack_numpy.unpackb(await websocket.recv())
                images = obs_data.get('images')
                state = obs_data.get('state')
                images_tensor = self.process_images(images)
                obs = {
                    "images": {
                        "cam_high": images_tensor['cam_high'],
                        "cam_left_wrist": images_tensor['cam_left_wrist'],
                        "cam_right_wrist": images_tensor['cam_right_wrist'],
                    },
                    "state": np.array(state[0]).astype(np.float32),
                    "prompt": "pick up the orange and put it into the basket",
                }

                self._latest_obs = obs
                self._update = True

                if self._thread is None or not self._thread.is_alive():
                    self._start_infer_thread()

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
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
