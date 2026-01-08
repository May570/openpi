import logging
import time
from typing import Dict, Optional, Tuple
import queue
import threading

from typing_extensions import override
import websockets.sync.client

import msgpack_numpy


class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, message_queue: queue.Queue, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()
        self.message_queue = message_queue
        self._recv_stop = False
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)
    
    def send_obs(self, obs: Dict) -> None:
        """Allow ActionChunkBroker to update the current observation."""
        data = self._packer.pack(obs)
        self._ws.send(data)

    def on_message(self, message):
        logging.info(f"Received message from server")
        self.message_queue.put(message)

    def _recv_loop(self):
        logging.info("WebsocketClientPolicy recv loop started")
        while not self._recv_stop:
            try:
                msg = self._ws.recv()   # 阻塞等待服务器的动作/消息
                if isinstance(msg, (bytes, bytearray)):
                    self.on_message(msgpack_numpy.unpackb(msg))
                else:
                    # server 若发回字符串代表错误（保持与 infer 一致的语义）
                    logging.error(f"Server error: {msg}")
            except websockets.exceptions.ConnectionClosed:
                logging.info("WebSocket connection closed")
                break
            except Exception as e:
                logging.exception(f"Recv loop error: {e}")
                break

    @override
    def reset(self) -> None:
        pass
