"""Exercise RoboLab's OpenPI client over loopback without booting a simulator."""
import json
import threading

import numpy as np
import torch
from openpi_client import msgpack_numpy
from policies.pi0_family.client import Pi0DroidJointposClient
from websockets.sync.server import serve


def main():
    calls = []
    horizon = 8

    def handler(ws):
        ws.send(msgpack_numpy.packb({"policy_name": "smoke-hold"}))
        for raw in ws:
            obs = msgpack_numpy.unpackb(raw)
            assert obs["prompt"] == "pick up the object"
            for camera in ["exterior_image_1_left", "wrist_image_left"]:
                rgb = obs["observation/" + camera]
                assert rgb.shape == (224, 224, 3) and rgb.dtype == np.uint8
            joints = obs["observation/joint_position"]
            grip = obs["observation/gripper_position"]
            assert joints.shape == (7,) and grip.shape == (1,)
            calls.append(float(joints[0]))
            actions = np.tile(np.concatenate([joints, grip]), (horizon, 1))
            ws.send(msgpack_numpy.packb({"actions": actions}))

    with serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            client = Pi0DroidJointposClient(remote_host="127.0.0.1", remote_port=server.socket.getsockname()[1], open_loop_horizon=horizon)
            rgb = torch.zeros((1, 180, 320, 3), dtype=torch.uint8)
            obs = {"image_obs": {"over_shoulder_left_camera": rgb, "wrist_cam": rgb},
                   "proprio_obs": {"gripper_pos": torch.tensor([[0.75]])}}
            for step in range(16):
                obs["proprio_obs"]["arm_joint_pos"] = torch.full((1, 7), step / 16)
                action = client.infer(obs, "pick up the object")["action"]
                assert np.array_equal(action, [float(step >= 8) / 2] * 7 + [1.0])
            client.reset()
            client.infer(obs, "pick up the object")
            assert calls == [0.0, 0.5, 15 / 16]
            client.client._ws.close()
        finally:
            server.shutdown()
            thread.join()
    print(json.dumps({"source": "synthetic", "steps_before_reset": 16, "queries_before_reset": 2,
                      "first_query_after_reset": True, "action_chunk": [8, 8]}))


if __name__ == "__main__":
    main()
