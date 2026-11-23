"""CPU-only WebSocket transport comparison; synthetic policy, no simulator/model score."""

import argparse
import asyncio
from collections import deque
import json
from pathlib import Path
import statistics
import time

import anyio
import numpy as np
from vla_eval.model_servers.serve import serve_async

from benchmarks.vla_harness.chunk_server import EvaluationModelServer
from benchmarks.vla_harness.client import EvaluationConnection
from benchmarks.vla_harness.execution import EvaluationCapabilities
from openwam.history import select_condition_images


class TransportProbe(EvaluationModelServer):
    def __init__(self, horizon, offsets, local):
        super().__init__(chunk_size=horizon, max_batch_size=8, max_wait_time=.002)
        key = 'clean_condition_images' if len(offsets) > 1 else None
        self.configure_evaluation(EvaluationCapabilities(horizon, offsets, key), local)
        self.history = {}
        self.requests = self.image_bytes = self.predictions = 0
        self.signatures = []

    async def on_observation(self, obs, ctx):
        self.requests += 1
        self.image_bytes += sum(image.nbytes for image in obs['images'].values())
        for images in obs.get('clean_condition_images', []):
            self.image_bytes += sum(image.nbytes for image in images.values())
        cap = self.evaluation_capabilities
        if not self.client_action_queue and cap.history_key:
            history = self.history.setdefault(ctx.session_id, deque(maxlen=1-cap.history_offsets[0]))
            history.append(obs['images'])
            obs = {**obs, 'clean_condition_images': select_condition_images(history, frame_offsets=cap.history_offsets)}
        await super().on_observation(obs, ctx)

    def predict_batch(self, observations, contexts):
        results = []
        for obs in observations:
            frame = int(obs['images']['head'][0, 0, 0])
            history = tuple(int(images['head'][0, 0, 0]) for images in obs.get('clean_condition_images', []))
            self.signatures.append((obs['prompt'], frame, history))
            self.predictions += 1
            actions = np.zeros((self.evaluation_capabilities.execution_horizon, 7), np.float32)
            actions[:, 0] = frame + np.arange(len(actions))
            results.append({'actions': actions})
        return results

    def get_action_spec(self): return {}
    def get_observation_spec(self): return {}


async def trial(*, sessions, steps, resolution, horizon, offsets, local):
    server = TransportProbe(horizon, offsets, local)
    async with anyio.create_task_group() as tg:
        port = await tg.start(serve_async, server, '127.0.0.1', 0)
        clients = [EvaluationConnection(f'ws://127.0.0.1:{port}') for _ in range(sessions)]
        try:
            for client in clients:
                await client.connect()
                await client.start_episode({})
            started = time.monotonic()
            for step in range(steps):
                observations = [{'images': {
                    'head': np.full((resolution, resolution, 3), step, np.uint8),
                    'wrist': np.full((resolution, resolution, 3), 255-step, np.uint8),
                }, 'prompt': str(index)} for index in range(sessions)]
                replies = await asyncio.gather(*(client.act(obs) for client, obs in zip(clients, observations)))
                if any(reply['actions'][0] != step for reply in replies):
                    raise AssertionError('Control actions differ from the reference sequence')
            elapsed = time.monotonic()-started
            return {'wall_seconds': elapsed, 'control_steps': sessions*steps,
                    'observation_requests': server.requests, 'predictions': server.predictions,
                    'input_image_array_bytes': server.image_bytes,
                    'prediction_signatures': sorted(server.signatures)}
        finally:
            for client in clients:
                await client.end_episode({'success': False})
                await client.close()
            tg.cancel_scope.cancel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sessions', type=int, default=4)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--execution-horizon', type=int, default=10)
    parser.add_argument('--history-offsets', default='0')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--image-format', choices=('raw', 'png'), default='raw')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from vla_eval.protocol.numpy_codec import set_image_format

    set_image_format(args.image_format)
    if min(args.sessions, args.steps, args.resolution, args.execution_horizon, args.repeats) < 1 or args.steps > 250:
        parser.error('Positive sizes required; steps must be <=250 for uint8 frame identifiers')
    offsets = tuple(int(x) for x in args.history_offsets.split(','))
    results = {'legacy': [], 'client_queue': []}
    for repeat in range(args.repeats):
        signatures = []
        # Alternate order to reduce warm filesystem/import scheduling bias.
        for local in ([False, True] if repeat % 2 == 0 else [True, False]):
            value = anyio.run(lambda: trial(sessions=args.sessions, steps=args.steps, resolution=args.resolution,
                                           horizon=args.execution_horizon, offsets=offsets, local=local))
            signatures.append(value.pop('prediction_signatures'))
            results['client_queue' if local else 'legacy'].append(value)
        if signatures[0] != signatures[1]:
            raise AssertionError('History or prediction inputs differ between execution modes')
    report = {'scope': 'Synthetic CPU policy and real WebSocket transport; excludes model/simulator/startup. Not closed-loop evaluation speed.',
              'config': {**vars(args), 'output': str(args.output)}, 'trials': results,
              'prediction_inputs_and_actions_match': True,
              'median_seconds': {name: statistics.median(r['wall_seconds'] for r in rows) for name, rows in results.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
