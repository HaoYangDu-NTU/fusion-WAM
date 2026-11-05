"""Real transport tests for shared chunks, history and per-step runner semantics."""

import asyncio
from collections import deque
import json
from types import SimpleNamespace

import anyio
import numpy as np
import pytest
from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.connection import Connection
from vla_eval.model_servers.serve import serve_async
from vla_eval.runners.sync_runner import SyncEpisodeRunner

from benchmarks.vla_harness.chunk_server import EvaluationModelServer
from benchmarks.vla_harness.client import EvaluationConnection, install_evaluation_connection
from benchmarks.vla_harness.execution import EvaluationCapabilities, PhaseStats
from openwam.history import select_condition_images


class ToyServer(EvaluationModelServer):
    def __init__(self, cap, local, batch=4):
        super().__init__(chunk_size=cap.execution_horizon, max_batch_size=batch, max_wait_time=.02)
        self.configure_evaluation(cap, local)
        self.histories = {}
        self.seen = []
        self.batches = []

    async def on_episode_start(self, config, ctx):
        self.histories.pop(ctx.session_id, None)
        await super().on_episode_start(config, ctx)

    async def on_observation(self, obs, ctx):
        cap = self.evaluation_capabilities
        if not self.client_action_queue and cap.history_key:
            history = self.histories.setdefault(ctx.session_id, deque(maxlen=1 - cap.history_offsets[0]))
            history.append(obs['images'])
            frames = select_condition_images(history, frame_offsets=cap.history_offsets)
            obs = {**obs, cap.history_key: frames[0] if cap.history_key == 'history_images' else frames}
        await super().on_observation(obs, ctx)

    def predict_batch(self, observations, contexts):
        self.batches.append(len(observations))
        results = []
        for obs in observations:
            frame = int(obs['images']['camera'][0, 0, 0])
            self.seen.append((frame, obs.get(self.evaluation_capabilities.history_key)))
            actions = np.zeros((self.evaluation_capabilities.execution_horizon + 3, 7), np.float32)
            actions[:, 0] = frame + np.arange(len(actions))
            results.append({'actions': actions})
        return results

    def get_action_spec(self):
        return {}

    def get_observation_spec(self):
        return {}


def observation(value):
    return {'images': {'camera': np.full((4, 4, 3), value, np.uint8)}, 'prompt': 'pick', 'state': np.zeros(1)}


@pytest.mark.parametrize('horizon,key,offsets', [
    (10, None, (0,)), (16, 'history_images', (-4, 0)),
    (10, 'clean_condition_images', (-4, 0)),
    (10, 'clean_condition_images', (-12, -8, -4, 0)),
])
def test_real_websocket_matches_legacy_and_batches_sessions(horizon, key, offsets):
    cap = EvaluationCapabilities(horizon, offsets, key)

    async def run(local):
        server = ToyServer(cap, local)
        async with anyio.create_task_group() as tg:
            port = await tg.start(serve_async, server, '127.0.0.1', 0)
            client_cls = EvaluationConnection if local else Connection
            clients = [client_cls(f'ws://127.0.0.1:{port}', timeout=5) for _ in range(3)]
            try:
                for client in clients:
                    await client.connect()
                    await client.start_episode({})
                for step in range(horizon + 3):
                    responses = await asyncio.gather(*(c.act(observation(i * 50 + step))
                                                       for i, c in enumerate(clients)))
                    for i, response in enumerate(responses):
                        assert response['actions'].shape == (7,)
                        assert response['actions'][0] == i * 50 + step
                # End halfway through a chunk and reuse the same connection.
                await clients[0].end_episode({'success': False})
                await clients[0].start_episode({})
                response = await clients[0].act(observation(200))
                assert response['actions'][0] == 200
                assert max(server.batches) > 1
                if local:
                    assert [c.performance.counts['request_roundtrip'] for c in clients] == [3, 2, 2]
                history_values = []
                for frame, history in server.seen:
                    if key:
                        frames = [history] if key == 'history_images' else history
                        history_values.append((frame, tuple(int(f['camera'][0, 0, 0]) for f in frames)))
                return sorted(history_values)
            finally:
                for client in clients:
                    await client.end_episode({'success': False})
                    await client.close()
                with anyio.fail_after(2):
                    while server._execution_sessions or server._chunk_buffers:
                        await anyio.sleep(.01)
                tg.cancel_scope.cancel()

    assert anyio.run(run, True) == anyio.run(run, False)


@pytest.mark.parametrize('max_steps,done_after', [(7, 100), (30, 13)])
def test_real_sync_runner_counts_control_steps_and_stops_mid_chunk(max_steps, done_after):
    class Env(StepBenchmark):
        def get_tasks(self): return []
        def reset(self, task):
            self.actions = []
            return observation(0)
        def make_obs(self, obs, task): return obs
        def step(self, action):
            self.actions.append(action['actions'][0])
            n = len(self.actions)
            return StepResult(obs=observation(n), reward=0., done=n >= done_after, info={})
        def get_step_result(self, result): return {'success': bool(result.done)}

    async def run():
        server = ToyServer(EvaluationCapabilities(10), True)
        async with anyio.create_task_group() as tg:
            port = await tg.start(serve_async, server, '127.0.0.1', 0)
            client = EvaluationConnection(f'ws://127.0.0.1:{port}')
            await client.connect()
            try:
                env = Env()
                result = await SyncEpisodeRunner().run_episode(env, {}, client, max_steps=max_steps)
                n = min(max_steps, done_after)
                assert result['steps'] == n
                np.testing.assert_array_equal(env.actions, np.arange(n))
                assert result['metrics']['success'] == (done_after <= max_steps)
                assert client.performance.counts['request_roundtrip'] == (n + 9) // 10
                assert not client._actions and not client._history
            finally:
                await client.close()
                tg.cancel_scope.cancel()
    anyio.run(run)


def test_rejects_incompatible_capabilities_and_preserves_unbatched_fallback():
    with pytest.raises(ValueError, match='intermediate'):
        ToyServer(EvaluationCapabilities(10, requires_step_observe=True), True)
    with pytest.raises(ValueError, match='batching'):
        ToyServer(EvaluationCapabilities(1, native_batch=False), False)
    ToyServer(EvaluationCapabilities(1, native_batch=False, requires_step_observe=True), False, batch=1)
    with pytest.raises(ValueError):
        EvaluationCapabilities(10, (-4, 0))
    with pytest.raises(ValueError):
        EvaluationCapabilities.from_dict({'execution_horizon': 10, 'protocol_version': 2})


def test_stale_chunk_and_unsafe_reconnect_are_rejected(monkeypatch):
    async def connect(self, **kwargs):
        self.server_info = {'observation_params': {'evaluation_capabilities': {
            'model': EvaluationCapabilities(10).to_dict(), 'client_action_queue': True}}}
    async def noop(*args): pass
    async def stale(*args):
        return {'actions': np.zeros((10, 7)), '_execution': {'episode': 'old', 'step': 0}}
    monkeypatch.setattr(Connection, 'connect', connect)
    monkeypatch.setattr(Connection, 'start_episode', noop)
    monkeypatch.setattr(Connection, 'act', stale)
    connected = [True]
    monkeypatch.setattr(EvaluationConnection, 'is_connected', property(lambda _: connected[0]))
    async def run():
        client = EvaluationConnection('unused')
        await client.connect()
        await client.start_episode({})
        with pytest.raises(RuntimeError, match='Stale'):
            await client.act(observation(0))
        connected[0] = False
        with pytest.raises(ConnectionError, match='lost'):
            await client.act(observation(0))
    anyio.run(run)


def test_history_snapshots_recycled_simulator_buffers(monkeypatch):
    async def noop(*args): pass
    cap = EvaluationCapabilities(2, (-1, 0), 'clean_condition_images')
    captured = []
    async def respond(self, obs):
        captured.append(obs)
        return {'actions': np.zeros((2, 7)), '_execution': obs['_execution']}
    monkeypatch.setattr(Connection, 'start_episode', noop)
    monkeypatch.setattr(Connection, 'act', respond)
    monkeypatch.setattr(EvaluationConnection, 'is_connected', property(lambda _: True))
    async def run():
        client = EvaluationConnection('unused')
        client._capabilities, client._local_chunks = cap, True
        client._history = deque(maxlen=2)
        await client.start_episode({})
        obs = observation(0)
        for i in range(3):
            obs['images']['camera'].fill(i)
            await client.act(obs)
        assert [int(frame['camera'][0, 0, 0]) for frame in captured[-1]['clean_condition_images']] == [1, 2]
    anyio.run(run)


def test_phase_statistics_written_without_changing_process_environment(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENWAM_PERFORMANCE_DIR', str(tmp_path))
    stats = PhaseStats('test')
    with stats.measure('work', count=3):
        pass
    stats.emit(episode='e')
    row = json.loads(next(tmp_path.glob('test-*.jsonl')).read_text())
    assert row['counts']['work'] == 3 and row['seconds']['work'] >= 0
    assert row['episode'] == 'e'


def test_summary_uses_last_cumulative_snapshot_and_actual_microbatches(tmp_path, monkeypatch):
    from benchmarks.vla_harness.performance import summarize_performance

    monkeypatch.setenv('OPENWAM_PERFORMANCE_DIR', str(tmp_path))
    stats = PhaseStats('client')
    stats.add('request_roundtrip', 1)
    stats.add('control_actions', count=10)
    stats.emit()
    stats.add('request_roundtrip', 2)
    stats.add('control_actions', count=10)
    stats.emit()
    model = PhaseStats('model')
    model.add('batch_size_8', count=3)
    model.add('batch_size_4', count=1)
    model.emit()
    # Ignore a partially written live snapshot.
    with next(tmp_path.glob('model-*.jsonl')).open('a') as handle:
        handle.write('{')
    result = summarize_performance(tmp_path)
    assert result['control_actions_per_request'] == 10
    assert result['roles']['client']['seconds']['request_roundtrip'] == 3
    assert result['mean_actual_batch'] == 7
    assert result['actual_batch_histogram'] == {4: 1, 8: 3}


def test_worker_selects_local_client_without_editing_dependency(monkeypatch):
    from vla_eval import orchestrator
    monkeypatch.setattr(orchestrator, 'Connection', Connection)
    install_evaluation_connection()
    assert orchestrator.Connection is EvaluationConnection


@pytest.mark.parametrize('family,horizon', [('fm', 16), ('vpp2', 10)])
def test_native_adapter_uses_shared_transport_with_original_horizon(monkeypatch, family, horizon):
    from benchmarks.vla_harness import fm_model_server, vpp2_model_server

    captured = []
    def infer(items):
        captured.extend(items)
        return [{'actions': np.tile(np.arange(7, dtype=np.float32), (32, 1))} for _ in items]
    if family == 'fm':
        def load(self, *args):
            self.policy = SimpleNamespace(infer_batch=infer)
        monkeypatch.setattr(fm_model_server.FMVJEPAModelServer, '_load', load)
        monkeypatch.setattr(fm_model_server.FMVJEPAModelServer, 'native_observation', staticmethod(lambda obs: obs))
        server = fm_model_server.FMVJEPAModelServer('unused', 'unused', 'unused', 'unused',
                                                   client_action_queue=True)
    else:
        monkeypatch.setattr(vpp2_model_server, 'VPP2Policy', lambda *args: SimpleNamespace(predict_batch=infer))
        server = vpp2_model_server.VPP2ModelServer('unused', client_action_queue=True)

    async def run():
        async with anyio.create_task_group() as tg:
            port = await tg.start(serve_async, server, '127.0.0.1', 0)
            client = EvaluationConnection(f'ws://127.0.0.1:{port}')
            await client.connect()
            await client.start_episode({})
            try:
                for step in range(horizon + 1):
                    reply = await client.act(observation(step))
                    np.testing.assert_array_equal(reply['actions'], np.arange(7, dtype=np.float32))
                assert len(captured) == 2
                assert int(captured[-1]['images']['camera'][0, 0, 0]) == horizon
                if family == 'fm':
                    assert int(captured[-1]['history_images']['camera'][0, 0, 0]) == horizon - 4
                assert client.performance.counts['request_roundtrip'] == 2
                await client.end_episode({'success': False})
            finally:
                await client.close()
                tg.cancel_scope.cancel()
    try:
        anyio.run(run)
    finally:
        server._thread.shutdown()


def test_launcher_enables_chunks_for_every_backend(tmp_path):
    from pathlib import Path
    from benchmarks.vla_harness.run_eval import make_plan

    for family in ('openwam', 'fm-vjepa', 'vpp2'):
        args = SimpleNamespace(
            policy_backend=family, fm_python=Path('/bin/true'), fm_encoder=tmp_path, fm_text_cache=tmp_path,
            gpus='0', suites='libero_spatial', task_ids='0', smoke=True, workers=1,
            max_batch_size=2, max_wait_time=.01, render='gpu', render_gpus='0', base_port=15940,
            ckpt_dir=tmp_path, ckpt_name='model', output_dir=tmp_path, eval_id='test', client_action_queue=True,
        )
        plan = make_plan(args)
        assert plan['client_action_queue'] is True
        command = plan['servers'][0]['command']
        assert command[command.index('--args.client_action_queue') + 1] == 'true'


def test_server_rejects_replayed_step_and_wrong_episode_token():
    server = ToyServer(EvaluationCapabilities(10), True, batch=1)
    sent = []
    async def send(reply): sent.append(reply)
    ctx = SimpleNamespace(session_id='s', episode_id='e', send_action=send, step=0)
    async def run():
        await server.on_episode_start({}, ctx)
        meta = {'episode': 'token', 'step': 0}
        await server.on_observation({**observation(0), '_execution': meta}, ctx)
        assert sent[0]['_execution'] == meta
        for invalid in (meta, {'episode': 'old', 'step': 10}):
            with pytest.raises(RuntimeError, match='control step'):
                await server.on_observation({**observation(10), '_execution': invalid}, ctx)
        await server.on_episode_end({}, ctx)
        with pytest.raises(RuntimeError, match='control step'):
            await server.on_observation({**observation(0), '_execution': meta}, ctx)
    anyio.run(run)


@pytest.mark.parametrize('enabled', [False, True])
def test_resume_identity_preserves_default_and_records_opt_in(tmp_path, monkeypatch, enabled):
    import hashlib
    import sys
    from benchmarks.vla_harness import run_resumable as resume

    (tmp_path / 'config.yaml').write_text('model: {}\n')
    weights = tmp_path / 'model.safetensors'
    weights.write_bytes(b'weights')
    settings = {
        'ckpt_dir': str(tmp_path), 'ckpt_name': weights.name, 'output_dir': str(tmp_path),
        'gpus': '0,1,2,3', 'workers': 32, 'max_batch_size': 8, 'render_gpus': '3',
        'base_port': 8940, 'retry_delay': 30, 'correct_instruction_suffix': True,
        'config_sha256': hashlib.sha256((tmp_path / 'config.yaml').read_bytes()).hexdigest(),
        'weight_size_mtime': [weights.stat().st_size, weights.stat().st_mtime_ns],
    }
    if enabled:
        settings['client_action_queue'] = True
    (tmp_path / 'job.json').write_text(json.dumps({'job_id': 'existing', 'settings': settings}))
    arguments = ['resume', '--ckpt-dir', str(tmp_path), '--ckpt-name', weights.name, '--output-dir', str(tmp_path)]
    if enabled:
        arguments.append('--client-action-queue')
    monkeypatch.setattr(sys, 'argv', arguments)
    monkeypatch.setattr(resume, 'stop_owned_processes', lambda _: None)
    stages = []
    monkeypatch.setattr(resume, 'run_stage', lambda a, j, b: stages.append(b) or True)
    resume.main()
    assert stages == ['libero', 'libero-plus']
    assert json.loads((tmp_path / 'job.json').read_text())['settings'] == settings
