"""Apply an inference-only multiplier to the learned sigmoid gate on each model replica."""
import json
from pathlib import Path
import sys


def set_gain(router, scale):
    import torch
    if scale not in (0, 1, 2) or not (router.fp32_gate and router.fp32_residual):
        raise ValueError('Expected an FP32 gated residual and a registered multiplier')
    with torch.no_grad():
        logit = router.gate_logit
        if logit.dtype != torch.float32:
            raise ValueError('Unexpected gate dtype')
        before = logit.sigmoid().item()
        target = before * scale
        if not 0 < before < 1 or target >= 1:
            raise ValueError(f'Gate outside the supported multiplier range: {before}, {scale}')
        if scale != 1:
            logit.copy_(torch.logit(logit.new_tensor(target)))
        after = logit.sigmoid().item()
        if abs(after-target) > 1e-6 * max(target, 1e-7):
            raise RuntimeError(f'Effective gate {after} differs from {target}')
    return {'scale': scale, 'native_gate': before, 'effective_gate': after}


def server(scale, receipt):
    from benchmarks.vla_harness.model_server import OpenWAMModelServer
    from vla_eval.model_servers.serve import run_server

    class ScaledServer(OpenWAMModelServer):
        def _load_model(self, *args, **kwargs):
            value = super()._load_model(*args, **kwargs)
            router = self.engine.architecture.action_router
            observed = set_gain(router, scale)

            def first_forward(module, inputs, output):
                receipt.write_text(json.dumps(observed))
                handle.remove()

            handle = router.register_forward_hook(first_forward)
            return value

    run_server(ScaledServer)


def launch(scale):
    from benchmarks.vla_harness import run_eval
    make_plan = run_eval.make_plan

    def plan(args):
        result = make_plan(args)
        for i, item in enumerate(result['servers']):
            cmd = item['command']
            if cmd[1:3] != ['-m', 'benchmarks.vla_harness.model_server']:
                raise ValueError('Unexpected model server')
            cmd[1:3] = [str(Path(__file__).resolve()), 'server', str(scale),
                        str(args.output_dir/f'gain_{i}.json')]
        return result

    run_eval.make_plan = plan
    return run_eval.main()


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    scale = int(sys.argv.pop(1))
    if mode == 'server':
        server(scale, Path(sys.argv.pop(1)))
    elif mode == 'evaluate':
        sys.exit(launch(scale))
    else:
        raise ValueError(mode)
