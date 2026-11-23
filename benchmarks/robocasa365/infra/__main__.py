"""Run from the repository root: python -m benchmarks.robocasa365.infra --help."""
import argparse
import importlib
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL','osmesa')
os.environ.setdefault('PYOPENGL_PLATFORM',os.environ['MUJOCO_GL'])
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID','0')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('env');p.add_argument('--output',type=Path)
    p=sub.add_parser('manifest');p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('download');p.add_argument('--manifest',required=True);p.add_argument('--workers',type=int,default=8)
    p=sub.add_parser('audit');p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--deep',action='store_true');p.add_argument('--workers',type=int,default=4)
    p=sub.add_parser('dry-run');p.add_argument('--manifest',required=True);p.add_argument('--tasks',nargs='+');p.add_argument('--history',type=int,default=3);p.add_argument('--chunk',type=int,default=4);p.add_argument('--batch-size',type=int,default=2);p.add_argument('--workers',type=int,default=0);p.add_argument('--batches',type=int,default=1);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('eval');p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-steps',type=int,default=2);p.add_argument('--trials',type=int,default=50);p.add_argument('--tasks',nargs='+');p.add_argument('--history',type=int,default=1);p.add_argument('--chunk',type=int,default=1);p.add_argument('--seed',type=int,default=0);p.add_argument('--policy',default='dummy',help='dummy or module:factory implementing Policy')
    args=parser.parse_args()
    if args.command=='env':
        from .provenance import verify_environment
        data=verify_environment()
        if args.output:args.output.write_text(json.dumps(data,indent=2))
        print(json.dumps(data,indent=2))
    elif args.command=='manifest':
        from .data_admin import build_manifest
        data=build_manifest(args.output);print(f"Registered {len(data['datasets'])} allowed tasks")
    elif args.command=='download':
        from .data_admin import download_all
        download_all(args.manifest,args.workers)
    elif args.command=='audit':
        from .data_admin import audit_all
        result=audit_all(args.manifest,args.output,deep=args.deep,workers=args.workers)
        print('Complete:',result['complete']);return 0 if result['complete'] else 1
    elif args.command=='dry-run':
        from .dataset import Human300Dataset,make_loader
        from .training import DummyPolicy,dry_run,TrainConfig,save_checkpoint,load_checkpoint
        cfg=TrainConfig(args.history,args.chunk,args.batch_size,args.workers)
        ds=Human300Dataset(args.manifest,history=cfg.history,chunk=cfg.chunk,tasks=args.tasks)
        result=dry_run(DummyPolicy(cfg.chunk),make_loader(ds,batch_size=cfg.batch_size,workers=cfg.workers),max_batches=args.batches)
        args.output.mkdir(parents=True,exist_ok=True)
        path=args.output/'dummy.pt';model=DummyPolicy(cfg.chunk)
        save_checkpoint(path,model,config=cfg,step=0);load_checkpoint(path,model)
        result['checkpoint_roundtrip']=True;result['dataset_frames']=len(ds)
        (args.output/'dry_run.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    elif args.command=='eval':
        from .evaluation import evaluate
        from .training import DummyPolicy
        if args.policy=='dummy':factory=lambda:DummyPolicy(args.chunk)
        else:
            module,name=args.policy.split(':');factory=getattr(importlib.import_module(module),name)
        result=evaluate(args.output,policy_factory=factory,trials=args.trials,history=args.history,seed=args.seed,
                        smoke=args.smoke,smoke_steps=args.smoke_steps,tasks=args.tasks,policy_name=args.policy)
        print(json.dumps(result,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
