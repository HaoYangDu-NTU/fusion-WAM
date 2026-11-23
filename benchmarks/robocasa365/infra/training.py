"""Model-independent contracts. CLI only exposes a forward-only dry run."""
from dataclasses import dataclass, asdict
from pathlib import Path
import random
from typing import Protocol
import numpy as np
import torch
from .schema import SCHEMA_VERSION

@dataclass(frozen=True)
class TrainConfig:
    history: int = 1
    chunk: int = 16
    batch_size: int = 2
    workers: int = 0
    schema_version: str = SCHEMA_VERSION

class TrainablePolicy(Protocol):
    def compute_loss(self, batch: dict) -> dict[str,torch.Tensor]:
        """Return scalar loss under 'loss'; honor batch['action_valid']."""
        ...
    def train(self,mode: bool=True): ...
    def eval(self): ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self,state: dict): ...

class DummyPolicy(torch.nn.Module):
    def __init__(self,chunk=1):
        super().__init__()
        if chunk<1:raise ValueError('chunk must be positive')
        self.chunk=chunk
    def reset(self):pass
    def predict(self, observation):return np.zeros((self.chunk,12),np.float32)
    def forward(self, observation):
        state=observation['state']
        return torch.zeros((state.shape[0],self.chunk,12),device=state.device,dtype=state.dtype)
    def compute_loss(self,batch):
        error=(self(batch['observation'])-batch['action']).square().mean(-1)
        valid=batch['action_valid']
        return {'loss':error[valid].mean()}

class TrainingPipeline:
    def __init__(self,model,optimizer=None):self.model,self.optimizer=model,optimizer
    def step(self,batch,*,update=False):
        if update:
            if self.optimizer is None:raise ValueError('Updating requires an explicit optimizer')
            self.model.train();self.optimizer.zero_grad(set_to_none=True)
            losses=self.model.compute_loss(batch)
            self._validate_loss(losses)
            losses['loss'].backward();self.optimizer.step()
        else:
            self.model.eval()
            with torch.no_grad():losses=self.model.compute_loss(batch)
        self._validate_loss(losses)
        return {k:float(v.detach()) for k,v in losses.items()}

    @staticmethod
    def _validate_loss(losses):
        if losses['loss'].ndim!=0 or not torch.isfinite(losses['loss']):
            raise ValueError('Model must return a finite scalar loss')


def dry_run(model,loader,*,max_batches=1):
    if max_batches<1:raise ValueError('max_batches must be positive')
    pipeline=TrainingPipeline(model)
    result={'mode':'dry_run_no_updates','batches':0,'valid_action_steps':0,'losses':[]}
    for batch in loader:
        if result['batches']==0:
            result['first_batch']={'state_shape':list(batch['observation']['state'].shape),
                                   'action_shape':list(batch['action'].shape),
                                   'rgb_shapes':{k:list(v.shape) for k,v in batch['observation'].get('rgb',{}).items()},
                                   'state_dtype':str(batch['observation']['state'].dtype),
                                   'action_dtype':str(batch['action'].dtype)}
        result['losses'].append(pipeline.step(batch,update=False))
        result['batches']+=1;result['valid_action_steps']+=int(batch['action_valid'].sum())
        if result['batches']>=max_batches:break
    if result['batches']==0:raise ValueError('Empty loader')
    return result


def save_checkpoint(path,model,*,config,step,optimizer=None):
    """Trusted local checkpoints only. Atomic save; model owns its state format."""
    payload={'schema_version':SCHEMA_VERSION,'config':asdict(config) if isinstance(config,TrainConfig) else config,
             'step':int(step),'model':model.state_dict(),
             'optimizer':optimizer.state_dict() if optimizer else None,
             'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state()}}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');torch.save(payload,tmp);tmp.replace(path)


def load_checkpoint(path,model,optimizer=None,*,restore_rng=False):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload['schema_version']!=SCHEMA_VERSION:raise ValueError('Incompatible checkpoint schema')
    model.load_state_dict(payload['model'])
    if optimizer is not None:
        if payload['optimizer'] is None:raise ValueError('Checkpoint has no optimizer')
        optimizer.load_state_dict(payload['optimizer'])
    if restore_rng:
        random.setstate(payload['rng']['python']);np.random.set_state(payload['rng']['numpy']);torch.set_rng_state(payload['rng']['torch'])
    return payload
