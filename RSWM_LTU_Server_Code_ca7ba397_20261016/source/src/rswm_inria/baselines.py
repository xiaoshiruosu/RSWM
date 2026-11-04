"""Diagnostic systems and a shared-window map mechanism for real completers."""
from __future__ import annotations
import copy
import hashlib
import uuid
from pathlib import Path
import numpy as np
from PIL import Image
from .geometry import WORLD, WIDTH, HEIGHT, crop, box, rgb, rng_for, file_hash


class CopyModel:
    def reset(self, initial, spec, seed):
        self.initial=rgb(initial).copy()
    def generate(self, queries):
        return [self.initial.copy() for _ in queries]
    def resource_state(self):
        return {k:v for k,v in self.__dict__.items() if isinstance(v,np.ndarray) and k!="world"}
    def snapshot(self):
        return copy.deepcopy(self.__dict__)
    def restore(self, state):
        self.__dict__=copy.deepcopy(state)


class RandomModel(CopyModel):
    def reset(self, initial, spec, seed):
        super().reset(initial,spec,seed); self.rng=rng_for("random-baseline",seed)
    def generate(self, queries):
        return [self.rng.integers(0,256,(HEIGHT,WIDTH,3),dtype=np.uint8) for _ in queries]


class StaticNoiseMap(CopyModel):
    def reset(self, initial, spec, seed):
        self.map=rng_for("static-noise",seed).integers(0,256,(WORLD,WORLD,3),dtype=np.uint8)
        x0,y0,x1,y1=box((0,0),(1024,1024)); self.map[y0:y1,x0:x1]=rgb(initial)
    def generate(self,queries):
        return [crop(self.map,q,(1024,1024)) for q in queries]


class OracleMap(CopyModel):
    """Evaluator-only. Never put this system in the single-view model ranking."""
    def __init__(self,world,origin):
        self.world=world; self.origin=origin
    def generate(self,queries):
        return [crop(self.world,q,self.origin) for q in queries]


class NoiseCompleter:
    """Diagnostic replacement. NOT a trained remote-sensing generator."""
    def __call__(self,image,known,seed):
        result=rng_for("noise-completer",seed).integers(0,256,image.shape,dtype=np.uint8)
        result[known]=image[known]
        return result


class DiffusersCompleter:
    """Local-only Diffusers inpainting. White mask means fill; output is hard-protected.

    Generic pipelines only: this is not a Genesis/RemoteBAGEL/Wan reproduction.
    The pipeline remains frozen and stateless; each request uses an explicit generator.
    """
    def __init__(self,model_dir,device="cuda",steps=30,prompt="nadir orthorectified RGB aerial photograph"):
        import torch
        from diffusers import AutoPipelineForInpainting
        root=Path(model_dir)
        if not root.is_dir():
            raise ValueError("A downloaded local model directory is required.")
        if not isinstance(steps,int) or steps <= 0:
            raise ValueError("steps must be positive")
        self.weight_manifest={str(p.relative_to(root)):file_hash(p) for p in sorted(root.rglob("*")) if p.is_file()}
        self.pipe=AutoPipelineForInpainting.from_pretrained(str(root),local_files_only=True,
                   torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32).to(device)
        self.device,self.steps,self.prompt=device,steps,prompt
    def __call__(self,image,known,seed):
        import torch
        result=self.pipe(prompt=self.prompt,image=Image.fromarray(image),
                    mask_image=Image.fromarray((~known).astype(np.uint8)*255),
                    height=image.shape[0],width=image.shape[1],num_inference_steps=self.steps,
                    generator=torch.Generator(device=self.device).manual_seed(int(seed))).images[0]
        if result.mode != "RGB":
            raise ValueError("Pipeline did not return RGB.")
        out=rgb(np.asarray(result),image.shape).copy();out[known]=image[known]
        return out


class WindowMap(CopyModel):
    """Same 256 core / 512 context window for eager and lazy map generation.

    A full RGB array is allocated in BOTH modes; do not claim sparse memory.
    Dynamic snapshots do not duplicate frozen model weights.
    """
    def __init__(self,completer=None,eager=False):
        self.completer=completer if completer is not None else NoiseCompleter();self.eager=eager
        self.resource_calls=0;self.resource_generated_pixels=0
        self.resource_counter_epoch=uuid.uuid4().hex
    def reset(self,initial,spec,seed):
        self.map=np.zeros((WORLD,WORLD,3),np.uint8);self.known=np.zeros((WORLD,WORLD),bool)
        self.seed=int(seed);self.calls=0;self.generated_pixels=0
        x0,y0,x1,y1=box((0,0),(1024,1024)); self.map[y0:y1,x0:x1]=rgb(initial);self.known[y0:y1,x0:x1]=True
        if self.eager:
            for iy in range(8):
                for ix in range(8): self._fill(ix,iy)
    def _fill(self,ix,iy):
        x,y=ix*256,iy*256
        if self.known[y:y+256,x:x+256].all(): return
        # Zero padding is private state, not an extra real observation.
        image=np.zeros((512,512,3),np.uint8);mask=np.zeros((512,512),bool)
        xa,ya,xb,yb=max(0,x-128),max(0,y-128),min(WORLD,x+384),min(WORLD,y+384)
        dest=np.s_[ya-(y-128):yb-(y-128),xa-(x-128):xb-(x-128)]
        image[dest]=self.map[ya:yb,xa:xb];mask[dest]=self.known[ya:yb,xa:xb]
        seed=int.from_bytes(hashlib.sha256(f"{self.seed}/{ix}/{iy}".encode()).digest()[:8],"big") % (2**63-1)
        self.resource_calls+=1
        output=rgb(self.completer(image,mask,seed),(512,512,3)).copy()
        unknown=~self.known[y:y+256,x:x+256]
        self.map[y:y+256,x:x+256][unknown]=output[128:384,128:384][unknown]
        self.known[y:y+256,x:x+256]=True;self.calls+=1
        self.generated_pixels+=int(unknown.sum())
        self.resource_generated_pixels+=int(unknown.sum())
    def generate(self,queries):
        out=[]
        for q in queries:
            x0,y0,x1,y1=box(q,(1024,1024))
            for iy in range(y0//256,(y1-1)//256+1):
                for ix in range(x0//256,(x1-1)//256+1): self._fill(ix,iy)
            out.append(crop(self.map,q,(1024,1024)))
        return out
    def resource_counters(self):
        return {"completion_calls":self.resource_calls,"generated_pixels":self.resource_generated_pixels}
    def snapshot(self):
        return {key:copy.deepcopy(getattr(self,key)) for key in ("map","known","seed","calls","generated_pixels")}
    def restore(self,state):
        for key,value in state.items(): setattr(self,key,copy.deepcopy(value))
