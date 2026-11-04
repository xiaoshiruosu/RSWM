"""Local Wan I2V adapter: fixed text, NO numerical coordinate conditioning.

The adapter returns 16 or 32 new frames after one context frame. Actual pretrained
weights and hardware must be validated separately; mock tests only verify wiring.
"""
import copy
import uuid
from pathlib import Path
import numpy as np
from PIL import Image
from .geometry import WIDTH,HEIGHT,rgb,rng_for,file_hash


class WanI2VAdapter:
    def __init__(self,model_dir=None,device="cuda",steps=30,pipe=None,prompt=None):
        self.device,self.steps=device,steps
        self.resource_native_frames=0
        self.resource_context_frames=0
        self.resource_discarded_frames=0
        self.resource_submitted_native_frames=0
        self.resource_counter_epoch=uuid.uuid4().hex
        self.prompt=prompt if prompt is not None else "A continuous static nadir-view RGB aerial scene, orthographic viewpoint, no changing objects."
        self.weight_manifest={}
        if pipe is None:
            import torch
            from diffusers import WanImageToVideoPipeline
            root=Path(model_dir or "")
            if not model_dir or not root.is_dir():raise ValueError("Downloaded local Wan I2V weights required.")
            self.weight_manifest={str(p.relative_to(root)):file_hash(p) for p in sorted(root.rglob("*")) if p.is_file()}
            pipe=WanImageToVideoPipeline.from_pretrained(str(root),local_files_only=True,
                         torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32).to(device)
        self.pipe=pipe
    def reset(self,initial,spec,seed):
        self.last=rgb(initial).copy();self.rng=rng_for("wan-i2v",int(seed));self.native_frames=0
    def generate(self,queries):
        import torch
        count=len(queries)
        if count not in (16,32):raise ValueError("Wan reference adapter supports 17/33 native frames only.")
        seed=int(self.rng.integers(0,2**63-1))
        image=Image.fromarray(self.last).resize((832,480),Image.Resampling.BILINEAR)
        output=self.pipe(image=image,prompt=self.prompt,height=480,width=832,num_frames=count+1,
                         num_inference_steps=self.steps,output_type="pil",
                         generator=torch.Generator(device=self.device).manual_seed(seed)).frames[0]
        self.resource_native_frames+=len(output)
        self.resource_context_frames+=min(1,len(output))
        if len(output)!=count+1:
            self.resource_discarded_frames+=max(0,len(output)-1)
            raise ValueError("Unexpected native frame count.")
        frames=[]
        try:
            for frame in output[1:]:
                if not isinstance(frame,Image.Image) or frame.mode!="RGB":raise ValueError("Expected PIL RGB frames.")
                frames.append(rgb(np.asarray(frame.resize((WIDTH,HEIGHT),Image.Resampling.BILINEAR))).copy())
        except Exception:
            # The batch is rejected as a whole; generated native frames still cost work.
            self.resource_discarded_frames+=count
            raise
        self.last=frames[-1].copy();self.native_frames+=count+1
        self.resource_submitted_native_frames+=count
        return frames
    def resource_counters(self):
        return {"native_frames":self.resource_native_frames,
                "context_frames":self.resource_context_frames,
                "discarded_frames":self.resource_discarded_frames,
                "submitted_native_frames":self.resource_submitted_native_frames}
    def resource_state(self):
        return {"last":getattr(self,"last",None)}
    def snapshot(self):
        return {"last":self.last.copy(),"rng_state":copy.deepcopy(self.rng.bit_generator.state),"native_frames":self.native_frames}
    def restore(self,state):
        self.last=state["last"].copy();self.rng.bit_generator.state=copy.deepcopy(state["rng_state"])
        self.native_frames=state["native_frames"]
