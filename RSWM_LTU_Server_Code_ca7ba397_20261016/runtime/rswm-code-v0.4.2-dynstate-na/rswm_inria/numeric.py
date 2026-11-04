"""Camera-conditioned last-frame adapter for an explicitly approved backend.

This implements the public wrapper; the selected backend owns the camera backbone.
Mock output validates transport/state only, never real numerical conditioning.
"""
import copy
import hashlib
import uuid
import numpy as np
from PIL import Image
from .geometry import rgb, rng_for, WIDTH, HEIGHT, digest
from .adapters.camera_conditioning import CameraConfig, build_camera_batch, backend_payload, to_native_rgb, from_native_rgb


def payload_hash(payload):
    encoded={}
    for key,value in payload.items():
        if isinstance(value,np.ndarray):
            encoded[key]={'shape':list(value.shape),'dtype':value.dtype.str,'sha256':hashlib.sha256(value.tobytes()).hexdigest()}
        else:encoded[key]=value
    return digest(encoded)


class NumericLastFrameAdapter:
    def __init__(self, backend, camera=None, *, expected_camera_sha256=None):
        self.backend=backend;self.camera=camera or CameraConfig()
        self.camera_sha256=digest(self.camera.manifest())
        if expected_camera_sha256 is not None and expected_camera_sha256!=self.camera_sha256:
            raise ValueError('Training/inference camera contract mismatch')
        for method in ('generate','snapshot','restore'):
            if not callable(getattr(backend,method,None)):raise ValueError('Backend state protocol incomplete')
        self.resource_counter_epoch=uuid.uuid4().hex
        self.counters={'returned_native_frames':0,'context_frames':0,'submitted_native_frames':0,'discarded_frames':0,'camera_calls':0,'terminated_calls':0}
        self._backend_epoch=None;self._backend_epoch_invalid=False
        self.ready=False

    def reset(self,initial,spec,seed):
        self.last=rgb(initial).copy();self.anchor=np.zeros(2,dtype=np.int64)
        self.rng=rng_for('numeric-lastframe',int(seed));self.call_index=0;self.terminated=False;self.ready=True
        if callable(getattr(self.backend,'reset',None)):
            self.backend.reset(seed=int(seed))

    def generate(self,queries):
        if not self.ready:raise RuntimeError('Not reset')
        q=np.asarray(queries)
        if len(q) not in (16,32):raise ValueError('Complete online16/offline32 call required')
        if self.terminated:
            self.counters['terminated_calls']+=1
            return [None]*len(q)
        camera=backend_payload(build_camera_batch(q,self.anchor,self.camera),expected_config_sha256=self.camera_sha256)
        seed=int(self.rng.integers(0,2**63-1));self.counters['camera_calls']+=1
        result=self.backend.generate(image=to_native_rgb(self.last),camera=camera,
                                     native_frames=len(q)+1,seed=seed)
        if not isinstance(result,dict) or not isinstance(result.get('frames'),list):raise ValueError('Backend native frame contract invalid')
        native=result['frames'];self.counters['returned_native_frames']+=len(native)
        self.counters['context_frames']+=min(1,len(native))
        # A backend receipt is an integration check; actual layer consumption
        # still needs its own real weight/backend audit.
        if result.get('camera_payload_sha256')!=payload_hash(camera):
            self.counters['discarded_frames']+=max(0,len(native)-1)
            raise ValueError('Backend camera receipt mismatch')
        if len(native)!=len(q)+1:
            self.counters['discarded_frames']+=max(0,len(native)-1)
            raise ValueError('Unexpected actual native frame count')
        frames=[]
        try:
            for frame in native[1:]:
                if frame is None:frames.append(None);continue
                if not isinstance(frame,Image.Image) or frame.mode!='RGB' or frame.size!=(2*WIDTH,2*HEIGHT):
                    raise ValueError('Expected native lossless PIL RGB geometry')
                frames.append(from_native_rgb(np.asarray(frame)).copy())
        except Exception:
            self.counters['discarded_frames']+=len(q)
            raise
        self.counters['submitted_native_frames']+=sum(f is not None for f in frames)
        self.counters['discarded_frames']+=sum(f is None for f in frames)
        self.call_index+=1
        if frames[-1] is None:
            self.terminated=True
        else:
            self.last=frames[-1].copy();self.anchor=np.array(q[-1],dtype=np.int64)
        return frames

    def snapshot(self):
        return {'last':self.last.copy(),'anchor':self.anchor.copy(),'rng':copy.deepcopy(self.rng.bit_generator.state),
                'call_index':self.call_index,'terminated':self.terminated,'backend':copy.deepcopy(self.backend.snapshot())}

    def restore(self,state):
        self.last=state['last'].copy();self.anchor=state['anchor'].copy();self.rng.bit_generator.state=copy.deepcopy(state['rng'])
        self.call_index=state['call_index'];self.terminated=state['terminated'];self.ready=True
        self.backend.restore(copy.deepcopy(state['backend']))

    def resource_counters(self):
        # Returned frames cannot account for native work performed before an
        # exception or discarded inside the backbone. Require its own lifetime
        # counter contract; absence stays unknown even after a successful call.
        try:
            raw=self.backend.resource_counters()
            if not isinstance(raw,dict):raise ValueError('Backend counters unavailable')
            values=raw.get('values',raw)
            epoch=raw.get('counter_epoch',getattr(self.backend,'resource_counter_epoch',None))
            if raw.get('status','ok')!='ok' or type(epoch) not in (str,int) or epoch=='':raise ValueError('Backend counters untrusted')
            if not isinstance(values,dict) or 'native_frames' not in values or any(not isinstance(k,str) or type(v) is not int or v<0 for k,v in values.items()):
                raise ValueError('Backend actual native work missing')
            if self._backend_epoch is None:self._backend_epoch=epoch
            elif self._backend_epoch!=epoch:self._backend_epoch_invalid=True
            if self._backend_epoch_invalid:raise ValueError('Backend lifetime counter epoch changed')
            measured={**self.counters,'native_frames':values['native_frames'],**{'backend_'+k:v for k,v in values.items() if k!='native_frames'}}
            return {'status':'ok','counter_epoch':self.resource_counter_epoch,'values':measured}
        except Exception:
            return {'status':'read_failed','values':dict(self.counters),'error_code':'BACKEND_NATIVE_WORK_UNOBSERVED'}

    def resource_state(self):
        # Unknown writable backend state cannot be presented as measured zero.
        backend=self.backend.resource_state() if callable(getattr(self.backend,'resource_state',None)) else object()
        return {'last':getattr(self,'last',None),'anchor':getattr(self,'anchor',None),'backend':backend}

    def generation_evidence(self):
        """Retain backend call observations separately from scores and F7 state.

        An optional observer is telemetry, not proof of real-model acceptance.
        Its lifecycle log must survive restoring a branch's output state.
        """
        method=getattr(self.backend,'generation_evidence',None)
        return copy.deepcopy(method()) if callable(method) else None

    def capabilities(self):
        return {'offline32':True,'online16':True,'snapshot':True,'numerical_input':self.camera.camera_channel,
                'camera_manifest_sha256':self.camera_sha256,'execution_mode':'cooperative_inprocess',
                'real_model_validation':'NOT_RUN','real_backend_layer_consumption':'NOT_RUN'}
