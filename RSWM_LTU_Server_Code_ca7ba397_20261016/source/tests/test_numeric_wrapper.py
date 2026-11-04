"""Explicit synthetic backend; no assertion of pretrained layer consumption."""
import copy
import numpy as np
from PIL import Image
import pytest
from rswm_inria.numeric import NumericLastFrameAdapter, payload_hash
from rswm_inria.adapters.camera_conditioning import CameraConfig


class ControlBackend:
    def __init__(self):self.calls=[];self.writable=0;self.missing=False;self.bad_receipt=False
    def reset(self,seed):self.writable=0
    def generate(self,**request):
        self.calls.append(request);self.writable+=1
        frames=[Image.new('RGB',(832,480),(self.writable,2,3)) for _ in range(request['native_frames'])]
        if self.missing:frames[-1]=None
        return {'frames':frames,'camera_payload_sha256':'wrong' if self.bad_receipt else payload_hash(request['camera'])}
    def snapshot(self):return {'writable':self.writable}
    def restore(self,state):self.writable=state['writable']
    def resource_state(self):return {'writable':self.writable}


def queries(start=(0,0)):
    q=np.tile(start,(16,1)).astype(np.int64);q[:,0]+=np.arange(1,17)*4
    return q


def test_numeric_payload_consumed_at_wrapper_and_own_anchor():
    b=ControlBackend();m=NumericLastFrameAdapter(b)
    m.reset(np.zeros((240,416,3),np.uint8),{},4)
    q=queries();frames=m.generate(q)
    assert len(frames)==16 and frames[-1].shape==(240,416,3)
    assert set(b.calls[0])=={'image','camera','native_frames','seed'}
    assert 'native_queries_xy' not in b.calls[0]['camera']
    m.generate(queries(q[-1]));assert np.all(m.anchor==[128,0])
    assert b.calls[1]['image'][0,0,0]==1  # own previously submitted image
    assert m.counters['returned_native_frames']==34


@pytest.mark.parametrize('channel',['drop','identity'])
def test_control_numeric_channels_never_receive_trace_queries(channel):
    b=ControlBackend();m=NumericLastFrameAdapter(b,CameraConfig(camera_channel=channel))
    m.reset(np.zeros((240,416,3),np.uint8),{},0);m.generate(queries())
    camera=b.calls[0]['camera']
    assert set(camera)=={'camera_poses','intrinsics_fx_fy_cx_cy','selected_event_indices','camera_manifest','camera_manifest_sha256'}
    if channel=='drop':assert camera['camera_poses'] is None
    else:assert np.all(camera['camera_poses'][:,:3,3]==0)


def test_f7_state_and_rng_restore_without_resource_rollback():
    b=ControlBackend();m=NumericLastFrameAdapter(b);m.reset(np.zeros((240,416,3),np.uint8),{},8)
    state=m.snapshot();epoch=m.resource_counter_epoch
    a=m.generate(queries());seed=b.calls[-1]['seed']
    m.restore(state);c=m.generate(queries())
    assert b.calls[-1]['seed']==seed and np.array_equal(a[-1],c[-1])
    assert m.resource_counter_epoch==epoch and m.counters['returned_native_frames']==34


def test_missing_endpoint_terminates_without_repair_and_receipt_error_charges_work():
    b=ControlBackend();b.missing=True;m=NumericLastFrameAdapter(b);m.reset(np.zeros((240,416,3),np.uint8),{},0)
    assert m.generate(queries())[-1] is None
    assert m.generate(queries())==[None]*16 and len(b.calls)==1
    b=ControlBackend();b.bad_receipt=True;m=NumericLastFrameAdapter(b);m.reset(np.zeros((240,416,3),np.uint8),{},0)
    with pytest.raises(ValueError,match='receipt'):m.generate(queries())
    assert m.counters['returned_native_frames']==17 and m.counters['discarded_frames']==16


def test_training_inference_camera_hash_must_match():
    with pytest.raises(ValueError,match='mismatch'):
        NumericLastFrameAdapter(ControlBackend(),expected_camera_sha256='0'*64)


def test_native_conversion_uses_declared_integer_2x2_mean():
    class PatternBackend(ControlBackend):
        def generate(self,**request):
            result=super().generate(**request)
            a=np.zeros((480,832,3),np.uint8);a[0,0]=255
            result['frames'][1:]=[Image.fromarray(a) for _ in range(request['native_frames']-1)]
            return result
    m=NumericLastFrameAdapter(PatternBackend());m.reset(np.zeros((240,416,3),np.uint8),{},0)
    f=m.generate(queries())[0]
    assert f[0,0].tolist()==[64,64,64] and not f[0,1].any()


def test_actual_native_cost_is_unknown_without_backend_lifetime_counters():
    from rswm_inria.resources import ResourceCollector
    b=ControlBackend();m=NumericLastFrameAdapter(b);m.reset(np.zeros((240,416,3),np.uint8),{},0)
    collector=ResourceCollector('cpu')
    with collector.phase('generate',m):m.generate(queries())
    record=collector.finish()
    assert not record['measurement_complete']
    assert m.resource_counters()['status']=='read_failed' and m.counters['returned_native_frames']==17


def test_backend_work_before_exception_is_measured_and_epoch_changes_fail_closed():
    class CountingBackend(ControlBackend):
        resource_counter_epoch='lifetime'
        def __init__(self):super().__init__();self.actual=0
        def resource_counters(self):return {'native_frames':self.actual}
        def generate(self,**request):self.actual+=17;raise RuntimeError('failed after native work')
    b=CountingBackend();m=NumericLastFrameAdapter(b);m.reset(np.zeros((240,416,3),np.uint8),{},0)
    assert m.resource_counters()['values']['native_frames']==0
    with pytest.raises(RuntimeError):m.generate(queries())
    assert m.resource_counters()['values']['native_frames']==17 and m.counters['returned_native_frames']==0
    b.resource_counter_epoch='changed'
    assert m.resource_counters()['status']=='read_failed'
