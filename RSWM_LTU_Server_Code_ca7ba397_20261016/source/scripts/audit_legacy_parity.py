"""Check the camera math against an explicitly supplied, hash-pinned legacy ZIP.

Only four whitelisted functions from the pinned member are compiled. No network,
model weights or vendor package initialization is involved. PyTorch CPU required.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
from pathlib import Path
import zipfile
import numpy as np
from rswm_inria.adapters.camera_conditioning import CameraConfig, build_camera_batch, plucker_at_pixels
from rswm_inria.geometry import DIAGONAL
from rswm_inria.source_probe import save_new_json

MEMBER = 'rswm_server_pack_synced_20260817/wan/utils/cam_utils.py'
SOURCE_SHA256 = '27d4801792316322e09711b31cdde595940ab2504e5580a0cf290b72c9789951'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-zip',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists(): raise FileExistsError('Output already exists.')
    with zipfile.ZipFile(args.legacy_zip) as archive:
        if archive.getinfo(MEMBER).file_size > 32768: raise ValueError('Unexpected legacy member size.')
        source=archive.read(MEMBER)
    if hashlib.sha256(source).hexdigest()!=SOURCE_SHA256:
        raise ValueError('Legacy source differs from the audited member; review before updating its lock.')
    import torch
    names={'SE3_inverse','compute_relative_poses','create_meshgrid','get_plucker_embeddings'}
    tree=ast.parse(source.decode('utf-8'))
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in functions}!=names: raise ValueError('Missing whitelisted functions.')
    context={'torch':torch}
    exec(compile(ast.Module(body=functions,type_ignores=[]),MEMBER,'exec'),context)
    q=np.vstack([np.arange(1,17)[:,None]*[12,0],[192,0]+np.arange(1,17)[:,None]*[0,12]])
    yy,xx=np.mgrid[:3,:4];xy=np.stack([xx.ravel()+.5,yy.ravel()+.5],axis=1)
    records=[]
    for framewise in (False,True):
        for scaling in ('fixed_units','legacy_maxnorm'):
            batch=build_camera_batch(q,config=CameraConfig(scaling,framewise))
            raw=np.tile(np.eye(4),(len(batch.selected_event_indices),1,1))
            raw[:,:2,3]=batch.native_queries_xy[batch.selected_event_indices]/DIAGONAL
            reference=context['compute_relative_poses'](torch.tensor(raw,dtype=torch.float64),
                         framewise=framewise,normalize_trans=scaling=='legacy_maxnorm')
            ref_rays=context['get_plucker_embeddings'](reference,
                         torch.tensor(batch.intrinsics_fx_fy_cx_cy,dtype=torch.float64),3,4).numpy().reshape(len(raw),-1,6)
            rays=plucker_at_pixels(batch.camera_poses,batch.intrinsics_fx_fy_cx_cy,xy)
            pose_error=float(np.max(np.abs(reference.numpy()-batch.camera_poses)))
            ray_error=float(np.max(np.abs(ref_rays-rays)))
            if pose_error>1e-12 or ray_error>1e-6: raise AssertionError('Camera math parity failed.')
            records.append({'framewise':framewise,'scaling':scaling,
                            'camera_manifest':batch.config.manifest(),
                            'camera_manifest_sha256':batch.config_sha256,
                            'pose_max_abs_error':pose_error,'ray_max_abs_error':ray_error})
    save_new_json(args.out,{'schema':'rswm-legacy-parity-v1','source_member':MEMBER,
                  'source_sha256':SOURCE_SHA256,'torch_version':torch.__version__,
                  'records':records,'scope':'pure translations and sparse 3x4 CPU ray grid',
                  'real_weights_used':False,'real_images_used':False,'backend_validated':False})


if __name__=='__main__': main()
