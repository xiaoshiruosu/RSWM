"""Run an audited map through the existing core runner and independent replay.

Requires an explicit local checkpoint OR --diagnostic-noise. This cooperative
in-process CLI is not an isolated evaluation service. It never downloads weights.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from rswm_inria.registry import create_adapter, capabilities
from rswm_inria.geometry import file_hash, digest, rgb
from rswm_inria.runner import run_task
from rswm_inria.replay import load_json, replay_run
from rswm_inria.source_probe import save_new_json
from rswm_inria.resources import ResourceCollector
from rswm_inria.replay import schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--world', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--order', choices=['raster', 'frontier', 'lazy'], default='frontier')
    parser.add_argument('--seed', type=int, default=0)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--model-dir', type=Path)
    choice.add_argument('--diagnostic-noise', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--prompt')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Choose a new attempt directory; existing runs are never overwritten.')
    if args.seed < 0 or args.steps < 1:
        parser.error('seed must be nonnegative and steps must be positive.')
    task = load_json(args.task.read_text(encoding='utf-8'))
    with Image.open(args.world) as image:
        if image.format != 'PNG' or image.mode != 'RGB' or image.size != (2048,2048):
            raise ValueError('Expected a canonical 2048-square RGB PNG world.')
        world = rgb(np.asarray(image), (2048,2048,3)).copy()
    config = {'model_id': 'audited-map', 'adapter': args.order,
              'device': args.device if args.model_dir else 'cpu'}
    if args.model_dir:
        config.update(model_dir=str(args.model_dir), steps=args.steps)
        if args.prompt is not None:
            config['prompt'] = args.prompt
    else:
        if args.prompt is not None:
            parser.error('diagnostic noise does not consume a prompt.')
        config['diagnostic_noise'] = True
    collector = ResourceCollector(config['device'])
    collector.calls['planned_output_slots'] = sum(r['kind'] == 'output' for r in schedule(task))
    with collector.phase('model_load'):
        model = create_adapter(config)
    completer = model.completer
    load_seconds = next(p['wall_seconds'] for p in collector.phases if p['phase'] == 'model_load')
    report, journal = run_task(task, world, model, seed=args.seed, out_dir=args.out, resources=collector)
    collector.calls['valid_outputs'] = sum(r.get('kind') == 'output' and r.get('status') == 'valid' for r in journal.rows)
    activity = model.activity_report()
    save_new_json(args.out/'map-activity.json', activity)
    from rswm_inria.geometry import crop
    with collector.phase('replay'):
        replay = replay_run(task, crop(world,(0,0),task['origin']), args.out,
                            expected_head=journal.head)
    collector.execution_complete = replay['verified_png_outputs'] == replay['planned_output_events']
    save_new_json(args.out/'resources.json', collector.finish(args.out))
    save_new_json(args.out/'recomputed.json', replay)
    manifest = {'schema':'rswm-map-run-v1','order':args.order,'seed':args.seed,
                'task_sha256':digest(task),'world_file_sha256':file_hash(args.world),
                'completer':'diagnostic_noise' if args.diagnostic_noise else 'local_diffusers',
                'public_config':config,'config_sha256':digest(config),'capabilities':capabilities(config),
                'weight_files':getattr(completer,'weight_manifest',None),
                'steps':args.steps if completer is not None else None,
                'model_load_cpu_wall_seconds':load_seconds,
                'journal_head':report['journal_head'],
                'receipt_scope':'same-process retained head, not externally trusted timestamping',
                'protocol_security_verified':False,'real_model_validated':False}
    save_new_json(args.out/'map-run.json',manifest)


if __name__ == '__main__':
    main()
