"""Run the CPU condition audit; no imagery, checkpoint or CUDA is required."""
import argparse
from pathlib import Path
from rswm_inria.adapters.camera_conditioning import scale_audit
from rswm_inria.source_probe import save_new_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = scale_audit()
    for row in result['records']:
        expected = 2. if row['translation_scaling'] == 'fixed_units' else 1.
        if abs(row['translation_l2_ratio'] - expected) > 1e-12:
            raise AssertionError('Condition scale contract failed.')
    save_new_json(args.out, result)


if __name__ == '__main__':
    main()
