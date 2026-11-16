from pathlib import Path
import argparse
import json

def main() -> int:
    parser = argparse.ArgumentParser(description='Detect PDAC in a contrast-enhanced CT volume.')
    parser.add_argument('--image', type=Path, required=True, help='One three-dimensional contrast-enhanced CT NIfTI image in HU.')
    parser.add_argument('--output', type=Path, required=True, help='A new or empty prediction directory.')
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parent / 'weights/frozen_bundle.json')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    from pdac_pipeline.inference import FrozenPredictor
    result = FrozenPredictor(args.bundle, device=args.device).predict_case(args.image, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['status'] == 'complete' else 1
if __name__ == '__main__':
    raise SystemExit(main())
