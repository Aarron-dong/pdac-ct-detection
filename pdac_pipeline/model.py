from __future__ import annotations
import math
from pathlib import Path
from .io import bundle_member, read_json, sha256_file
def verify_bundle(manifest: Path) -> dict:
    manifest = manifest.resolve()
    root = manifest.parent.parent
    bundle = read_json(manifest)
    if bundle.get('status') != 'complete' or [m['fold'] for m in bundle['models']] != list(range(5)):
        raise ValueError('A complete five-fold bundle is required.')
    threshold = bundle['threshold']
    if not math.isfinite(threshold) or not 0 <= threshold <= 1 or bundle['comparison_operator'] != '>=':
        raise ValueError('The model requires a finite threshold in [0, 1] and the >= comparison operator.')
    checked_configurations = set()
    for row in bundle['models']:
        for role in ('localizer', 'detector'):
            model = row[role]
            if sha256_file(bundle_member(root, model['path'])) != model['sha256']:
                raise ValueError(f"Weight checksum mismatch: fold {row['fold']} {role}.")
            folder = bundle_member(root, model['model_directory'])
            for name, expected in [('plans.json', model['plans_sha256']), ('dataset.json', model['dataset_sha256'])]:
                key = (folder / name, expected)
                if key not in checked_configurations:
                    if sha256_file(key[0]) != expected:
                        raise ValueError(f'Model configuration checksum mismatch: {role}/{name}.')
                    checked_configurations.add(key)
    return bundle

def load_network(root: Path, model: dict):
    import torch
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    folder = bundle_member(root, model['model_directory'])
    plans = PlansManager(read_json(folder / 'plans.json'))
    dataset = read_json(folder / 'dataset.json')
    checkpoint = torch.load(bundle_member(root, model['path']), map_location='cpu', weights_only=True)
    configuration = plans.get_configuration(checkpoint['init_args']['configuration'])
    network = nnUNetTrainer.build_network_architecture(plans, configuration, determine_num_input_channels(plans, configuration, dataset), plans.get_label_manager(dataset).num_segmentation_heads, enable_deep_supervision=False)
    network.load_state_dict(checkpoint['network_weights'], strict=True)
    return (network, plans, configuration, dataset, checkpoint)

def load_predictor(root: Path, model: dict, device: str):
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    if device == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA is unavailable. Install a compatible CUDA build or explicitly use --device cpu.')
    network, plans, configuration, dataset, checkpoint = load_network(root, model)
    predictor = nnUNetPredictor(use_mirroring=True, allow_tqdm=False, verbose=False, device=torch.device(device))
    predictor.manual_initialization(network, plans, configuration, [checkpoint['network_weights']], dataset, checkpoint['trainer_name'], tuple(checkpoint['inference_allowed_mirroring_axes']))
    return predictor
