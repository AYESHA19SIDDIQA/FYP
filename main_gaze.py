#!/usr/bin/env python3
"""
Updated training script with robust on-disk -> dataset matching and improved debugging.

Key fixes:
- Recursive on-disk discovery of .npz and .json files (os.walk).
- Case-insensitive, normalized basename matching (optional suffix stripping).
- Probe dataset samples for 'file' entries and normalize them the same way.
- Filter datasets by dataset indices whose normalized basename appears in the on-disk matched set.
- Attach gaze_json_dir metadata to Subset objects so downstream code can access it.
- Many clear debug prints and diagnostic helpers to explain differences (e.g. "23 vs 8").

Usage: replace your previous training script with this file (or import helper classes/functions).
"""

import os
import sys
import json
import traceback
from datetime import datetime
from collections import defaultdict, Counter

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Project path
sys.path.append('.')
sys.path.append('..')

# Try import project modules; fallback placeholders
try:
    from cereprocess.datasets.pipeline import general_pipeline
    from cereprocess.train.xloop import get_datanpz
    from cereprocess.train.misc import def_dev, def_hyp
except Exception:
    def def_dev():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def def_hyp(batch_size=32, lr=0.0001, epochs=50, accum_iter=1):
        return {"batch_size": batch_size, "lr": lr, "epochs": epochs, "accum_iter": accum_iter}

    def get_def_ds(mins):
        return None, ("E:/NMT_events/NMT_events/edf", "NMT", "description", "results"), None

# Import model + dataset classes (must be available in your environment)
try:
    from neurogate_gaze import NeuroGATE_Gaze_MultiRes
    from eeg_gaze_fixations import EEGGazeFixationDataset
except Exception as e:
    print("Error importing NeuroGATE_Gaze_MultiRes or EEGGazeFixationDataset:", e)
    raise

# ----------------- Configuration -----------------
# Suffixes to strip from EEG basenames when normalizing (adjust if needed)
# ----------------- Configuration -----------------
# Suffixes to strip from EEG basenames when normalizing (adjust if needed)
DEFAULT_SUFFIXES_TO_STRIP = [
    '_clean', '_interp', '_filtered', '_fix', '_fixations', 
    '_epochs', '_epoch', 
    '_p0', '_p1', '_p2', '_p3', '_p4', '_p5',  # Handle various subject numbers
    '_p0_clean', '_p0_filtered',  # Handle combined suffixes
    '_session1', '_session2',  # Handle session numbers
    '_run1', '_run2'  # Handle run numbers
]

# ----------------- Utilities -----------------


def list_files_recursive(dir_path, ext):
    """Return sorted list of full paths under dir_path with extension ext (case-insensitive)."""
    out = []
    if not dir_path or not os.path.exists(dir_path):
        return out
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.lower().endswith(ext.lower()):
                out.append(os.path.join(root, f))
    out.sort()
    return out


def normalize_basename(path_or_name, strip_suffixes=None):
    """Return normalized basename string for matching: lowercased and optional suffix stripping."""
    if path_or_name is None:
        return None
    
    # Get base filename without extension
    if isinstance(path_or_name, (bytes, bytearray)):
        try:
            path_or_name = path_or_name.decode('utf-8', errors='ignore')
        except:
            path_or_name = str(path_or_name)
    
    base = os.path.splitext(os.path.basename(str(path_or_name)))[0]
    b = base.lower()
    
    # STRIP LEADING ZEROS from numbers at the beginning
    # This handles cases like "0000001" -> "1"
    import re
    # Match any leading zeros followed by digits, possibly with suffix
    # This handles: "0000001", "0000001_p0", "0000001_P0", etc.
    match = re.match(r'^0*(\d+)([_-].*)?$', b)
    if match:
        num_part = match.group(1)  # The number without leading zeros (e.g., "1")
        suffix_part = match.group(2) if match.group(2) else ""  # Optional suffix (e.g., "_p0")
        b = num_part + suffix_part
    
    # Also handle case where there's no leading zeros but might have other patterns
    # For example: "subject1_p0" -> "1_p0"
    # This regex extracts any number in the string (not just at beginning)
    # but let's keep it simple for now
    
    # Then strip any specified suffixes (like _p0, _P0, etc.)
    if strip_suffixes:
        # Sort by length (longest first) to handle complex suffixes
        sorted_suffixes = sorted(strip_suffixes, key=len, reverse=True)
        for s in sorted_suffixes:
            s_l = s.lower()
            if b.endswith(s_l):
                b = b[: -len(s_l)]
                break
    
    return b


# ----------------- Filtered Dataset Wrapper -----------------


class FilteredEEGGazeFixationDataset:
    """
    Wraps EEGGazeFixationDataset and filters it to only include indices whose
    normalized basename exists both as a .npz (under data_dir) and a .json (under gaze_json_dir).
    """

    def __init__(self, data_dir, gaze_json_dir, dataset_cls, dataset_kwargs=None, suffixes_to_strip=None):
        """
        dataset_cls: EEGGazeFixationDataset class
        dataset_kwargs: dict of kwargs to pass to dataset_cls
        suffixes_to_strip: list of suffixes to strip from basenames during normalization
        """
        self.data_dir = data_dir
        self.gaze_json_dir = gaze_json_dir
        self.dataset_cls = dataset_cls
        self.dataset_kwargs = dataset_kwargs or {}
        self.suffixes_to_strip = suffixes_to_strip or DEFAULT_SUFFIXES_TO_STRIP

        # Build on-disk normalized maps
        self._build_on_disk_maps()

        # Instantiate underlying dataset once (unfiltered)
        self.original_dataset = self.dataset_cls(data_dir=self.data_dir, gaze_json_dir=self.gaze_json_dir,
                                                **self.dataset_kwargs)
        # Probe dataset to build index -> normalized basename map
        self._build_dataset_index_map()

        # Compute list of dataset indices to keep (those whose normalized basename appears in disk_matched_basenames)
        self.filtered_indices = self._compute_filtered_indices()

        # Diagnostics printed on creation
        self._print_diagnostics()

    def _build_on_disk_maps(self):
        npz_paths = list_files_recursive(self.data_dir, '.npz')
        json_paths = list_files_recursive(self.gaze_json_dir, '.json')

        self.npz_map = defaultdict(list)
        for p in npz_paths:
            nb = normalize_basename(p, self.suffixes_to_strip)
            self.npz_map[nb].append(p)

        self.json_map = defaultdict(list)
        for p in json_paths:
            nb = normalize_basename(p, self.suffixes_to_strip)
            self.json_map[nb].append(p)

        self.disk_npz_basenames = set(self.npz_map.keys())
        self.disk_json_basenames = set(self.json_map.keys())
        self.disk_matched_basenames = self.disk_npz_basenames & self.disk_json_basenames

    def _build_dataset_index_map(self):
        self.dataset_index_to_base = {}
        self.dataset_base_to_indices = defaultdict(list)
        n = len(self.original_dataset)
        for idx in range(n):
            try:
                sample = self.original_dataset[idx]
                f = None
                if isinstance(sample, dict) and 'file' in sample:
                    f = sample['file']
                elif isinstance(sample, (list, tuple)) and len(sample) >= 3:
                    f = sample[2]
                if isinstance(f, (bytes, bytearray)):
                    try:
                        f = f.decode('utf-8', errors='ignore')
                    except:
                        f = str(f)
                if isinstance(f, str):
                    nb = normalize_basename(f, self.suffixes_to_strip)
                else:
                    nb = None
            except Exception:
                nb = None
            self.dataset_index_to_base[idx] = nb
            if nb:
                self.dataset_base_to_indices[nb].append(idx)

    def _compute_filtered_indices(self):
        kept = []
        for nb in sorted(self.disk_matched_basenames):
            idxs = self.dataset_base_to_indices.get(nb, [])
            if idxs:
                kept.extend(idxs)
        return sorted(set(kept))

    def _print_diagnostics(self):
        print("\n" + "=" * 70)
        print("FILTEREDEEGGAZEFIXATIONDATASET DIAGNOSTICS".center(70))
        print("=" * 70)
        print(f"  data_dir: {self.data_dir}")
        print(f"  gaze_json_dir: {self.gaze_json_dir}")
        print(f"\n  Disk: {len(self.npz_map)} unique npz basenames, {len(self.json_map)} unique json basenames")
        print(f"  Disk matched basenames: {len(self.disk_matched_basenames)}")
        print(f"    Examples (first 20): {sorted(list(self.disk_matched_basenames))[:20]}")
        dataset_bases = set(k for k in self.dataset_base_to_indices.keys() if k)
        print(f"\n  Dataset reported basenames: {len(dataset_bases)}")
        print(f"    Examples (first 20): {sorted(list(dataset_bases))[:20]}")
        on_disk_not_in_dataset = sorted(list(self.disk_matched_basenames - dataset_bases))
        in_dataset_not_on_disk = sorted(list(dataset_bases - self.disk_matched_basenames))
        print(f"\n  Normalized diff counts:")
        print(f"    On-disk matched but NOT in dataset: {len(on_disk_not_in_dataset)}")
        if on_disk_not_in_dataset:
            print(f"      Examples: {on_disk_not_in_dataset[:20]}")
        print(f"    In dataset but NOT matched on-disk: {len(in_dataset_not_on_disk)}")
        if in_dataset_not_on_disk:
            print(f"      Examples: {in_dataset_not_on_disk[:20]}")
        print(f"\n  Filtered indices kept: {len(self.filtered_indices)} (out of original {len(self.original_dataset)})")
        if self.filtered_indices:
            print(f"    Sample kept basenames (first 20): {[self.dataset_index_to_base[i] for i in self.filtered_indices[:20]]}")
        print("=" * 70)

    # Dataset protocol
    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        orig_idx = self.filtered_indices[idx]
        return self.original_dataset[orig_idx]


# ----------------- Debugging helpers -----------------


class DataDebugger:
    """Data debugging helpers (keeps a few utilities from your previous code)."""

    @staticmethod
    def print_header(title, width=80, char="="):
        print("\n" + char * width)
        print(title.center(width))
        print(char * width)

    @staticmethod
    def analyze_dataset(dataset, name="Dataset", max_samples=20):
        DataDebugger.print_header(f"ANALYZE {name}")
        print(f"  Total samples: {len(dataset)}")
        sample_count = min(max_samples, len(dataset))
        all_labels = []
        all_files = []
        for i in range(sample_count):
            try:
                sample = dataset[i]
                label = sample.get('label', None) if isinstance(sample, dict) else None
                all_labels.append(label)
                f = None
                if isinstance(sample, dict) and 'file' in sample:
                    f = sample['file']
                    if isinstance(f, (bytes, bytearray)):
                        try:
                            f = f.decode('utf-8', errors='ignore')
                        except:
                            f = str(f)
                    basename = os.path.splitext(os.path.basename(str(f)))[0]
                    all_files.append(basename)
                    print(f"  Sample {i}: Label={label}, File={basename}, EEG shape={sample['eeg'].shape}")
                    if 'gaze' in sample and sample['gaze'] is not None:
                        print(f"    Gaze shape: {sample['gaze'].shape}")
                else:
                    print(f"  Sample {i}: (no file key) Label={label}")
            except Exception as e:
                print(f"  Sample {i}: Error reading sample: {e}")
        if all_labels:
            counter = Counter(all_labels)
            print(f"\n  Label distribution (sampled): {dict(counter)}")
        return all_labels, all_files

    @staticmethod
    def analyze_dataloader(dataloader, name="Dataloader", max_batches=3):
        DataDebugger.print_header(f"ANALYZE {name}")
        print(f"  Total batches: {len(dataloader)}")
        all_labels = []
        all_files = []
        for bidx, batch in enumerate(dataloader):
            if bidx >= max_batches:
                break
            print(f"\n  Batch {bidx+1}: EEG shape {batch['eeg'].shape}, Labels {batch['label'].numpy()}")
            all_labels.extend(batch['label'].numpy().tolist())
            if 'file' in batch:
                files = []
                for f in batch['file']:
                    if isinstance(f, (bytes, bytearray)):
                        try:
                            f = f.decode('utf-8', errors='ignore')
                        except:
                            f = str(f)
                    files.append(os.path.splitext(os.path.basename(str(f)))[0])
                all_files.extend(files)
                print(f"    Files: {files}")
            if 'gaze' in batch and batch['gaze'] is not None:
                g = batch['gaze']
                print(f"    Gaze shape: {g.shape}, first val: {g.flatten()[0].item() if g.numel()>0 else 'N/A'}")
        if all_labels:
            print(f"\n  Label distribution in seen batches: {dict(Counter(all_labels))}")
        return all_labels, all_files

    @staticmethod
    def analyze_model_predictions(model, dataloader, device, name="Model Predictions"):
        DataDebugger.print_header(f"ANALYZE {name}")
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for batch in dataloader:
                eeg = batch['eeg'].to(device)
                labels = batch['label'].to(device)
                logits = model(eeg, return_attention=False)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())
        if not all_preds:
            print("  No predictions collected")
            return None, None, None, None
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        acc = (all_preds == all_labels).mean() * 100
        print(f"  Total: {len(all_preds)}, Accuracy: {acc:.2f}%")
        print(f"  Prediction counts: {dict(Counter(all_preds))}")
        print(f"  Label counts: {dict(Counter(all_labels))}")
        return acc, all_preds, all_labels, all_probs


# ----------------- Dataloader builder using Filtered wrapper -----------------


def get_dataloaders_fixed(datadir, batch_size, seed,
                          target_length=None, indexes=None,
                          gaze_json_dir=None, only_matched=True,
                          suffixes_to_strip=None, **kwargs):
    """
    Build dataloaders using FilteredEEGGazeFixationDataset to ensure only
    on-disk matched EEG<->JSON pairs are used.
    """
    DataDebugger.print_header("BUILD DATALOADERS (FIXED)")
    train_dir = os.path.join(datadir, 'train')
    eval_dir = os.path.join(datadir, 'eval')

    print(f"  data_dir: {datadir}")
    print(f"  train_dir: {train_dir}")
    print(f"  eval_dir: {eval_dir}")
    print(f"  gaze_json_dir: {gaze_json_dir}")
    print(f"  only_matched: {only_matched}")

    # instantiate the filtered wrapper for train and eval
    dataset_kwargs = {
        'indexes': indexes,
        'target_length': target_length,
        'eeg_sampling_rate': kwargs.get('eeg_sampling_rate', 50.0)
    }

    trainset = FilteredEEGGazeFixationDataset(
        data_dir=train_dir,
        gaze_json_dir=gaze_json_dir,
        dataset_cls=EEGGazeFixationDataset,
        dataset_kwargs=dataset_kwargs,
        suffixes_to_strip=suffixes_to_strip or DEFAULT_SUFFIXES_TO_STRIP
    )

    evalset = FilteredEEGGazeFixationDataset(
        data_dir=eval_dir,
        gaze_json_dir=gaze_json_dir,
        dataset_cls=EEGGazeFixationDataset,
        dataset_kwargs=dataset_kwargs,
        suffixes_to_strip=suffixes_to_strip or DEFAULT_SUFFIXES_TO_STRIP
    )

    # If not only_matched, you might want to use original_dataset instead; keep current behavior
    # Attach gaze_json_dir metadata on Subset-like wrappers (for downstream code)
    for ds in (trainset, evalset):
        # if ds is a wrapper with original_dataset attribute and not Subset, set attribute on wrapper
        if hasattr(ds, 'gaze_json_dir'):
            ds.gaze_json_dir = gaze_json_dir

    # Create DataLoaders
    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=min(batch_size, len(trainset)) if len(trainset) > 0 else 1,
        shuffle=True,
        num_workers=0,
        worker_init_fn=worker_init_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    eval_loader = torch.utils.data.DataLoader(
        evalset,
        batch_size=min(batch_size, len(evalset)) if len(evalset) > 0 else 1,
        shuffle=False,
        num_workers=0,
        worker_init_fn=worker_init_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    print("\nDATALOADER SUMMARY")
    print(f"  Train samples: {len(trainset)} | batches: {len(train_loader)}")
    print(f"  Eval samples:  {len(evalset)} | batches: {len(eval_loader)}")

    # Quick dataset diagnostics
    DataDebugger.analyze_dataset(trainset, "Filtered Train Dataset", max_samples=5)
    DataDebugger.analyze_dataset(evalset, "Filtered Eval Dataset", max_samples=5)

    return train_loader, eval_loader, {
        'train_filtered': len(trainset),
        'eval_filtered': len(evalset),
        'train_disk_matched': len(trainset.disk_matched_basenames) if hasattr(trainset, 'disk_matched_basenames') else 0,
        'eval_disk_matched': len(evalset.disk_matched_basenames) if hasattr(evalset, 'disk_matched_basenames') else 0
    }


# ----------------- Training utilities -----------------


def compute_class_weights(labels, device='cpu'):
    """
    Compute class weights for handling class imbalance.
    Classes with fewer samples will get higher weights.
    
    Args:
        labels (array-like): All training labels (list, numpy array, or tensor)
        device (str): Device to place the weights tensor on ('cpu' or 'cuda')
    
    Returns:
        torch.Tensor: Class weights tensor of shape [max_class_label + 1].
    """
    # Convert to numpy array if needed
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)
    
    # Get unique classes and their counts
    classes, counts = np.unique(labels, return_counts=True)
    
    # Compute weights: inversely proportional to class frequency
    # Formula matches sklearn's compute_class_weight with mode='balanced':
    # weight[i] = total_samples / (num_classes * class_count[i])
    total_samples = len(labels)
    num_classes = len(classes)
    weights = total_samples / (num_classes * counts)
    
    # Normalize weights to sum to num_classes
    # This maintains relative class importance while keeping loss scale stable
    weights = weights * (num_classes / weights.sum())
    
    # Create tensor with weights in proper class order
    max_class = int(classes.max())
    class_weights = torch.ones(max_class + 1, dtype=torch.float32)
    for idx, cls in enumerate(classes):
        class_weights[int(cls)] = weights[idx]
    
    return class_weights.to(device)


def compute_gaze_attention_loss(attention_map, gaze, labels, loss_type='mse'):
    if loss_type == 'mse':
        return F.mse_loss(attention_map, gaze)
    elif loss_type == 'weighted_mse':
        weights = gaze * 2 + 0.1
        return (weights * (attention_map - gaze) ** 2).mean()
    elif loss_type == 'cosine':
        att = attention_map.view(attention_map.shape[0], -1)
        gz = gaze.view(gaze.shape[0], -1)
        return 1 - F.cosine_similarity(att, gz).mean()
    elif loss_type == 'kl':
        att_prob = F.softmax(attention_map.view(attention_map.shape[0], -1), dim=1)
        gaze_prob = F.softmax(gaze.view(gaze.shape[0], -1), dim=1)
        return F.kl_div(att_prob.log(), gaze_prob, reduction='batchmean')
    else:
        raise ValueError("Unknown gaze loss type")


def train_epoch_with_gaze(model, train_loader, optimizer, device, gaze_weight=0.1, gaze_loss_type='mse', class_weights=None):
    model.train()
    total_loss = total_cls = total_gaze = 0.0
    correct = total = 0
    batches_with_gaze = samples_with_gaze = 0
    
    # Move class_weights to device if provided
    if class_weights is not None:
        class_weights = class_weights.to(device)

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
    for batch_idx, batch in pbar:
        eeg = batch['eeg'].to(device)
        labels = batch['label'].to(device)

        # files for debug
        batch_files = []
        if 'file' in batch:
            for f in batch['file']:
                if isinstance(f, (bytes, bytearray)):
                    try:
                        f = f.decode('utf-8', errors='ignore')
                    except:
                        f = str(f)
                batch_files.append(os.path.basename(str(f)))
        else:
            batch_files = ["unknown"] * eeg.shape[0]

        has_gaze = 'gaze' in batch and batch['gaze'] is not None
        if has_gaze:
            gaze = batch['gaze'].to(device)
            batches_with_gaze += 1
            samples_with_gaze += eeg.shape[0]

        # debug print for first two samples in batch
        debug_lines = []
        # for i, fname in enumerate(batch_files[:2]):
        #     debug_lines.append(f"[B{batch_idx} S{i}] {fname}")
        # print(f"\n[TRAIN] Batch {batch_idx+1}/{len(train_loader)} | " + "; ".join(debug_lines))

        # forward
        if has_gaze:
            outputs = model(eeg, return_attention=True)
            if isinstance(outputs, tuple):
                logits, attention_map = outputs
            else:
                logits = outputs['logits']
                attention_map = outputs['attention_map']
        else:
            logits = model(eeg, return_attention=False)
            attention_map = None

        # classification loss with class weights
        if class_weights is not None:
            cls_loss = F.cross_entropy(logits, labels, weight=class_weights)
        else:
            cls_loss = F.cross_entropy(logits, labels)
        if has_gaze and attention_map is not None:
            gaze_loss = compute_gaze_attention_loss(attention_map, gaze, labels, gaze_loss_type)
            loss = cls_loss + gaze_weight * gaze_loss
        else:
            gaze_loss = torch.tensor(0.0).to(device)
            loss = cls_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_cls += cls_loss.item()
        total_gaze += gaze_loss.item() if has_gaze else 0.0

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{correct/total*100:.1f}%"})

    avg_loss = total_loss / max(len(train_loader), 1)
    avg_cls = total_cls / max(len(train_loader), 1)
    avg_gaze = total_gaze / max(len(train_loader), 1)
    acc = correct / total * 100 if total > 0 else 0.0

    gaze_stats = {
        'batches_with_gaze': batches_with_gaze,
        'samples_with_gaze': samples_with_gaze,
        'total_batches': len(train_loader),
        'total_samples': total
    }
    return avg_loss, avg_cls, avg_gaze, acc, gaze_stats


def evaluate_model(model, eval_loader, device):
    model.eval()
    correct = total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in eval_loader:
            eeg = batch['eeg'].to(device)
            labels = batch['label'].to(device)
            logits = model(eeg, return_attention=False)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
    acc = correct / total * 100 if total > 0 else 0.0
    return acc, all_labels, all_preds


# ----------------- Main -----------------


def main():
    DataDebugger.print_header("GAZE-GUIDED ATTENTION TRAINING (FIXED)", width=80)

    device = def_dev()
    print(f"Device: {device}")

    mins = 5
    length = mins * 3000
    input_size = (22, length)

    try:
        tuh, nmt, nmt_4k = get_def_ds(mins)
        curr_data = nmt
        print(f"Using dataset: {curr_data[1]}")
    except Exception:
        curr_data = ("E:/NMT_events/NMT_events/edf", "NMT", "description", "results")

    print("\nStep 1: Preprocessing / load preprocessed npz")
    try:
        data_dir, data_description = get_datanpz(
            curr_data[0], curr_data[3], general_pipeline(dataset='NMT', length_minutes=mins), input_size
        )
        print(f"Preprocessed data at: {data_dir}")
    except Exception as e:
        print("get_datanpz error:", e)
        data_dir = "./preprocessed_data"
        data_description = {'channel_no': 22, 'sampling_rate': 100, 'time_span': 300}

    gaze_json_dir = "results/gaze"
    hyps = def_hyp(batch_size=16, epochs=15, lr=1e-4, accum_iter=2)

    # Build dataloaders (fixed)
    try:
        train_loader, eval_loader, gaze_stats = get_dataloaders_fixed(
            datadir=data_dir,
            batch_size=hyps['batch_size'],
            seed=42,
            target_length=length,
            gaze_json_dir=gaze_json_dir,
            only_matched=True,
            suffixes_to_strip=DEFAULT_SUFFIXES_TO_STRIP,
            eeg_sampling_rate=50.0
        )
    except Exception as e:
        print("Error building dataloaders:", e)
        traceback.print_exc()
        return

    # Sanity checks
    if len(train_loader.dataset) == 0 or len(eval_loader.dataset) == 0:
        print("No data in train or eval after filtering. Inspect diagnostics above.")
        return

    # Model
    model = NeuroGATE_Gaze_MultiRes(
        n_chan=int(data_description.get('channel_no', 22)),
        n_outputs=2,
        original_time_length=length
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hyps['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.1)

    # Compute class weights for handling class imbalance
    print("\nComputing class weights from training data...")
    all_train_labels = []
    for batch in train_loader:
        all_train_labels.extend(batch['label'].numpy().tolist())
    class_weights = compute_class_weights(all_train_labels, device=str(device))
    print(f"Class weights computed: {class_weights}")
    print(f"Label distribution: {dict(Counter(all_train_labels))}")

    # Quick check forward
    try:
        sample_batch = next(iter(train_loader))
        test_eeg = sample_batch['eeg'].to(device)[:2]
        model.eval()
        with torch.no_grad():
            logits = model(test_eeg, return_attention=False)
            print("Model forward OK, logits shape:", logits.shape)
    except Exception as e:
        print("Model forward error:", e)
        traceback.print_exc()
        return

    # Training loop
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'eval_acc': []}

    for epoch in range(hyps['epochs']):
        DataDebugger.print_header(f"EPOCH {epoch+1}/{hyps['epochs']}", width=60, char='-')
        tr_loss, tr_cls, tr_gaze, tr_acc, tr_stats = train_epoch_with_gaze(
            model, train_loader, optimizer, device, gaze_weight=0.1, gaze_loss_type='mse', class_weights=class_weights
        )
        ev_acc, ev_labels, ev_preds = evaluate_model(model, eval_loader, device)
        scheduler.step(ev_acc)

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['eval_acc'].append(ev_acc)

        print(f"\nEpoch {epoch+1}: Train acc {tr_acc:.2f}% | Eval acc {ev_acc:.2f}%")
        if ev_acc > best_acc:
            best_acc = ev_acc
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'accuracy': ev_acc
            }, 'best_model_gaze_attention_fixed.pth')
            print(f"Saved best model at epoch {epoch+1} (acc {ev_acc:.2f}%)")

        # Diagnostic analyze predictions if eval acc is 0
        if ev_acc == 0 or (epoch+1) % 2 == 0:
            DataDebugger.analyze_model_predictions(model, eval_loader, device, f"Epoch {epoch+1} analysis")

    print("\nTraining complete. Best eval acc: %.2f%%" % best_acc)
    np.save('training_history_fixed.npy', history)


if __name__ == "__main__":
    main()