import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
import json
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import signal, interpolate
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class Fixation:
    """Represents a single gaze fixation from your JSON data"""
    start_time: float  # time.time() timestamp (IGNORE for EEG alignment)
    end_time: float    # time.time() timestamp (IGNORE for EEG alignment)
    duration: float    # end_time - start_time
    x: float          # EEG TIME in seconds! (0-300 for 5-min EEG)
    y: float          # Channel position (vertical)
    channels: List[str]  # EEG channels expert looked at
    num_points: int   # Number of gaze points in this fixation
    
    @property
    def attention_weight(self) -> float:
        """Calculate attention weight based on fixation characteristics"""
        # Higher weight for longer fixations with more points
        duration_weight = min(1.0, self.duration * 2)
        density_weight = min(1.0, self.num_points / 50)
        return duration_weight * density_weight * 0.8 + 0.2

class EEGGazeFixationDataset(Dataset):
    """
    Dataset that works with your get_datanpz() pipeline AND gaze JSON fixation data.
    
    KEY FIX: Uses ONLY x coordinate (EEG time in seconds) for alignment.
    Ignores start_time/end_time (they're in time.time() units).
    """
    
    def __init__(self, data_dir: str, 
                 indexes: Optional[List[int]] = None,
                 target_length: Optional[int] = None,
                 gaze_json_dir: Optional[str] = None,
                 gaze_json_pattern: str = "*.json",
                 eeg_sampling_rate: float = 100.0,
                 channel_mapping: Optional[Dict[str, int]] = None,
                 debug: bool = False,
                 **kwargs):
        """
        Args:
            data_dir: Directory with EEG .npz files (from get_datanpz)
            indexes: Subset indices to use
            target_length: Target time length for EEG (e.g., 15000 for 5 mins)
            gaze_json_dir: Directory with gaze JSON files
            gaze_json_pattern: Pattern to match gaze files
            eeg_sampling_rate: EEG sampling rate (from data_description)
            channel_mapping: Map from channel names to indices
            debug: Enable debugging output and plots
        """
        super().__init__()
        
        self.data_dir = data_dir
        self.target_length = target_length
        self.eeg_sampling_rate = eeg_sampling_rate
        self.debug = debug
        
        # Calculate EEG duration in seconds
        self.eeg_duration_seconds = target_length / eeg_sampling_rate if target_length else None
        
        if self.debug:
            print(f"\n=== INITIALIZING EEGGazeFixationDataset ===")
            print(f"Target length: {target_length} samples")
            print(f"EEG sampling rate: {eeg_sampling_rate} Hz")
            print(f"EEG duration: {self.eeg_duration_seconds:.1f} seconds ({self.eeg_duration_seconds/60:.1f} minutes)")
        
        # Default channel mapping (from your pipeline)
        if channel_mapping is None:
            self.channel_mapping = self._create_nmt_channel_mapping()
        else:
            self.channel_mapping = channel_mapping
        
        self.channel_to_index = {ch: idx for idx, ch in enumerate(self.channel_mapping)}
        
        # Load data.csv to get file list and labels
        csv_file = os.path.join(data_dir, 'data.csv')
        if os.path.exists(csv_file):
            import pandas as pd
            self.df = pd.read_csv(csv_file)
            if self.debug:
                print(f"Loaded {len(self.df)} samples from data.csv")
        else:
            # Fallback: list all .npz files
            self.df = None
            print("Warning: data.csv not found, using .npz files directly")
        
        # Get list of EEG files
        self.eeg_files = self._get_eeg_files()
        
        # Apply index filtering
        if indexes is not None:
            self.eeg_files = [self.eeg_files[i] for i in indexes]
            if self.debug:
                print(f"Using {len(self.eeg_files)} files after index filtering")
        
        # Find gaze JSON files
        gaze_dir = gaze_json_dir if gaze_json_dir else data_dir
        self.gaze_files = sorted(glob.glob(os.path.join(gaze_dir, gaze_json_pattern)))
        
        if self.debug:
            print(f"Found {len(self.gaze_files)} gaze JSON files")
        
        # Create mapping from EEG files to gaze files
        self.eeg_to_gaze = self._create_file_mappings()
        
        # Preload gaze data
        self.gaze_cache = {}
        self._preload_gaze_data()
    
    def _create_nmt_channel_mapping(self) -> List[str]:
        """Create channel mapping for NMT dataset (from your pipeline)"""
        return [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
            'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
            'Fz', 'Cz', 'Pz', 'A1', 'A2', 'EKG'
        ]
    
    def _get_eeg_files(self) -> List[str]:
        """Get list of EEG .npz files"""
        files = []
        
        if self.df is not None and 'File' in self.df.columns:
            for idx, csv_file in enumerate(self.df['File']):
                file_path = str(csv_file).strip()
                
                if os.path.isabs(file_path):
                    if os.path.exists(file_path):
                        files.append(file_path)
                    else:
                        filename = os.path.basename(file_path)
                        alt_path = os.path.join(self.data_dir, filename)
                        if os.path.exists(alt_path):
                            files.append(alt_path)
                else:
                    rel_path = os.path.join(self.data_dir, file_path)
                    if os.path.exists(rel_path):
                        files.append(rel_path)
                    else:
                        filename = os.path.basename(file_path)
                        alt_path = os.path.join(self.data_dir, filename)
                        if os.path.exists(alt_path):
                            files.append(alt_path)
        
        # Fallback: list all .npz files in directory
        if not files:
            npz_files = sorted(glob.glob(os.path.join(self.data_dir, '*.npz')))
            files = npz_files
        
        return files
    
    def _create_file_mappings(self) -> Dict[str, Optional[str]]:
        """Map EEG files to corresponding gaze JSON files"""
        mappings = {}
        
        for eeg_file in self.eeg_files:
            eeg_basename = os.path.basename(eeg_file)
            eeg_name = os.path.splitext(eeg_basename)[0]
            
            gaze_file = None
            for g_file in self.gaze_files:
                g_basename = os.path.basename(g_file)
                g_name = os.path.splitext(g_basename)[0]
                
                # Strategy 1: Exact match
                if eeg_name == g_name:
                    gaze_file = g_file
                    break
                
                # Strategy 2: Contains match
                if eeg_name in g_name or g_name in eeg_name:
                    gaze_file = g_file
                    break
                
                # Strategy 3: Remove suffixes
                eeg_base = eeg_name.replace('_eeg', '').replace('_processed', '')
                g_base = g_name.replace('_gaze', '').replace('_fixations', '')
                if eeg_base == g_base:
                    gaze_file = g_file
                    break
            
            mappings[eeg_file] = gaze_file
            
            if not gaze_file and self.debug:
                print(f"  Warning: No gaze match for {eeg_basename}")
        
        # Count matches
        matches = sum(1 for g in mappings.values() if g is not None)
        
        if self.debug:
            print(f"Matched {matches}/{len(mappings)} EEG files with gaze data")
        
        return mappings
    
    def _preload_gaze_data(self):
        """Preload all gaze JSON files into memory"""
        for gaze_file in self.gaze_files:
            try:
                with open(gaze_file, 'r') as f:
                    gaze_json = json.load(f)
                
                all_fixations = []
                for session in gaze_json.get('sessions', []):
                    for fix_dict in session.get('fixations', []):
                        fixation = Fixation(
                            start_time=fix_dict['start_time'],  # time.time()
                            end_time=fix_dict['end_time'],      # time.time()
                            duration=fix_dict['duration'],
                            x=fix_dict['x'],                    # EEG TIME in seconds!
                            y=fix_dict['y'],
                            channels=fix_dict.get('channel', []),
                            num_points=fix_dict.get('num_points', 1)
                        )
                        all_fixations.append(fixation)
                
                self.gaze_cache[gaze_file] = all_fixations
                
                if self.debug and all_fixations:
                    x_values = [f.x for f in all_fixations]
                    print(f"  Loaded {len(all_fixations)} fixations from {os.path.basename(gaze_file)}")
                    print(f"    EEG time (x) range: {min(x_values):.1f}s - {max(x_values):.1f}s")
                    print(f"    start_time range: {min(f.start_time for f in all_fixations):.1f} - {max(f.start_time for f in all_fixations):.1f}")
                
            except Exception as e:
                print(f"  Error loading {gaze_file}: {e}")
                self.gaze_cache[gaze_file] = []
    
    def _load_eeg_from_npz(self, file_path: str) -> np.ndarray:
        """Load EEG from .npz file (compatible with get_datanpz output)"""
        data = np.load(file_path)
        
        # Your pipeline saves EEG as numpy arrays
        eeg_keys = ['eeg', 'data', 'X', 'signal']
        for key in eeg_keys:
            if key in data:
                eeg = data[key]
                break
        else:
            # Look for any array
            for key in data.files:
                arr = data[key]
                if isinstance(arr, np.ndarray) and arr.ndim == 2:
                    eeg = arr
                    break
            else:
                raise ValueError(f"No EEG data found in {file_path}")
        
        # Ensure shape is (channels, time)
        if eeg.shape[0] > eeg.shape[1]:
            eeg = eeg.T
        
        return eeg.astype(np.float32)
    
    def _get_label(self, file_path: str) -> int:
        """Get label for EEG file"""
        if self.df is not None and 'File' in self.df.columns and 'Label' in self.df.columns:
            filename = os.path.basename(file_path)
            
            for idx, row in self.df.iterrows():
                csv_file = str(row['File'])
                
                if (filename in csv_file or
                    os.path.basename(csv_file) == filename or
                    file_path in csv_file or
                    csv_file in file_path):
                    
                    return int(row['Label'])
        
        # Fallback: check .npz file
        try:
            data = np.load(file_path)
            for key in ['label', 'y', 'target']:
                if key in data:
                    return int(data[key])
        except:
            pass
        
        # Default
        return 0
    
    def _filter_fixations_by_eeg_time(self, fixations: List[Fixation]) -> List[Fixation]:
        """
        CRITICAL FIX: Filter fixations based on x coordinate (EEG time in seconds)
        Ignores start_time/end_time (they're in time.time() units)
        """
        if not fixations:
            return []
        
        # EEG duration in seconds
        max_eeg_time = self.eeg_duration_seconds
        
        if self.debug:
            print(f"\n  Filtering by EEG time (x coordinate):")
            print(f"    Keep fixations where x ≤ {max_eeg_time}s")
            
            x_values = [f.x for f in fixations]
            print(f"    Original x range: {min(x_values):.1f}s - {max(x_values):.1f}s")
            
            # Show difference between x and start_time
            print(f"    Example fixation:")
            print(f"      start_time: {fixations[0].start_time} (time.time())")
            print(f"      x: {fixations[0].x}s (EEG time)")
        
        # Filter based on x (EEG time)
        filtered = [f for f in fixations if f.x <= max_eeg_time]
        
        if self.debug:
            print(f"    Original: {len(fixations)} fixations")
            print(f"    Filtered: {len(filtered)} fixations")
            
            if filtered:
                filtered_x = [f.x for f in filtered]
                print(f"    Filtered x range: {min(filtered_x):.1f}s - {max(filtered_x):.1f}s")
        
        return filtered
    
    def _convert_fixations_to_gaze_map(self, 
                                      fixations: List[Fixation],
                                      eeg_duration: float,
                                      n_channels: int) -> np.ndarray:
        """
        Convert fixations to 2D gaze map matching EEG shape
        Uses x coordinate as EEG time reference
        """
        n_time = int(eeg_duration * self.eeg_sampling_rate)  # Should be target_length
        
        if self.debug:
            print(f"\n  Creating gaze map from {len(fixations)} fixations:")
            print(f"    EEG duration: {eeg_duration:.1f}s")
            print(f"    Time samples: {n_time}")
            print(f"    Channels: {n_channels}")
        
        # Create empty gaze map
        gaze_map = np.zeros((n_channels, n_time), dtype=np.float32)
        
        if not fixations:
            if self.debug:
                print("  No fixations, returning baseline attention")
            return gaze_map + 0.01  # Baseline attention
        
        # Process each fixation
        for fixation_idx, fixation in enumerate(fixations):
            # CRITICAL: Use fixation.x as EEG time in seconds
            eeg_time_seconds = fixation.x
            
            # Convert EEG time to sample index
            time_normalized = eeg_time_seconds / eeg_duration  # Should be 0 to 1
            center_idx = int(time_normalized * n_time)
            
            # Clamp to valid range
            center_idx = max(0, min(center_idx, n_time - 1))
            
            # Calculate spread based on fixation duration
            # Longer fixations = wider attention spread
            duration_samples = int(fixation.duration * self.eeg_sampling_rate)
            half_duration = max(1, duration_samples // 2)
            
            # Define attention window
            start_idx = max(0, center_idx - half_duration)
            end_idx = min(n_time, center_idx + half_duration)
            
            # Get channel indices
            channel_indices = []
            for ch_name in fixation.channels:
                if ch_name in self.channel_to_index:
                    channel_indices.append(self.channel_to_index[ch_name])
            
            if not channel_indices:
                # If no specific channels, distribute lightly
                channel_indices = list(range(n_channels))
                weight = fixation.attention_weight * 0.3
            else:
                weight = fixation.attention_weight
            
            # Apply Gaussian attention around center_idx
            if end_idx > start_idx:
                center = (start_idx + end_idx) / 2
                sigma = max(1.0, (end_idx - start_idx) / 4.0)
                
                time_indices = np.arange(start_idx, end_idx)
                gaussian = np.exp(-0.5 * ((time_indices - center) / sigma) ** 2)
                
                for ch_idx in channel_indices:
                    gaze_map[ch_idx, start_idx:end_idx] += gaussian * weight
            
            if self.debug and fixation_idx < 3:  # Debug first 3 fixations
                print(f"\n    Fixation {fixation_idx}:")
                print(f"      EEG time (x): {eeg_time_seconds:.2f}s -> idx: {center_idx}")
                print(f"      Duration: {fixation.duration:.2f}s -> {duration_samples} samples")
                print(f"      Channels: {fixation.channels}")
                print(f"      Weight: {weight:.3f}")
                print(f"      Attention window: {start_idx}-{end_idx} samples")
        
        # Normalize
        if gaze_map.max() > 0:
            gaze_map = gaze_map / gaze_map.max()
        
        # Add baseline
        gaze_map = gaze_map * 0.9 + 0.1
        
        if self.debug:
            print(f"\n  Gaze map statistics:")
            print(f"    Min: {gaze_map.min():.3f}, Max: {gaze_map.max():.3f}")
            print(f"    Mean: {gaze_map.mean():.3f}")
            
            # Check if any attention beyond EEG duration
            gaze_temporal = gaze_map.mean(axis=0)
            significant_idx = np.where(gaze_temporal > 0.15)[0]
            if len(significant_idx) > 0:
                max_idx = significant_idx[-1]
                max_time = max_idx / self.eeg_sampling_rate
                print(f"    Last significant gaze at: {max_time:.1f}s")
        
        return gaze_map
    
    def _process_eeg(self, eeg: np.ndarray) -> np.ndarray:
        """Process EEG: normalize and crop/pad to target_length"""
        # Z-score normalize each channel
        eeg_processed = np.zeros_like(eeg)
        for ch in range(eeg.shape[0]):
            channel_data = eeg[ch]
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            if std > 0:
                eeg_processed[ch] = (channel_data - mean) / std
            else:
                eeg_processed[ch] = channel_data - mean
        
        # Crop or pad to target_length
        if self.target_length is not None:
            current_length = eeg_processed.shape[1]
            
            if self.debug:
                print(f"\n  Processing EEG:")
                print(f"    Original shape: {eeg.shape}")
                print(f"    Target length: {self.target_length}")
                print(f"    Current length: {current_length}")
            
            if current_length > self.target_length:
                # CROP: Take first target_length samples (FIRST 5 MINUTES)
                eeg_processed = eeg_processed[:, :self.target_length]
                if self.debug:
                    print(f"    Cropped to first {self.target_length} samples")
            elif current_length < self.target_length:
                # PAD
                pad_width = self.target_length - current_length
                eeg_processed = np.pad(
                    eeg_processed, 
                    ((0, 0), (0, pad_width)), 
                    mode='edge'
                )
                if self.debug:
                    print(f"    Padded with {pad_width} samples")
        
        return eeg_processed
    
    def __len__(self) -> int:
        return len(self.eeg_files)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample with PROPER TIME ALIGNMENT"""
        if self.debug:
            print(f"\n{'='*60}")
            print(f"Loading sample {idx}")
            print('='*60)
        
        # Get EEG file
        eeg_file = self.eeg_files[idx]
        
        if self.debug:
            print(f"EEG file: {os.path.basename(eeg_file)}")
        
        # Load EEG
        eeg_raw = self._load_eeg_from_npz(eeg_file)
        
        if self.debug:
            print(f"Raw EEG shape: {eeg_raw.shape}")
            print(f"Raw EEG duration: {eeg_raw.shape[1]/self.eeg_sampling_rate:.1f}s")
        
        # Get label
        label = self._get_label(eeg_file)
        
        # Process EEG to target length (FIRST 5 MINUTES)
        eeg = self._process_eeg(eeg_raw)
        
        # Calculate actual EEG duration (should match target)
        eeg_duration = eeg.shape[1] / self.eeg_sampling_rate
        
        if self.debug:
            print(f"Processed EEG shape: {eeg.shape}")
            print(f"Processed EEG duration: {eeg_duration:.1f}s")
        
        # Get gaze fixations
        gaze_file = self.eeg_to_gaze.get(eeg_file)
        all_fixations = self.gaze_cache.get(gaze_file, []) if gaze_file else []
        
        if self.debug:
            print(f"Gaze file: {os.path.basename(gaze_file) if gaze_file else 'None'}")
            print(f"All fixations: {len(all_fixations)}")
        
        # ========== CRITICAL FIX: FILTER BY EEG TIME (x coordinate) ==========
        filtered_fixations = self._filter_fixations_by_eeg_time(all_fixations)
        
        # Convert filtered fixations to gaze map
        gaze_map = self._convert_fixations_to_gaze_map(
            fixations=filtered_fixations,
            eeg_duration=eeg_duration,
            n_channels=eeg.shape[0]
        )
        
        # ========== DEBUG: VISUALIZE ALIGNMENT ==========
        if self.debug and idx == 0:  # Only for first sample
            self._debug_visualize_alignment(eeg, gaze_map, eeg_duration, filtered_fixations)
        
        # Resize if needed (shouldn't be needed if we did everything right)
        if gaze_map.shape[1] != eeg.shape[1]:
            if self.debug:
                print(f"  WARNING: Resizing gaze map from {gaze_map.shape[1]} to {eeg.shape[1]}")
            
            original_times = np.linspace(0, 1, gaze_map.shape[1])
            new_times = np.linspace(0, 1, eeg.shape[1])
            
            gaze_resized = np.zeros((gaze_map.shape[0], eeg.shape[1]))
            for ch in range(gaze_map.shape[0]):
                interp_func = interpolate.interp1d(
                    original_times, 
                    gaze_map[ch], 
                    kind='linear',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                gaze_resized[ch] = interp_func(new_times)
            
            gaze_map = gaze_resized
        
        # Convert to tensors
        eeg_tensor = torch.FloatTensor(eeg)
        gaze_tensor = torch.FloatTensor(gaze_map)
        label_tensor = torch.LongTensor([label]).squeeze()
        
        return {
            'eeg': eeg_tensor,
            'gaze': gaze_tensor,
            'label': label_tensor,
            'file': eeg_file,
            'num_fixations': len(filtered_fixations),
            'eeg_duration': eeg_duration
        }
    
    def _debug_visualize_alignment(self, eeg: np.ndarray, gaze_map: np.ndarray, 
                                  eeg_duration: float, fixations: List[Fixation]):
        """Create debug visualization for time alignment"""
        os.makedirs('debug_plots', exist_ok=True)
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # 1. Plot EEG signal (channel average)
        time_axis = np.arange(eeg.shape[1]) / self.eeg_sampling_rate
        eeg_avg = eeg.mean(axis=0)
        
        axes[0].plot(time_axis, eeg_avg, 'b-', linewidth=0.5, alpha=0.7)
        axes[0].set_title(f'EEG Signal (Channel Average) - Duration: {eeg_duration:.1f}s')
        axes[0].set_xlabel('Time (seconds)')
        axes[0].set_ylabel('Amplitude (normalized)')
        axes[0].axvline(x=eeg_duration, color='r', linestyle='--', 
                       label=f'EEG end ({eeg_duration:.1f}s)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Plot gaze attention over time
        gaze_temporal = gaze_map.mean(axis=0)
        axes[1].plot(time_axis, gaze_temporal, 'g-', linewidth=2, alpha=0.8)
        axes[1].set_title('Gaze Attention Over Time (from x coordinate)')
        axes[1].set_xlabel('Time (seconds)')
        axes[1].set_ylabel('Attention Weight')
        axes[1].axvline(x=eeg_duration, color='r', linestyle='--')
        axes[1].set_ylim(0, 1)
        axes[1].fill_between(time_axis, 0, gaze_temporal, alpha=0.3, color='green')
        axes[1].grid(True, alpha=0.3)
        
        # Mark fixation times (x coordinates)
        if fixations:
            fixation_times = [f.x for f in fixations]  # EEG times!
            fixation_weights = [f.attention_weight for f in fixations]
            
            axes[1].scatter(fixation_times, 
                          [0.8] * len(fixation_times),  # Position near top
                          c=fixation_weights, cmap='hot', 
                          s=50, alpha=0.7, edgecolors='black',
                          label=f'Fixations ({len(fixation_times)})')
            axes[1].legend()
        
        # 3. Plot heatmap of gaze attention
        im = axes[2].imshow(gaze_map, aspect='auto', cmap='hot',
                           extent=[0, eeg_duration, 0, gaze_map.shape[0]],
                           vmin=0, vmax=1)
        axes[2].set_title('Gaze Attention Heatmap (Channels × Time)')
        axes[2].set_xlabel('Time (seconds)')
        axes[2].set_ylabel('Channel Index')
        axes[2].axvline(x=eeg_duration, color='r', linestyle='--')
        
        # Add channel labels
        channel_labels = self.channel_mapping[:gaze_map.shape[0]]
        axes[2].set_yticks(np.arange(len(channel_labels)))
        axes[2].set_yticklabels(channel_labels, fontsize=8)
        
        plt.colorbar(im, ax=axes[2], label='Attention Weight')
        
        plt.tight_layout()
        plt.savefig(f'debug_plots/time_alignment_sample0.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also create a summary text file
        with open('debug_plots/time_alignment_summary.txt', 'w') as f:
            f.write(f"=== EEG-GAZE TIME ALIGNMENT DEBUG ===\n\n")
            f.write(f"EEG Duration: {eeg_duration:.1f} seconds\n")
            f.write(f"EEG Samples: {eeg.shape[1]}\n")
            f.write(f"Gaze Map Shape: {gaze_map.shape}\n")
            f.write(f"Number of Fixations: {len(fixations)}\n\n")
            
            f.write("KEY INSIGHT:\n")
            f.write("  - Using x coordinate as EEG time reference\n")
            f.write("  - Ignoring start_time/end_time (time.time() units)\n")
            f.write("  - x = EEG time in seconds (0-300 for 5-min EEG)\n\n")
            
            f.write("Fixation Details:\n")
            for i, fix in enumerate(fixations[:10]):  # First 10
                f.write(f"  Fixation {i}:\n")
                f.write(f"    EEG time (x): {fix.x:.2f}s\n")
                f.write(f"    start_time: {fix.start_time:.2f} (time.time())\n")
                f.write(f"    Duration: {fix.duration:.2f}s\n")
                f.write(f"    Channels: {fix.channels}\n")
                f.write(f"    Weight: {fix.attention_weight:.3f}\n")
            
            # Check alignment
            max_gaze_idx = np.where(gaze_temporal > 0.1)[0]
            if len(max_gaze_idx) > 0:
                max_gaze_time = max_gaze_idx[-1] / self.eeg_sampling_rate
                f.write(f"\nAlignment Check:\n")
                f.write(f"  Last significant gaze at: {max_gaze_time:.1f}s\n")
                f.write(f"  EEG duration: {eeg_duration:.1f}s\n")
                if max_gaze_time > eeg_duration:
                    f.write(f"  ⚠️ WARNING: Gaze exceeds EEG duration!\n")
                else:
                    f.write(f"  ✓ OK: Gaze within EEG duration\n")

# Wrapper for backward compatibility
def EEGDataset_with_gaze(data_dir, indexes=None, target_length=None, debug=False, **kwargs):
    """Wrapper that matches your existing EEGDataset interface"""
    return EEGGazeFixationDataset(
        data_dir=data_dir,
        indexes=indexes,
        target_length=target_length,
        debug=debug,
        **kwargs
    )