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

@dataclass
class Fixation:
    """Represents a single gaze fixation from your JSON data"""
    start_time: float
    end_time: float
    duration: float
    x: float
    y: float
    channels: List[str]
    num_points: int
    
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
    
    Uses the same structure as your existing EEGDataset but adds gaze support.
    """
    
    def __init__(self, data_dir: str, 
                 indexes: Optional[List[int]] = None,
                 target_length: Optional[int] = None,
                 gaze_json_dir: Optional[str] = None,
                 gaze_json_pattern: str = "*.json",
                 eeg_sampling_rate: float = 100.0,
                 channel_mapping: Optional[Dict[str, int]] = None,
                 **kwargs):
        """
        Args:
            data_dir: Directory with EEG .npz files (from get_datanpz)
            indexes: Subset indices to use
            target_length: Target time length for EEG
            gaze_json_dir: Directory with gaze JSON files
            gaze_json_pattern: Pattern to match gaze files
            eeg_sampling_rate: EEG sampling rate (from data_description)
            channel_mapping: Map from channel names to indices
        """
        super().__init__()
        
        self.data_dir = data_dir
        self.target_length = target_length
        self.eeg_sampling_rate = eeg_sampling_rate
        
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
            print(f"Using {len(self.eeg_files)} files after index filtering")
        
        # Find gaze JSON files
        gaze_dir = gaze_json_dir if gaze_json_dir else data_dir
        self.gaze_files = sorted(glob.glob(os.path.join(gaze_dir, gaze_json_pattern)))
        print(f"Found {len(self.gaze_files)} gaze JSON files")
        
        # Create mapping from EEG files to gaze files
        self.eeg_to_gaze = self._create_file_mappings()
        
        # Preload gaze data
        self.gaze_cache = {}
        self._preload_gaze_data()
    
    def _create_nmt_channel_mapping(self) -> List[str]:
        """Create channel mapping for NMT dataset (from your pipeline)"""
        # Based on your NMT_CHANNELS - adjust as needed
        return [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
            'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
            'Fz', 'Cz', 'Pz', 'A1', 'A2', 'EKG'
        ]
    
    def _get_eeg_files(self) -> List[str]:
        """Get list of EEG .npz files"""
        if self.df is not None and 'filename' in self.df.columns:
            # Get files from data.csv
            files = []
            for filename in self.df['filename']:
                file_path = os.path.join(self.data_dir, filename)
                if os.path.exists(file_path):
                    files.append(file_path)
            return files
        else:
            # Fallback: list all .npz files
            return sorted(glob.glob(os.path.join(self.data_dir, '*.npz')))
    
    def _create_file_mappings(self) -> Dict[str, Optional[str]]:
        """Map EEG files to corresponding gaze JSON files"""
        mappings = {}
        
        for eeg_file in self.eeg_files:
            eeg_basename = os.path.basename(eeg_file)
            eeg_name = os.path.splitext(eeg_basename)[0]
            
            # Try different matching strategies
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
            
            if not gaze_file:
                print(f"  Warning: No gaze match for {eeg_basename}")
        
        # Count matches
        matches = sum(1 for g in mappings.values() if g is not None)
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
                            start_time=fix_dict['start_time'],
                            end_time=fix_dict['end_time'],
                            duration=fix_dict['duration'],
                            x=fix_dict['x'],
                            y=fix_dict['y'],
                            channels=fix_dict.get('channel', []),
                            num_points=fix_dict.get('num_points', 1)
                        )
                        all_fixations.append(fixation)
                
                self.gaze_cache[gaze_file] = all_fixations
                
            except Exception as e:
                print(f"  Error loading {gaze_file}: {e}")
                self.gaze_cache[gaze_file] = []
    
    def _load_eeg_from_npz(self, file_path: str) -> np.ndarray:
        """Load EEG from .npz file (compatible with get_datanpz output)"""
        data = np.load(file_path)
        
        # Your pipeline saves EEG as numpy arrays
        # Try common keys
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
        if eeg.shape[0] > eeg.shape[1]:  # Transpose if needed
            eeg = eeg.T
        
        return eeg.astype(np.float32)
    
    def _get_label(self, file_path: str) -> int:
        """Get label for EEG file"""
        if self.df is not None:
            # Get label from data.csv
            filename = os.path.basename(file_path)
            row = self.df[self.df['filename'] == filename]
            if not row.empty:
                label_cols = ['label', 'Label', 'y', 'target']
                for col in label_cols:
                    if col in row.columns:
                        return int(row[col].iloc[0])
        
        # Try to get from .npz file
        try:
            data = np.load(file_path)
            for key in ['label', 'y', 'target']:
                if key in data:
                    return int(data[key])
        except:
            pass
        
        # Default
        return 0
    
    def _convert_fixations_to_gaze_map(self, 
                                      fixations: List[Fixation],
                                      eeg_duration: float,
                                      n_channels: int) -> np.ndarray:
        """
        Convert fixations to 2D gaze map matching EEG shape
        """
        n_time = int(eeg_duration * self.eeg_sampling_rate)
        gaze_map = np.zeros((n_channels, n_time), dtype=np.float32)
        
        if not fixations:
            return gaze_map + 0.01  # Baseline attention
        
        # Find time range of fixations
        all_times = []
        for fix in fixations:
            all_times.extend([fix.start_time, fix.end_time])
        
        if not all_times:
            return gaze_map
        
        min_time = min(all_times)
        max_time = max(all_times)
        time_range = max_time - min_time
        
        for fixation in fixations:
            # Normalize fixation times to [0, 1]
            norm_start = (fixation.start_time - min_time) / time_range
            norm_end = (fixation.end_time - min_time) / time_range
            
            # Convert to sample indices
            start_idx = int(norm_start * n_time)
            end_idx = int(norm_end * n_time)
            start_idx = max(0, min(start_idx, n_time - 1))
            end_idx = max(start_idx + 1, min(end_idx, n_time))
            
            # Get channel indices
            channel_indices = []
            for ch_name in fixation.channels:
                if ch_name in self.channel_to_index:
                    channel_indices.append(self.channel_to_index[ch_name])
                else:
                    # Fuzzy matching
                    for map_ch in self.channel_to_index:
                        if ch_name.upper() == map_ch.upper():
                            channel_indices.append(self.channel_to_index[map_ch])
                            break
            
            if not channel_indices:
                channel_indices = list(range(n_channels))
                weight = fixation.attention_weight * 0.3
            else:
                weight = fixation.attention_weight
            
            # Apply Gaussian attention
            duration_samples = end_idx - start_idx
            if duration_samples > 0:
                center = (start_idx + end_idx) / 2
                sigma = duration_samples / 4
                
                time_indices = np.arange(start_idx, end_idx)
                gaussian = np.exp(-0.5 * ((time_indices - center) / sigma) ** 2)
                
                for ch_idx in channel_indices:
                    gaze_map[ch_idx, start_idx:end_idx] += gaussian * weight
        
        # Normalize
        if gaze_map.max() > 0:
            gaze_map = gaze_map / gaze_map.max()
        
        gaze_map = gaze_map * 0.9 + 0.1  # Add baseline
        
        return gaze_map
    
    def _process_eeg(self, eeg: np.ndarray) -> np.ndarray:
        """Process EEG: normalize and crop/pad"""
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
        
        # Crop or pad
        if self.target_length is not None:
            current_length = eeg_processed.shape[1]
            if current_length > self.target_length:
                start = np.random.randint(0, current_length - self.target_length)
                eeg_processed = eeg_processed[:, start:start + self.target_length]
            elif current_length < self.target_length:
                pad_width = self.target_length - current_length
                eeg_processed = np.pad(
                    eeg_processed, 
                    ((0, 0), (0, pad_width)), 
                    mode='edge'
                )
        
        return eeg_processed
    
    def __len__(self) -> int:
        return len(self.eeg_files)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample"""
        # Get EEG file
        eeg_file = self.eeg_files[idx]
        
        # Load EEG
        eeg_raw = self._load_eeg_from_npz(eeg_file)
        
        # Get label
        label = self._get_label(eeg_file)
        
        # Process EEG
        eeg = self._process_eeg(eeg_raw)
        
        # Calculate duration
        eeg_duration = eeg.shape[1] / self.eeg_sampling_rate
        
        # Get gaze fixations
        gaze_file = self.eeg_to_gaze.get(eeg_file)
        fixations = self.gaze_cache.get(gaze_file, []) if gaze_file else []
        
        # Convert to gaze map
        gaze_map = self._convert_fixations_to_gaze_map(
            fixations=fixations,
            eeg_duration=eeg_duration,
            n_channels=eeg.shape[0]
        )
        
        # Resize if needed
        if gaze_map.shape[1] != eeg.shape[1]:
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
            'num_fixations': len(fixations)
        }

# Wrapper for backward compatibility
def EEGDataset_with_gaze(data_dir, indexes=None, target_length=None, **kwargs):
    """Wrapper that matches your existing EEGDataset interface"""
    return EEGGazeFixationDataset(
        data_dir=data_dir,
        indexes=indexes,
        target_length=target_length,
        **kwargs
    )