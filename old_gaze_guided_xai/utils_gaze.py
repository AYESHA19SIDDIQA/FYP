import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def create_gaze_data_from_eeg(eeg_data, method='random', **kwargs):
    """
    Create synthetic gaze data if you don't have real gaze annotations
    
    Methods:
    - 'random': Random attention
    - 'amplitude': Attention on high amplitude regions
    - 'frequency': Attention on specific frequency bands
    - 'event': Attention on simulated events
    """
    n_samples, n_channels, n_time = eeg_data.shape
    gaze_data = np.zeros_like(eeg_data)
    
    if method == 'random':
        # Random gaze patterns
        for i in range(n_samples):
            # Create random fixations
            n_fixations = np.random.randint(3, 10)
            for _ in range(n_fixations):
                channel = np.random.randint(0, n_channels)
                time_start = np.random.randint(0, n_time - 1000)
                duration = np.random.randint(200, 1000)
                intensity = np.random.uniform(0.5, 1.0)
                
                gaze_data[i, channel, time_start:time_start+duration] = intensity
    
    elif method == 'amplitude':
        # Focus on high amplitude regions
        for i in range(n_samples):
            # Calculate amplitude envelope
            amplitude = np.abs(eeg_data[i])
            avg_amplitude = amplitude.mean(axis=0)
            
            # Threshold for high amplitude
            threshold = np.percentile(avg_amplitude, 75)
            high_amp_mask = avg_amplitude > threshold
            
            # Spread attention across channels
            for ch in range(n_channels):
                gaze_data[i, ch] = high_amp_mask.astype(float)
    
    elif method == 'frequency':
        # Focus on specific frequency bands
        fs = kwargs.get('fs', 100)  # Sampling rate
        target_band = kwargs.get('target_band', (8, 13))  # Alpha band
        
        for i in range(n_samples):
            for ch in range(n_channels):
                # Compute spectrogram
                from scipy import signal
                f, t, Sxx = signal.spectrogram(
                    eeg_data[i, ch], 
                    fs=fs, 
                    nperseg=256
                )
                
                # Find target frequency band
                band_mask = (f >= target_band[0]) & (f <= target_band[1])
                band_power = Sxx[band_mask].mean(axis=0)
                
                # Resize to original time
                band_power_resized = np.interp(
                    np.linspace(0, n_time, len(band_power)),
                    np.arange(len(band_power)),
                    band_power
                )
                
                gaze_data[i, ch] = band_power_resized
    
    # Normalize each sample
    for i in range(n_samples):
        if gaze_data[i].max() > 0:
            gaze_data[i] = gaze_data[i] / gaze_data[i].max()
    
    return gaze_data

def save_gaze_data(eeg_files, gaze_data, suffix='_gaze'):
    """
    Save gaze data as .npz files alongside EEG files
    """
    for i, eeg_file in enumerate(eeg_files):
        base = os.path.splitext(eeg_file)[0]
        gaze_file = f"{base}{suffix}.npz"
        
        np.savez_compressed(gaze_file, gaze=gaze_data[i])
        print(f"Saved gaze data to {gaze_file}")

def calculate_alignment_metrics(model, dataloader, device):
    """
    Calculate alignment metrics between model CAM and gaze
    """
    model.eval()
    correlations = []
    similarities = []
    
    with torch.no_grad():
        for batch in dataloader:
            eeg = batch['eeg'].to(device).float()
            gaze = batch['gaze'].to(device).float()
            
            # Get model CAM
            _, cam_maps = model(eeg, return_cam=True)
            
            # For each sample in batch
            for i in range(eeg.shape[0]):
                # Get CAM for predicted class
                cam_flat = cam_maps[i].mean(dim=0).flatten().cpu().numpy()
                gaze_flat = gaze[i].flatten().cpu().numpy()
                
                # Correlation
                corr = np.corrcoef(cam_flat, gaze_flat)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
                
                # Cosine similarity
                cos_sim = np.dot(cam_flat, gaze_flat) / (
                    np.linalg.norm(cam_flat) * np.linalg.norm(gaze_flat) + 1e-8
                )
                similarities.append(cos_sim)
    
    return {
        'correlation_mean': np.mean(correlations),
        'correlation_std': np.std(correlations),
        'similarity_mean': np.mean(similarities),
        'similarity_std': np.std(similarities)
    }