import torch
import numpy as np
import os
import sys

# Add your project to path
sys.path.append('.')
sys.path.append('..')

# Import your existing modules
try:
    from cereprocess.datasets.pipeline import general_pipeline
    from cereprocess.train.xloop import get_datanpz
    from cereprocess.train.misc import  def_dev, def_hyp
except ImportError:
    print("Warning: Could not import your existing modules. Using placeholders.")
    
    # Placeholder functions
    def def_dev():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def def_hyp(batch_size=32, lr=0.0001, epochs=50, accum_iter=1):
        return {
            "batch_size": batch_size,
            "lr": lr,
            "epochs": epochs,
            "accum_iter": accum_iter
        }
    
    def get_def_ds(mins):
        return None, ("E:/NMT_events/NMT_events/edf", "NMT", "description", "results"), None

# Import gaze modules
from neurogate_gaze import NeuroGATE_Gaze
from eeg_gaze_fixations import EEGGazeFixationDataset
from train_gaze_integrated import train_gaze_guided_your_pipeline, GazeGuidedLoss, evaluate_gaze_guided

def get_dataloaders_with_gaze_fixations(datadir, batch_size, seed, 
                                       target_length=None, indexes=None,
                                       gaze_json_dir=None, **kwargs):
    """
    Modified version of your get_dataloaders that includes gaze data
    """
    train_dir = os.path.join(datadir, 'train')
    eval_dir = os.path.join(datadir, 'eval')
    
    # Create datasets with gaze support
    trainset = EEGGazeFixationDataset(
        data_dir=train_dir,
        indexes=indexes,
        target_length=target_length,
        gaze_json_dir=gaze_json_dir,
        eeg_sampling_rate=100.0,  # From your data_description
        **kwargs
    )
    
    evalset = EEGGazeFixationDataset(
        data_dir=eval_dir,
        indexes=indexes,
        target_length=target_length,
        gaze_json_dir=gaze_json_dir,
        eeg_sampling_rate=100.0,
        **kwargs
    )
    
    # Define a proper function instead of lambda
    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)
    
    # Create DataLoaders - FIX: Remove num_workers or use 0 for debugging
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid pickle issues initially
        worker_init_fn=worker_init_fn,  # Use defined function
        pin_memory=torch.cuda.is_available()
    )
    
    eval_loader = torch.utils.data.DataLoader(
        evalset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 initially
        worker_init_fn=worker_init_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, eval_loader

def main():
    """Main function that integrates gaze with your existing pipeline"""
    # Your existing setup
    device = def_dev()
    print(f"Using device: {device}")
    
    mins = 5
    length = mins * 3000  # 5 minutes at 100Hz
    input_size = (22, length)
    
    # Get your datasets
    try:
        tuh, nmt, nmt_4k = get_def_ds(mins)
        curr_data = nmt  # Using NMT dataset
        print(f"Using dataset: {curr_data[1]}")
    except:
        print("Using default dataset path")
        curr_data = ("E:/NMT_events/NMT_events/edf", "NMT", "description", "results")
    
    print("\nStep 1: Preprocessing EEG data with your pipeline...")
    try:
        data_dir, data_description = get_datanpz(
            curr_data[0], 
            curr_data[3], 
            general_pipeline(dataset='NMT', length_minutes=mins), 
            input_size
        )
        print(f"Preprocessed data saved to: {data_dir}")
        print(f"Data description: {data_description}")
    except Exception as e:
        print(f"Error in get_datanpz: {e}")
        print("Using placeholder data directory")
        data_dir = "./preprocessed_data"
        data_description = {'channel_no': 22, 'sampling_rate': 100, 'time_span': 300}
    
    # Path to gaze JSON files
    gaze_json_dir = "results/gaze"  # UPDATE THIS TO YOUR ACTUAL PATH
    print(f"Gaze JSON directory: {gaze_json_dir}")
    
    print("\nStep 2: Creating dataloaders with gaze data...")
    hyps = def_hyp(batch_size=8, epochs=10, lr=0.0003, accum_iter=1)
    
    try:
        train_loader, eval_loader = get_dataloaders_with_gaze_fixations(
            datadir=data_dir,
            batch_size=hyps['batch_size'],
            seed=42,
            target_length=length,
            gaze_json_dir=gaze_json_dir
        )
        
        # Check a sample
        sample_batch = next(iter(train_loader))
        print(f"\nSample batch shapes:")
        print(f"  EEG: {sample_batch['eeg'].shape}")
        print(f"  Gaze: {sample_batch['gaze'].shape}")
        print(f"  Labels: {sample_batch['label'].shape}")
        print(f"  Label values: {sample_batch['label'][:5]}")
        print(f"  Has gaze data: {(sample_batch['gaze'].sum(dim=(1,2)) > 0).all().item()}")
        
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        print("Creating dummy data for testing...")
        # Create dummy data for testing
        import torch.utils.data as data
        class DummyDataset(data.Dataset):
            def __len__(self): return 100
            def __getitem__(self, idx):
                return {
                    'eeg': torch.randn(22, length),
                    'gaze': torch.rand(22, length),
                    'label': torch.randint(0, 2, (1,)).squeeze()
                }
        train_loader = data.DataLoader(DummyDataset(), batch_size=8, shuffle=True)
        eval_loader = data.DataLoader(DummyDataset(), batch_size=8, shuffle=False)
    
    print("\nStep 3: Initializing NeuroGATE_Gaze model...")
    model = NeuroGATE_Gaze(
        n_chan=data_description.get('channel_no', 22),
        n_outputs=2  # Binary classification
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    try:
        with torch.no_grad():
            test_eeg = sample_batch['eeg'][:2].to(device)
            logits, cam_maps = model(test_eeg, return_cam=True)
            
            print(f"  Input EEG shape: {test_eeg.shape}")
            print(f"  Output logits shape: {logits.shape}")
            print(f"  CAM maps shape: {cam_maps.shape}")
            
            # Check if CAM matches gaze shape
            test_gaze = sample_batch['gaze'][:2].to(device)
            if cam_maps.shape[2:] == test_gaze.shape[1:]:
                print("   CAM and gaze shapes match!")
            else:
                print(f"   Shape mismatch!")
                print(f"    CAM shape: {cam_maps.shape[2:]}")
                print(f"    Gaze shape: {test_gaze.shape[1:]}")
    except Exception as e:
        print(f"Error in forward pass test: {e}")
    
    print("\nStep 4: Training with gaze guidance...")
    
    # You'll need to adapt this part to your existing training setup
    # For now, just showing the basic structure
    
    # Example of how to integrate with your existing training
    print("To integrate with your existing pipeline:")
    print("1. Use NeuroGATE_Gaze instead of NeuroGATE")
    print("2. Use EEGGazeFixationDataset instead of EEGDataset")
    print("3. Use train_gaze_guided_your_pipeline in your oneloop() function")
    print("4. Add gaze_weight parameter to your hyperparameters")
    
    print("\nExample usage in your existing code:")
    print("""
    # Replace:
    # model = NeuroGATE().to(device)
    # With:
    model = NeuroGATE_Gaze(n_chan=22, n_outputs=2).to(device)
    
    # Replace:
    # train_loader, eval_loader = get_dataloaders(...)
    # With:
    train_loader, eval_loader = get_dataloaders_with_gaze_fixations(
        data_dir, batch_size, seed, target_length=length,
        gaze_json_dir="your_gaze_json_path"
    )
    
    # In your training loop, the loss will automatically use gaze guidance
    # when you call model(x, return_cam=True) and pass gaze maps
    """)
    
    print("\nGaze-guided system ready for integration!")

if __name__ == "__main__":
    main()