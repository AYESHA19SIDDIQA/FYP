import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import json

def compute_class_weights(labels, device='cpu'):
    """
    Compute class weights for handling class imbalance.
    Classes with fewer samples will get higher weights.
    
    Args:
        labels (array-like): All training labels (list, numpy array, or tensor)
        device (str): Device to place the weights tensor on ('cpu' or 'cuda')
    
    Returns:
        torch.Tensor: Class weights tensor of shape [num_classes]
    
    Example:
        # Collect all training labels
        all_labels = []
        for batch in train_loader:
            all_labels.extend(batch['label'].numpy())
        
        # Compute balanced weights
        class_weights = compute_class_weights(all_labels, device='cuda')
        
        # Use with GazeGuidedLoss
        criterion = GazeGuidedLoss(class_weights=class_weights)
    """
    # Convert to numpy array if needed
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)
    
    # Get unique classes and their counts
    classes, counts = np.unique(labels, return_counts=True)
    
    # Compute weights: inversely proportional to class frequency
    # weight = total_samples / (num_classes * class_count)
    total_samples = len(labels)
    num_classes = len(classes)
    weights = total_samples / (num_classes * counts)
    
    # Normalize weights so they sum to num_classes (optional, for stability)
    weights = weights * (num_classes / weights.sum())
    
    # Create tensor with weights in proper class order
    class_weights = torch.zeros(num_classes, dtype=torch.float32)
    for idx, cls in enumerate(classes):
        class_weights[int(cls)] = weights[idx]
    
    return class_weights.to(device)

class GazeGuidedLoss(nn.Module):
    """
    Loss function for gaze-guided training
    Combines classification loss with gaze alignment loss
    
    Args:
        cls_weight (float): Weight for classification loss component (default: 1.0)
        gaze_weight (float): Weight for gaze alignment loss component (default: 0.3)
        class_weights (torch.Tensor, optional): Weights for each class to handle class imbalance.
            Should be a 1D tensor of shape [num_classes] with positive values.
            Higher values give more importance to the corresponding class.
            Use compute_class_weights() function to automatically calculate balanced weights.
            If None, all classes are weighted equally (default: None)
    """
    def __init__(self, cls_weight=1.0, gaze_weight=0.3, class_weights=None):
        super(GazeGuidedLoss, self).__init__()
        self.cls_weight = cls_weight
        self.gaze_weight = gaze_weight
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    def _gaze_alignment_loss(self, model_cam, gaze_map, label):
        """
        Compute alignment between model CAM and gaze map
        
        Args:
            model_cam: (batch, classes, channels, time) - model attention
            gaze_map: (batch, channels, time) - doctor's gaze
            label: (batch,) - true labels
        """
        batch_size = model_cam.shape[0]
        
        # For each sample, get CAM for the true class
        true_cams = []
        for i in range(batch_size):
            true_class = label[i].item()
            true_cam = model_cam[i, true_class]  # (channels, time)
            true_cams.append(true_cam)
        
        true_cams = torch.stack(true_cams)  # (batch, channels, time)
        
        # Normalize both to probability distributions
        model_flat = true_cams.flatten(1) + 1e-8
        gaze_flat = gaze_map.flatten(1) + 1e-8
        
        model_norm = model_flat / model_flat.sum(dim=1, keepdim=True)
        gaze_norm = gaze_flat / gaze_flat.sum(dim=1, keepdim=True)
        
        # KL divergence (symmetric)
        kl1 = F.kl_div(
            torch.log(model_norm + 1e-8),
            gaze_norm, 
            reduction='batchmean',
            log_target=False
        )
        kl2 = F.kl_div(
            torch.log(gaze_norm + 1e-8), 
            model_norm, 
            reduction='batchmean',
            log_target=False
        )
        kl_loss = (kl1 + kl2) / 2
        
        # Cosine similarity
        cos_sim = F.cosine_similarity(model_flat, gaze_flat, dim=1)
        cos_loss = 1 - cos_sim.mean()
        
        # Combined alignment loss
        alignment_loss = 0.7 * kl_loss + 0.3 * cos_loss
        
        return alignment_loss
    
    def forward(self, logits, label, model_cam=None, gaze_map=None):
        # Classification loss
        cls_loss = self.ce_loss(logits, label)
        
        # Gaze alignment loss (if gaze data is provided)
        gaze_loss = torch.tensor(0.0).to(logits.device)
        if model_cam is not None and gaze_map is not None:
            # Check if we have valid gaze data (not all zeros)
            if gaze_map.sum() > 0:
                gaze_loss = self._gaze_alignment_loss(model_cam, gaze_map, label)
        
        # Total loss
        total_loss = self.cls_weight * cls_loss + self.gaze_weight * gaze_loss
        
        return total_loss, {
            'cls_loss': cls_loss.item(),
            'gaze_loss': gaze_loss.item() if gaze_loss != 0 else 0,
            'total_loss': total_loss.item()
        }

def train_gaze_guided_your_pipeline(model, train_loader, eval_loader, optimizer, criterion, 
                                   epochs, history, metrics, device, save_path, earlystopping,
                                   accum_iter=1, scheduler=None, save_best_acc=False,
                                   gaze_weight=0.3, class_weights=None):
    """
    Modified version of your train() function that includes gaze guidance
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        eval_loader: DataLoader for evaluation data
        optimizer: Optimizer for training
        criterion: Loss criterion (not used, gaze_criterion created internally)
        epochs: Number of training epochs
        history: History object to track metrics
        metrics: Metrics object to compute evaluation metrics
        device: Device to run training on ('cuda' or 'cpu')
        save_path: Path to save model checkpoints
        earlystopping: EarlyStopping object for early stopping
        accum_iter (int): Gradient accumulation iterations (default: 1)
        scheduler: Learning rate scheduler (optional)
        save_best_acc (bool): Whether to save best accuracy model (default: False)
        gaze_weight (float): Weight for gaze alignment loss (default: 0.3)
        class_weights (torch.Tensor, optional): Weights for each class to handle class imbalance.
            Should be a 1D tensor of shape [num_classes]. Use the compute_class_weights() function
            to automatically compute balanced weights from training labels (default: None)
    """
    model = model.to(device)
    
    # Create gaze-guided loss
    gaze_criterion = GazeGuidedLoss(cls_weight=1.0, gaze_weight=gaze_weight, class_weights=class_weights)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_gaze_loss = 0
        train_cls_loss = 0
        metrics.reset()
        batch_idx = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} Training')
        
        for batch in pbar:
            # Get data - now includes gaze
            eeg = batch['eeg'].to(device).float()
            target = batch['label'].to(device).long()  # Changed: .long() instead of .float()
            gaze = batch['gaze'].to(device).float()
            
            # Forward pass with CAM
            output, cam_maps = model(eeg, return_cam=True)
            
            # Compute gaze-guided loss
            loss, loss_dict = gaze_criterion(output, target, cam_maps, gaze)
            
            # Backward pass
            loss.backward()
            
            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
            
            # Track losses
            train_loss += loss_dict['total_loss']
            train_gaze_loss += loss_dict['gaze_loss']
            train_cls_loss += loss_dict['cls_loss']
            
            # Compute metrics
            output_softmax = torch.nn.functional.softmax(output, dim=-1)
            _, predicted = torch.max(output_softmax, 1)
            label_check = target  # Already long tensor
            
            metrics.update(label_check, predicted)
            batch_idx += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'gaze': f"{loss_dict['gaze_loss']:.4f}",
                'acc': f"{100 * (predicted == label_check).float().mean():.2f}%"
            })
            
            # Clear memory
            del eeg, target, gaze, output, cam_maps, loss
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        # Calculate epoch metrics
        train_loss /= len(train_loader)
        train_gaze_loss /= len(train_loader)
        train_cls_loss /= len(train_loader)
        
        results = metrics.compute()
        results.update({
            "loss": train_loss,
            "gaze_loss": train_gaze_loss,
            "cls_loss": train_cls_loss
        })
        history.update(results, 'train')
        
        # Validation
        val_loss = evaluate_gaze_guided(model, eval_loader, gaze_criterion, device, metrics, history)
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{epochs}:')
        print(f'  Train - Loss: {train_loss:.4f}, Gaze: {train_gaze_loss:.4f}, Cls: {train_cls_loss:.4f}')
        print(f'  Val   - Loss: {val_loss:.4f}')
        
        # Early stopping
        if save_best_acc:
            earlystopping(history.history["val"]["accuracy"][-1], model, save_best_acc=True)
        else:
            earlystopping(val_loss, model)
        
        if earlystopping.early_stop:
            print("Early stopping")
            break
        
        if scheduler:
            scheduler.step(val_loss)
        
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Load best model
    model.load_state_dict(torch.load(earlystopping.path))
    
    return model

def evaluate_gaze_guided(model, val_loader, criterion, device, metrics, history):
    """Evaluation function for gaze-guided model"""
    model.to(device)
    val_loss = 0.0
    model.eval()
    metrics.reset()
    
    actual = torch.tensor([], device=device, dtype=torch.long)
    pred = torch.tensor([], device=device, dtype=torch.long)
    
    with torch.no_grad():
        for batch in val_loader:
            eeg = batch['eeg'].to(device).float()
            target = batch['label'].to(device).long()  # Changed: .long()
            gaze = batch['gaze'].to(device).float()
            
            # Forward pass with CAM
            output, cam_maps = model(eeg, return_cam=True)
            
            # Compute loss
            loss, loss_dict = criterion(output, target, cam_maps, gaze)
            val_loss += loss_dict['total_loss']
            
            # Compute metrics
            output_softmax = torch.nn.functional.softmax(output, dim=-1)
            _, predicted = output_softmax.max(1)
            label_check = target  # Already long tensor
            
            metrics.update(label_check, predicted)
            actual = torch.cat([actual, label_check])
            pred = torch.cat([pred, predicted])
    
    val_loss /= len(val_loader)
    results = metrics.compute()
    results.update({"loss": val_loss})
    history.update(results, 'val')
    history.update_cm(actual.tolist(), pred.tolist())
    
    return val_loss