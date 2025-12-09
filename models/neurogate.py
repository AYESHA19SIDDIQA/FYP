# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class GateDilateLayer(nn.Module):
#     def __init__(self, in_channels, kernel_size, dilation):
#         super(GateDilateLayer, self).__init__()
#         self.padding = (kernel_size - 1) * dilation
#         self.conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding=self.padding, dilation=dilation)
#         self.tanh = nn.Tanh()
#         self.sig = nn.Sigmoid()
#         self.filter = nn.Conv1d(in_channels, in_channels, 1)
#         self.gate = nn.Conv1d(in_channels, in_channels, 1)
#         self.conv2 = nn.Conv1d(in_channels, in_channels, 1)

#         # Initialize weights
#         torch.nn.init.xavier_uniform_(self.conv.weight, gain=1.0)
#         torch.nn.init.xavier_uniform_(self.filter.weight, gain=1.0)
#         torch.nn.init.xavier_uniform_(self.gate.weight, gain=1.0)
#         torch.nn.init.xavier_uniform_(self.conv2.weight, gain=1.0)

#     def forward(self, x):
#         output = self.conv(x)
#         filter = self.filter(output)
#         gate = self.gate(output)
#         tanh = self.tanh(filter)
#         sig = self.sig(gate)
#         z = tanh*sig
#         z = z[:,:,:-self.padding]
#         z = self.conv2(z)
#         x = x + z
#         return x

# class GateDilate(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, dilation_rates):
#         super(GateDilate, self).__init__()
#         self.layers = nn.ModuleList()
#         dilations = [2**i for i in range(dilation_rates)]
#         self.conv1d = nn.Conv1d(in_channels, out_channels, 1)
#         for dilation in dilations:
#             self.layers.append(GateDilateLayer(out_channels, kernel_size, dilation))
#         torch.nn.init.xavier_uniform_(self.conv1d.weight, gain=1.0)

#     def forward(self, x):
#         x = self.conv1d(x)
#         for layer in self.layers:
#             x = layer(x)
#         return x

# class ResConv(nn.Module):
#     def __init__(self, in_channels):
#         super(ResConv, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=8, kernel_size=5, padding=2)
#         self.bn1 = nn.BatchNorm1d(8)
#         self.conv2 = nn.Conv1d(in_channels=in_channels + 8, out_channels=16, kernel_size=5, padding=2)
#         self.bn2 = nn.BatchNorm1d(16)

#         nn.init.xavier_uniform_(self.conv1.weight)
#         nn.init.xavier_uniform_(self.conv2.weight)

#     def forward(self, input):
#         x1 = F.relu(self.bn1(self.conv1(input)))
#         x1 = torch.cat((x1, input), dim=1)
#         x2 = F.relu(self.bn2(self.conv2(x1)))
#         return torch.cat((x2, x1), dim=1)

# class NeuroGATE(nn.Module):
#     def __init__(self):
#         super(NeuroGATE, self).__init__()
#         self.res_conv1 = ResConv(44)
#         self.gate_dilate1 = GateDilate(44, 68, 3, 8)
#         self.res_conv2 = ResConv(20)
#         self.gate_dilate2 = GateDilate(20, 20 + 24, 3, 8)
#         self.res_conv3 = ResConv(20)
#         self.conv1 = nn.Conv1d(in_channels=68, out_channels=20, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm1d(44)
#         self.conv2 = nn.Conv1d(in_channels=20 + 24, out_channels=20, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm1d(20)
#         self.conv3 = nn.Conv1d(in_channels=20 + 24, out_channels=20, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm1d(20)
#         self.bn4 = nn.BatchNorm1d(20)
#         self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(20, 4, dropout=0.5, batch_first=True), 2)
#         self.fc = nn.Linear(20, 2)

#         nn.init.xavier_uniform_(self.conv1.weight)
#         nn.init.xavier_uniform_(self.conv2.weight)
#         nn.init.xavier_uniform_(self.conv3.weight)
#         nn.init.xavier_uniform_(self.fc.weight)

#     def forward(self, x):
#         # Pool Fusion Block
#         x1 = F.avg_pool1d(x, kernel_size=5, stride=5)
#         x2 = F.max_pool1d(x, kernel_size=5, stride=5)
#         x = torch.cat((x1, x2), dim=1)
#         x = self.bn1(x)

#         # ResGate Dilated Fusion 1
#         x1 = self.res_conv1(x)
#         x2 = self.gate_dilate1(x)
#         x = x1 + x2

#         #Apply spatial dropout
#         x = F.dropout2d(x, 0.5, training=self.training)

#         x = F.max_pool1d(x, kernel_size=5, stride=5)

#         # ConvNorm Block 1
#         x = F.relu(self.bn2(self.conv1(x)))

#         # ResGate Dilated Fusion 2
#         x1 = self.res_conv2(x)
#         x2 = self.gate_dilate2(x)
#         x = x1 + x2

#         # ConvNorm Block 2
#         x = self.bn3(self.conv2(x))

#         # ResConv Block
#         x = self.res_conv3(x)

#         x = F.max_pool1d(x, kernel_size=5, stride=5)

#         # ConvNorm Block 3
#         x = self.bn4(self.conv3(x))

#         # Encoder
#         x = x.permute(0, 2, 1)
#         x = self.encoder(x)
#         x = x.permute(0, 2, 1)

#         # Feature Aggregation Pool
#         x = torch.mean(x, dim=2)
#         x = self.fc(x)
#         return x


# updated navaal code

import torch
import torch.nn as nn
import torch.nn.functional as F

class EEG_CAM(nn.Module):
    """Class Activation Mapping for EEG signals (2D: channels × time)"""
    def _init_(self, in_channels, num_classes, eeg_channels=22):
        super(EEG_CAM, self)._init_()
        self.num_classes = num_classes
        self.eeg_channels = eeg_channels
        
        # Global Average Pooling over time
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Attention weights for each class
        self.class_fc = nn.Linear(in_channels, num_classes)
        
        # Channel-specific weights (NEW: for spatial attention)
        self.channel_fc = nn.Linear(in_channels, eeg_channels)
        
        # Initialize weights
        nn.init.normal_(self.class_fc.weight, 0, 0.01)
        nn.init.constant_(self.class_fc.bias, 0)
        nn.init.normal_(self.channel_fc.weight, 0, 0.01)
        nn.init.constant_(self.channel_fc.bias, 0)
        
    def forward(self, features, original_time_length=None):
        """
        features: (batch, feature_channels, reduced_time) from model
        Returns CAM aligned with original EEG dimensions
        """
        batch_size, feat_channels, feat_time = features.shape
        
        # Global average pooling
        gap_features = self.gap(features).squeeze(-1)  # (batch, feat_channels)
        
        # Get class-specific weights
        class_weights = self.class_fc(gap_features)  # (batch, num_classes)
        
        # Get channel-specific weights
        channel_weights = self.channel_fc(gap_features)  # (batch, eeg_channels)
        
        # Weighted sum of feature maps for each class
        cam_maps = []
        for c in range(self.num_classes):
            # Get weights for this class across batch
            class_weight = class_weights[:, c].unsqueeze(1).unsqueeze(2)  # (batch, 1, 1)
            
            # Weight features by class importance
            weighted_features = features * class_weight  # (batch, feat_channels, feat_time)
            
            # Sum across feature channels
            temporal_cam = torch.sum(weighted_features, dim=1)  # (batch, feat_time)
            
            # Add channel dimension using channel weights
            # Expand temporal CAM to channels
            temporal_cam = temporal_cam.unsqueeze(1)  # (batch, 1, feat_time)
            channel_weight = channel_weights.unsqueeze(-1)  # (batch, eeg_channels, 1)
            
            # Combine temporal and spatial attention
            combined_cam = temporal_cam * channel_weight  # (batch, eeg_channels, feat_time)
            
            # Normalize
            combined_cam = (combined_cam - combined_cam.min(dim=2, keepdim=True)[0]) / \
                          (combined_cam.max(dim=2, keepdim=True)[0] - combined_cam.min(dim=2, keepdim=True)[0] + 1e-8)
            
            cam_maps.append(combined_cam)
        
        # Stack: (batch, num_classes, channels, time)
        cam_maps = torch.stack(cam_maps, dim=1)
        
        # If we need to resize to original EEG time length
        if original_time_length is not None and feat_time != original_time_length:
            cam_maps = F.interpolate(
                cam_maps.view(batch_size * self.num_classes * self.eeg_channels, 1, -1), 
                size=original_time_length, 
                mode='linear', 
                align_corners=False
            ).view(batch_size, self.num_classes, self.eeg_channels, original_time_length)
        
        return cam_maps, class_weights

class GateDilateLayer(nn.Module):
    def _init_(self, in_channels, kernel_size, dilation):
        super(GateDilateLayer, self)._init_()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size, 
                             padding=self.padding, dilation=dilation)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.filter = nn.Conv1d(in_channels, in_channels, 1)
        self.gate = nn.Conv1d(in_channels, in_channels, 1)
        self.conv2 = nn.Conv1d(in_channels, in_channels, 1)

        torch.nn.init.xavier_uniform_(self.conv.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.filter.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.gate.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.conv2.weight, gain=1.0)

    def forward(self, x):
        output = self.conv(x)
        filter = self.filter(output)
        gate = self.gate(output)
        tanh = self.tanh(filter)
        sig = self.sig(gate)
        z = tanh * sig
        z = z[:, :, :-self.padding] if self.padding > 0 else z
        z = self.conv2(z)
        x = x + z
        return x

class GateDilate(nn.Module):
    def _init_(self, in_channels, out_channels, kernel_size, dilation_rates):
        super(GateDilate, self)._init_()
        self.layers = nn.ModuleList()
        dilations = [2**i for i in range(dilation_rates)]
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1)
        for dilation in dilations:
            self.layers.append(GateDilateLayer(out_channels, kernel_size, dilation))
        torch.nn.init.xavier_uniform_(self.conv1d.weight, gain=1.0)

    def forward(self, x):
        x = self.conv1d(x)
        for layer in self.layers:
            x = layer(x)
        return x

class ResConv(nn.Module):
    def _init_(self, in_channels):
        super(ResConv, self)._init_()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=8, 
                              kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(in_channels=in_channels + 8, out_channels=16, 
                              kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(16)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, input):
        x1 = F.relu(self.bn1(self.conv1(input)))
        x1 = torch.cat((x1, input), dim=1)
        x2 = F.relu(self.bn2(self.conv2(x1)))
        return torch.cat((x2, x1), dim=1)

class NeuroGATE_Gaze(nn.Module):
    def _init_(self, n_chan: int = 22, n_outputs: int = 2):
        super(NeuroGATE_Gaze, self)._init_()
        
        self.n_chan = n_chan
        self.n_outputs = n_outputs
        
        # Keep your original architecture up to encoder
        fused_ch = 2 * n_chan
        res1_in = fused_ch
        res1_out = res1_in + 24
        
        self.res_conv1 = ResConv(res1_in)
        self.gate_dilate1 = GateDilate(res1_in, res1_out, 3, 8)
        
        self.conv1 = nn.Conv1d(in_channels=res1_out, out_channels=20, 
                              kernel_size=3, padding=1)
        
        self.res_conv2 = ResConv(20)
        self.gate_dilate2 = GateDilate(20, 20 + 24, 3, 8)
        
        self.res_conv3 = ResConv(20)
        
        self.bn1 = nn.BatchNorm1d(fused_ch)
        self.bn2 = nn.BatchNorm1d(20)
        self.conv2 = nn.Conv1d(in_channels=20 + 24, out_channels=20, 
                              kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(20)
        self.conv3 = nn.Conv1d(in_channels=20 + 24, out_channels=20, 
                              kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(20)
        
        # Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(20, 4, dropout=0.5, batch_first=True), 2
        )
        
        # Add CAM layer
        self.cam_layer = EEG_CAM(in_channels=20, num_classes=n_outputs, eeg_channels=n_chan)
        
        # Final classification layer
        self.fc = nn.Linear(20, n_outputs)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        
    def forward(self, x, return_cam=False):
        """
        x: (batch_size, channels, time_steps) - Original EEG
        """
        batch_size, channels, original_time = x.shape
        
        # Store original input shape
        original_shape = (batch_size, channels, original_time)
        
        # Your original forward pass until encoder
        x1 = F.avg_pool1d(x, kernel_size=5, stride=5)
        x2 = F.max_pool1d(x, kernel_size=5, stride=5)
        x = torch.cat((x1, x2), dim=1)
        x = self.bn1(x)
        
        x1 = self.res_conv1(x)
        x2 = self.gate_dilate1(x)
        x = x1 + x2
        x = F.dropout2d(x, 0.5, training=self.training)
        x = F.max_pool1d(x, kernel_size=5, stride=5)
        
        x = F.relu(self.bn2(self.conv1(x)))
        
        x1 = self.res_conv2(x)
        x2 = self.gate_dilate2(x)
        x = x1 + x2
        x = self.bn3(self.conv2(x))
        
        x = self.res_conv3(x)
        x = F.max_pool1d(x, kernel_size=5, stride=5)
        
        x = self.bn4(self.conv3(x))
        
        # Pass through encoder
        x = x.permute(0, 2, 1)  # (batch, time, features)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)  # Back to (batch, features, time)
        
        # Store features for CAM
        features_for_cam = x
        
        # Final classification
        x_pooled = torch.mean(x, dim=2)  # Global average pooling
        logits = self.fc(x_pooled)
        
        if return_cam:
            # Generate CAM aligned with original EEG time
            cam_maps, _ = self.cam_layer(features_for_cam, original_time_length=original_time)
            
            # Ensure cam_maps has correct shape: (batch, classes, channels, time)
            if cam_maps.shape[2] != channels:
                # Resize channel dimension if needed
                cam_maps = F.interpolate(
                    cam_maps, 
                    size=(channels, original_time), 
                    mode='bilinear', 
                    align_corners=False
                )
            
            return logits, cam_maps
        
        return logits