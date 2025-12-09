import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== UPDATED EEG ATTENTION MODULE (Multi-Resolution) ==========

class EEG_Attention_MultiRes(nn.Module):
    """
    Multi-Resolution Learnable Attention Module
    FIXES the upsampling bottleneck by using features from multiple scales
    """
    def __init__(self, n_channels=22, feature_channels=20, original_time_length=15000):
        super(EEG_Attention_MultiRes, self).__init__()
        
        self.n_channels = n_channels
        self.original_time_length = original_time_length
        
        # Calculate reduced time dimensions at each pooling stage
        self.time_3000 = 3000  # After first pooling
        self.time_600 = 600    # After second pooling
        self.time_120 = 120    # Final features
        
        # ========== TEMPORAL ATTENTION AT 3 SCALES ==========
        
        # 1. Attention at 3000 points (early features, high temporal resolution)
        self.temporal_3000 = nn.Sequential(
            nn.Conv1d(44, 32, kernel_size=3, padding=1),  # Input: 44 channels from concatenation
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output: [batch, 1, 3000]
        )
        
        # 2. Attention at 600 points (middle features)
        self.temporal_600 = nn.Sequential(
            nn.Conv1d(20, 32, kernel_size=3, padding=1),  # Input: 20 channels
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output: [batch, 1, 600]
        )
        
        # 3. Attention at 120 points (final features) - SAME AS ORIGINAL
        self.temporal_120 = nn.Sequential(
            nn.Conv1d(feature_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output: [batch, 1, 120]
        )
        
        # Spatial Attention: Which channels to focus on - SAME AS ORIGINAL
        self.spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global average pooling over time
            nn.Flatten(1),            # [batch, feature_channels]
            nn.Linear(feature_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_channels),
            nn.Sigmoid()  # Output: [batch, n_channels]
        )
        
        # Learnable weights for combining different scales
        self.scale_weights = nn.Parameter(torch.ones(3) / 3)  # [weight_3000, weight_600, weight_120]
        
    def forward(self, features_3000, features_600, features_120):
        """
        features_3000: [batch, 44, 3000] - Early features after first pooling
        features_600: [batch, 20, 600]   - Middle features after second pooling
        features_120: [batch, 20, 120]   - Final features from encoder
        Returns: attention_map [batch, n_channels, 15000]
        """
        # 1. Generate temporal attention at ALL 3 scales
        att_3000 = self.temporal_3000(features_3000)  # [batch, 1, 3000]
        att_600 = self.temporal_600(features_600)      # [batch, 1, 600]
        att_120 = self.temporal_120(features_120)      # [batch, 1, 120]
        
        # 2. Upsample all to original resolution (15000)
        # Note: Each has different upsampling factor:
        # - att_3000: 3000 → 15000 (5x upsampling)
        # - att_600: 600 → 15000 (25x upsampling)  
        # - att_120: 120 → 15000 (125x upsampling)
        
        att_3000_up = F.interpolate(att_3000, size=self.original_time_length, 
                                   mode='linear', align_corners=False)
        att_600_up = F.interpolate(att_600, size=self.original_time_length, 
                                  mode='linear', align_corners=False)
        att_120_up = F.interpolate(att_120, size=self.original_time_length, 
                                  mode='linear', align_corners=False)
        
        # 3. Weighted combination of different scales
        # Learn which scale is most important
        weights = F.softmax(self.scale_weights, dim=0)
        
        # Combine: early features give precise timing, final features give context
        temporal_fused = (
            weights[0] * att_3000_up +  # High temporal precision (only 5x upsampling)
            weights[1] * att_600_up +   # Medium precision (25x upsampling)
            weights[2] * att_120_up     # Global context (125x upsampling)
        )  # [batch, 1, 15000]
        
        # 4. Spatial attention - SAME AS ORIGINAL
        spatial_att = self.spatial_attention(features_120)  # [batch, 22]
        
        # 5. Combine temporal and spatial - SAME AS ORIGINAL
        # Expand dimensions for broadcasting
        temporal_expanded = temporal_fused.unsqueeze(1)  # [batch, 1, 1, 15000]
        spatial_expanded = spatial_att.unsqueeze(-1).unsqueeze(-1)  # [batch, 22, 1, 1]
        
        # Combined attention at full resolution
        combined = temporal_expanded * spatial_expanded  # [batch, 22, 1, 15000]
        attention_map = combined.squeeze(2)  # [batch, 22, 15000]
        
        # 6. Ensure values are in [0, 1] - SAME AS ORIGINAL
        attention_map = torch.clamp(attention_map, 0, 1)
        
        return attention_map


# ========== EXISTING ARCHITECTURE COMPONENTS (NO CHANGES) ==========

class GateDilateLayer(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation):
        super(GateDilateLayer, self).__init__()
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
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rates):
        super(GateDilate, self).__init__()
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
    def __init__(self, in_channels):
        super(ResConv, self).__init__()
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


# ========== UPDATED MAIN NEUROGATE MODEL ==========

class NeuroGATE_Gaze_MultiRes(nn.Module):
    def __init__(self, n_chan: int = 22, n_outputs: int = 2, original_time_length: int = 15000):
        super(NeuroGATE_Gaze_MultiRes, self).__init__()
        
        self.n_chan = n_chan
        self.n_outputs = n_outputs
        self.original_time_length = original_time_length
        
        # ========== EVERYTHING BELOW IS IDENTICAL TO ORIGINAL ==========
        
        # NeuroGATE architecture
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
        
        # ========== ONLY CHANGE: Use MultiRes Attention ==========
        self.attention_layer = EEG_Attention_MultiRes(
            n_channels=n_chan,
            feature_channels=20,
            original_time_length=original_time_length
        )
        
        # Final classification layer
        self.fc = nn.Linear(20, n_outputs)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        
    def forward(self, x, return_attention=False):
        """
        x: (batch_size, channels, time_steps) - Original EEG
        
        Returns:
        - If return_attention=False: logits only [batch, n_outputs]
        - If return_attention=True: dictionary with 'logits' and 'attention_map'
        """
        batch_size, channels, original_time = x.shape
        
        # ========== 1. FEATURE EXTRACTION ==========
        # EXACTLY SAME as original, but we SAVE features at each scale
        
        # First pooling: 15000 → 3000
        x1 = F.avg_pool1d(x, kernel_size=5, stride=5)
        x2 = F.max_pool1d(x, kernel_size=5, stride=5)
        x = torch.cat((x1, x2), dim=1)  # [batch, 44, 3000]
        x = self.bn1(x)
        
        # SAVE early features (3000 time points)
        features_3000 = x
        
        x1 = self.res_conv1(x)
        x2 = self.gate_dilate1(x)
        x = x1 + x2
        x = F.dropout1d(x, 0.5, training=self.training)
        x = F.max_pool1d(x, kernel_size=5, stride=5)  # [batch, *, 600]
        
        x = F.relu(self.bn2(self.conv1(x)))
        
        # SAVE middle features (600 time points)
        features_600 = x  # [batch, 20, 600]
        
        x1 = self.res_conv2(x)
        x2 = self.gate_dilate2(x)
        x = x1 + x2
        x = self.bn3(self.conv2(x))
        
        x = self.res_conv3(x)
        x = F.max_pool1d(x, kernel_size=5, stride=5)  # [batch, *, 120]
        
        x = self.bn4(self.conv3(x))  # [batch, 20, 120]
        
        # ========== 2. TRANSFORMER ENCODER ==========
        x = x.permute(0, 2, 1)  # (batch, time, features)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)  # Back to (batch, features, time)
        
        # These are our final encoded features
        features_120 = x  # [batch, 20, 120]
        
        # ========== 3. GENERATE ATTENTION MAP ==========
        attention_map = None
        if return_attention:
            # USE MULTI-RESOLUTION: Pass features from ALL 3 scales
            attention_map = self.attention_layer(
                features_3000, features_600, features_120
            )  # [batch, 22, 15000]
        
        # ========== 4. CLASSIFICATION ==========
        # SAME AS ORIGINAL
        if attention_map is not None:
            # Downsample attention to match feature resolution
            attention_down = F.avg_pool1d(attention_map, kernel_size=125, stride=125)
            # attention_down: [batch, 22, 120]
            
            # Apply attention (average over channels for features)
            attention_weights = attention_down.mean(dim=1, keepdim=True)  # [batch, 1, 120]
            attended_features = features_120 * attention_weights
            x_pooled = torch.mean(attended_features, dim=2)
        else:
            x_pooled = torch.mean(features_120, dim=2)
        
        logits = self.fc(x_pooled)
        
        # ========== 5. RETURN VALUES ==========
        if return_attention:
            return {
                'logits': logits,
                'attention_map': attention_map,
                'features': features_120
            }
        else:
            return logits
    
    def get_attention(self, x):
        """Convenience method to get attention map only"""
        with torch.no_grad():
            outputs = self.forward(x, return_attention=True)
            return outputs['attention_map']
    
    def get_features(self, x):
        """Get intermediate features"""
        with torch.no_grad():
            outputs = self.forward(x, return_attention=True)
            return outputs['features']


# ========== BACKWARD COMPATIBLE VERSION ==========

class NeuroGATE_Gaze_MultiRes_Simple(nn.Module):
    """
    Simple version that always returns tuple (logits, attention_map) 
    for backward compatibility with old code
    """
    def __init__(self, n_chan: int = 22, n_outputs: int = 2, original_time_length: int = 15000):
        super().__init__()
        self.model = NeuroGATE_Gaze_MultiRes(n_chan, n_outputs, original_time_length)
        
    def forward(self, x, return_attention=False):
        if return_attention:
            outputs = self.model(x, return_attention=True)
            return outputs['logits'], outputs['attention_map']
        else:
            return self.model(x, return_attention=False)



#  navaal's code NeuroGATE EEG Gaze Classification Model