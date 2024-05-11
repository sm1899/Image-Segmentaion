import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import mobilenet_v2



class CustomDecoder(nn.Module):
    def __init__(self, num_classes=1):
        super(CustomDecoder, self).__init__()
        self.num_classes = num_classes

        # Define layers for upsampling
        self.upconv1 = nn.ConvTranspose2d(1280, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(528, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv3 = nn.ConvTranspose2d(280, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv4 = nn.ConvTranspose2d(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Additional layers for combining features from multiple encoder layers
        self.combined_conv = nn.Conv2d(160, 64, kernel_size=1)

        # Final convolutional layer for segmentation
        self.final_conv = nn.Conv2d(64, num_classes , kernel_size=1)

    def forward(self, x, skip_connections):
        # Upsample and concatenate features from encoder layers
        x = self.upconv1(x)
        skip_connections[0] = F.interpolate(skip_connections[0], size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip_connections[0]], dim=1)

        x = self.upconv2(x)
        skip_connections[1] = F.interpolate(skip_connections[1], size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip_connections[1]], dim=1)

        x = self.upconv3(x)
        skip_connections[2] = F.interpolate(skip_connections[2], size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip_connections[2]], dim=1)

        x = self.upconv4(x)
        skip_connections[3] = F.interpolate(skip_connections[3], size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip_connections[3]], dim=1)


        x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)

        # Additional convolution for combining features
        x = self.combined_conv(x)

        # Final convolution for segmentation
        x = self.final_conv(x)

        return torch.sigmoid(x)


class MobileNetEncoder(nn.Module):
    def __init__(self, freeze_mobile_net=True):
        super(MobileNetEncoder, self).__init__()
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.features = mobilenet.features

        # Freeze MobileNet layers if specified
        if freeze_mobile_net:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Store features from different layers for skip connections
        skip_connections = []

        # Layer 1
        x = self.features[0:2](x)
        skip_connections.append(x)

        # Layer 2
        x = self.features[2:4](x)
        skip_connections.append(x)

        # Layer 3
        x = self.features[4:7](x)
        skip_connections.append(x)

        # Layer 4
        x = self.features[7:14](x)
        skip_connections.append(x)

        # Layer 5
        x = self.features[14:](x)

        return x, skip_connections


class MobileNetEncoderDecoder(nn.Module):
    def __init__(self, num_classes=1, freeze_mobile_net=True):
        super(MobileNetEncoderDecoder, self).__init__()
        self.encoder = MobileNetEncoder(freeze_mobile_net)
        self.decoder = CustomDecoder(num_classes)

    def forward(self, x):
        # Encoder
        x, skip_connections = self.encoder(x)

        # Decoder
        x = self.decoder(x, skip_connections)

        return x


model = MobileNetEncoderDecoder()
# input tensor
input_tensor = torch.randn(1, 3,128, 128)
output_tensor = model(input_tensor)
print(output_tensor.shape)  # Example output shape
# Output: torch.Size([1, 1, 128, 128])

