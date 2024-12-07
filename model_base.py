import torch
import torch.nn    as nn
import torch.onnx
import datetime
import torchinfo

class BasicConvolution(nn.Module):
    def __init__(self, input_ch : int, output_ch : int) -> None:
        super().__init__()

        self.convolution   = nn.Conv2d(input_ch, output_ch, kernel_size = 3, padding = 'same')
        self.normalization = nn.BatchNorm2d(output_ch)
        self.activation    = nn.ReLU(inplace=False)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.convolution(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x

class BasicConvBlock(nn.Module):
    def __init__(self, input_ch : int, output_ch : int) -> None:
        super().__init__()
        self.conv = BasicConvolution(input_ch, output_ch)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, output_class: int) -> None:
        super().__init__()

        # self.mean = torch.tensor([0.4824, 0.4790, 0.4372]).view(1, 3, 1, 1)  # Mean untuk normalisasi
        # self.std = torch.tensor([0.2118, 0.2099, 0.2135]).view(1, 3, 1, 1)  # Std untuk normalisasi

        # Arsitektur jaringan
        self.conv_1 = nn.Conv2d(3, 16, kernel_size=7, padding="same")  # (3, 177, 177) -> (16, 177, 177)
        self.block_1 = BasicConvBlock(16, 32)  # (16, 177, 177) -> (32, 88, 88)
        self.block_2 = BasicConvBlock(32, 64)  # (32, 88, 88) -> (64, 44, 44)
        self.block_3 = BasicConvBlock(64, 128)  # (64, 44, 44) -> (128, 22, 22)
        self.block_4 = BasicConvBlock(128, 256)  # (128, 22, 22) -> (256, 11, 11)

        self.pool = nn.AdaptiveMaxPool2d(1)  # (256, 11, 11) -> (256, 1, 1)

        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, output_class)
        )

    # def normalize(self, x: torch.Tensor) -> torch.Tensor:
    #     """Normalisasi input dengan mean dan std."""
    #     if x.device != self.mean.device:
    #         self.mean = self.mean.to(x.device)
    #         self.std = self.std.to(x.device)
    #     return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalisasi input
        # x = self.normalize(x)

        # Forward pass
        x = self.conv_1(x)  # (N, 3, 177, 177) -> (N, 16, 177, 177)
        x = self.block_1(x)  # -> (N, 32, 88, 88)
        x = self.block_2(x)  # -> (N, 64, 44, 44)
        x = self.block_3(x)  # -> (N, 128, 22, 22)
        x = self.block_4(x)  # -> (N, 256, 11, 11)
        x = self.pool(x)     # -> (N, 256, 1, 1)

        # Flatten untuk fully connected layers
        x = x.view(x.size(0), -1)  # -> (N, 256)
        x = self.head(x)           # -> (N, output_class)

        # Softmax untuk probabilitas output
        x = torch.softmax(x, dim=1)
        return x

if __name__ == "__main__":
    print("Model Base Run")

    t     = torch.rand(1, 3, 177, 177)
    model = SimpleCNN(7)
    y     = model(t)
    print(y)

# if __name__ == "__main__":
#     print("Model Simple CNN")

#     cux = torch.device('cpu')

#     # Model definition
#     mod = SimpleCNN(7).to(cux)

#     with torch.no_grad():
#         t = torch.rand(1, 3, 177, 177).to(cux)
#         y = mod(t)

#     torchinfo.summary(mod, input_data = t)

#     start_t = datetime.datetime.now()
#     for _ in range(10):
#         with torch.no_grad():
#             t = torch.rand(1, 3, 177, 177).to(cux)
#             y = mod(t)
#     stop_t   = datetime.datetime.now()
#     exc_time = (stop_t- start_t).total_seconds()
#     print("Total Time :", exc_time / 10)

#     mod.eval()  

#     # Dummy input for size [batch_size, channels, height, width] -> [1, 3, 177, 177]
#     dummy_input = torch.rand(1, 3, 177, 177).to(cux)
#     # mean = torch.tensor([0.4824, 0.4790, 0.4372], device=cux).view(1, 3, 1, 1)
#     # std = torch.tensor([0.2118, 0.2099, 0.2135], device=cux).view(1, 3, 1, 1)
#     # dummy_input = (dummy_input - mean) / std

#     # Perform inference to verify the model
#     with torch.no_grad():
#         output = mod(dummy_input)

#     print("Model output:", output)
#     print("Input size:", dummy_input.shape)

#     assert output.shape == (1, 7), f"Unexpected output shape: {output.shape}"
#     assert torch.is_tensor(output), "Output must be a tensor"
#     assert torch.allclose(output.sum(dim=1), torch.tensor(1.0)), "Probabilities must sum to 1"
#     print("Output is probabilities. Example values:", output[0])

#     # Export model to ONNX
#     torch.onnx.export(
#         mod,                         # Model being exported
#         dummy_input,                 # Input tensor
#         "model_base.onnx",                # Output file name
#         export_params=True,          # Store the trained parameter weights
#         opset_version=19,            # ONNX opset version
#         do_constant_folding=True,    # Perform constant folding optimization
#         input_names=['input'],       # Input tensor names
#         output_names=['output'],     # Output tensor names
#         dynamic_axes={               # Dynamic axis for batch size
#             'input': {0: 'batch_size'},
#             'output': {0: 'batch_size'}
#         }
#     )

#     print("Model successfully exported to ONNX as 'model.onnx'!")