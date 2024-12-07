import torch
import torch.nn as nn
import torch.onnx
import datetime
import torchinfo

class ConvBlock2D(nn.Module):
    def __init__(self, input_channels : int, output_channels : int, kernel_size = 3, padding = 1, stride = 1) -> None:
        super().__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, bias = False),
            nn.BatchNorm2d(output_channels, momentum = 0.9),
            nn.LeakyReLU(inplace = False)
        )
    
    def forward(self, x):
        return self.block_1(x)

class ConvPoolBlock(nn.Module):
    def __init__(self, input_channels : int, output_channels : int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            ConvBlock2D(input_channels,   output_channels, kernel_size = 3, padding = "same"),
            ConvBlock2D(output_channels,  output_channels, kernel_size = 1, padding = "same"),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        )
    
    def forward(self, x) :
        return self.block(x)

class ExtendedSimpleCNN2D(nn.Module):
    def __init__(self, input_channels : int, output_classes : int) -> None:
        super().__init__()

        self.mean = torch.tensor([0.4824, 0.4790, 0.4372])  
        self.std = torch.tensor([0.2118, 0.2099, 0.2135])   

        self.feet = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size = 5, stride = 2, padding = 1)
        )

        self.body = nn.Sequential(
            ConvPoolBlock(16,   32),
            ConvPoolBlock(32,   64),
            ConvPoolBlock(64,  128),
            ConvPoolBlock(128, 256),
            ConvPoolBlock(256, 256)
        )

        self.neck = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, output_classes),
            nn.Softmax(dim=1)
        )
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean[None, :, None, None]) / self.std[None, :, None, None]
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # normalisasi input data
        x = self.normalize(x)
        
        # proses model
        x = self.feet(x)
        x = self.body(x)
        x = self.neck(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

if __name__ == "__main__":
    print("Model Run")

    cux = torch.device('cpu')

    # Model definition
    mod = ExtendedSimpleCNN2D(3, 7).to(cux)

    with torch.no_grad():
        t = torch.rand(1, 3, 177, 177).to(cux)
        y = mod(t)

    torchinfo.summary(mod, input_data = t)

    start_t = datetime.datetime.now()
    for _ in range(10):
        with torch.no_grad():
            t = torch.rand(1, 3, 177, 177).to(cux)
            y = mod(t)
    stop_t   = datetime.datetime.now()
    exc_time = (stop_t- start_t).total_seconds()
    print("Total Time :", exc_time / 10)

    mod.eval()  

    # Dummy input for size [batch_size, channels, height, width] -> [1, 3, 177, 177]
    dummy_input = torch.rand(1, 3, 177, 177).to(cux)
    mean = torch.tensor([0.4824, 0.4790, 0.4372], device=cux).view(1, 3, 1, 1)
    std = torch.tensor([0.2118, 0.2099, 0.2135], device=cux).view(1, 3, 1, 1)
    dummy_input = (dummy_input - mean) / std

    # Perform inference to verify the model
    with torch.no_grad():
        output = mod(dummy_input)

    print("Model output:", output)
    print("Input size:", dummy_input.shape)

    assert output.shape == (1, 7), f"Unexpected output shape: {output.shape}"
    assert torch.is_tensor(output), "Output must be a tensor"
    assert torch.allclose(output.sum(dim=1), torch.tensor(1.0)), "Probabilities must sum to 1"
    print("Output is probabilities. Example values:", output[0])

    # Export model to ONNX
    torch.onnx.export(
        mod,                         # Model being exported
        dummy_input,                 # Input tensor
        "model.onnx",                # Output file name
        export_params=True,          # Store the trained parameter weights
        opset_version=19,            # ONNX opset version
        do_constant_folding=True,    # Perform constant folding optimization
        input_names=['input'],       # Input tensor names
        output_names=['output'],     # Output tensor names
        dynamic_axes={               # Dynamic axis for batch size
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print("Model successfully exported to ONNX as 'model.onnx'!")
