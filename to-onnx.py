import torch
from model_base import SimpleCNN

# Load the trained model
model = SimpleCNN(output_class=7)
checkpoint = torch.load('./runs/MammalsClassification-9/best_checkpoint.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Create a dummy input tensor with the same shape as your model's input
dummy_input = torch.randn(1, 3, 177, 177)

# Export the model to ONNX format
torch.onnx.export(
    model,
    dummy_input,
    'model_base.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
)

print("Model has been converted to ONNX format and saved as 'model.onnx'")