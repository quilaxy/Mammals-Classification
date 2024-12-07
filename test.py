import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

# Daftar label kelas (sesuaikan dengan urutan saat pelatihan)
class_labels = ["Artic Fox", "Camel", "Dolphin", "Koala", "Orangutan", "Snow Leopard", "Water Buffalo"]

# Load model ONNX
session = ort.InferenceSession("model_base.onnx")

# Fungsi untuk memuat dan memproses gambar
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((177, 177)),  # Resize ke ukuran input model
        transforms.ToTensor()          # Ubah gambar menjadi tensor
    ])
    img = Image.open(image_path).convert("RGB")  # Pastikan 3 channel (RGB)
    img_tensor = transform(img).unsqueeze(0)  # Tambahkan dimensi batch
    # Normalisasi ke rentang [0, 1] sudah dilakukan secara implisit oleh ToTensor
    return img_tensor.numpy()

# Fungsi untuk menjalankan inferensi
def predict(image_path):
    # Preprocess gambar
    input_tensor = preprocess_image(image_path)

    # Masukkan input ke model
    inputs = {session.get_inputs()[0].name: input_tensor}
    outputs = session.run(None, inputs)  # Lakukan inferensi

    # Ambil probabilitas output
    probabilities = np.array(outputs[0])

    # Tampilkan probabilitas tiap kelas
    print("Probabilities:")
    for i, prob in enumerate(probabilities[0]):
        print(f"{class_labels[i]}: {prob:.4f}")

    # Tentukan kelas dengan probabilitas tertinggi
    predicted_index = np.argmax(probabilities)
    predicted_class = class_labels[predicted_index]
    print(f"\nPredicted Class: {predicted_class}")

# Path gambar untuk testing
image_path = "image-test/water_buffalo.jpg"  # Ubah sesuai dengan lokasi gambar Anda
predict(image_path)
