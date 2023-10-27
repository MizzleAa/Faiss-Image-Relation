import struct
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2
import numpy as np

def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

def main():
    input_shape = (224, 224, 3)

    data_transforms = transforms.Compose([
        transforms.Resize(input_shape[:2]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    data_dir = './dataset'
    train_dataset = ImageFolder(data_dir, transform=data_transforms)

    batch_size = 20
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = mobilenet_v2(pretrained=False)

    num_classes = len(train_dataset.classes)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    num_epochs = 20
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        accuracy = calculate_accuracy(model, train_loader, device)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}, Accuracy: {accuracy:.2f}%')

        # Update the learning rate
        scheduler.step()

        if accuracy > 0.95:
            break

    print('Finished Training')

    torch.save(model.state_dict(), './model/mobilenetv2_custom.pth')

    # Extract features from the model
    feature_vectors = []
    model.eval()
    with torch.no_grad():
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            features = model.features(inputs)  # Extract features
            feature_vectors.append(features)

    feature_vectors = torch.cat(feature_vectors, dim=0)

    # Save feature vectors to a file
    with open('./fvec/fvecs.bin', 'wb') as f:
        fvecs = feature_vectors.cpu().numpy()
        fmt = f'{np.prod(fvecs.shape)}f'
        f.write(struct.pack(fmt, *(fvecs.flatten())))

if __name__ == '__main__':
    main()