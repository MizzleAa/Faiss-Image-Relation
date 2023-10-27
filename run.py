import numpy as np
import faiss
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2
from PIL import Image
import matplotlib.pyplot as plt

def extract_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for inputs, labels_batch in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            embeddings.append(outputs.cpu().numpy())
            labels.append(labels_batch.numpy())

    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)

    return embeddings, labels

def load_faiss():
    dim = 1280
    fvec_file = 'fvecs.bin'
    index_type = 'hnsw'
    index_file = f'{fvec_file}.{index_type}.index'
    fvecs = np.memmap("./fvec/fvecs.bin", dtype='float32', mode='r').view('float32').reshape(-1, dim)

def build_faiss_index(embeddings, dimension=128):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)  # L2 거리 메트릭 사용
    index.add(embeddings)
    return index

def search_faiss_index(query_embedding, index, k=5):
    D, I = index.search(query_embedding, k)
    return D, I

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    return model

def show_image_with_label(image_path, label):
    image = Image.open(image_path)
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(f'Label: {label}')
    plt.axis('off')
    plt.show()

def show_images_with_labels(image_paths, labels):
    num_images = len(image_paths)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

    for i in range(num_images):
        image = Image.open(image_paths[i])
        axes[i].imshow(image)
        axes[i].set_title(f'Label: {labels[i]}')
        axes[i].axis('off')

    plt.show()

def show_input_and_related_images(input_image_path, related_image_paths, labels,):
    num_images = len(related_image_paths) + 1

    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

    input_image = Image.open(input_image_path)
    axes[0].imshow(input_image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    # 연관 이미지 표시
    for i in range(1, num_images):
        related_image = Image.open(related_image_paths[i - 1])
        axes[i].imshow(related_image)
        axes[i].set_title(f'Label: {labels[i - 1]}')
        axes[i].axis('off')

    plt.show()

def main():
    data_dir = './dataset'
    model_path = './model/mobilenetv2_custom.pth'

    input_shape = (224, 224, 3)
    batch_size = 20
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transforms = transforms.Compose([
        transforms.Resize(input_shape[:2]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(data_dir, transform=data_transforms)

    model = mobilenet_v2(pretrained=False)
    num_classes = len(train_dataset.classes)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = load_model(model, model_path)
    model.to(device)

    test_dataset = ImageFolder(data_dir, transform=data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_embeddings, test_labels = extract_embeddings(model, test_loader, device)
    # Faiss 인덱스 빌드
    faiss_index = build_faiss_index(test_embeddings)

    #####################################################
    # 1. 특정 이미지에 대한 검색 예제
    query_index = 201  # 검색할 이미지 인덱스
    query_embedding = test_embeddings[query_index].reshape(1, -1)
    D, I = search_faiss_index(query_embedding, faiss_index)

    # 가장 가까운 이미지와 거리 출력
    print("inner train")
    for i in range(len(I[0])):
        print(f"Distance: {D[0][i]}, Image Index: {I[0][i]}, Label: {test_labels[I[0][i]]}")

    #####################################################
    # 2. 이미지 불러와서 검색 예제
    image_path = './test/qrcode_30_270_295.png'

    # 이미지 불러오기 및 변환
    image = Image.open(image_path).convert('RGB')
    image_transform= transforms.Compose([
        transforms.Resize(input_shape[:2]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_image = image_transform(image).unsqueeze(0)  # 배치 차원을 추가합니다

    # 입력 이미지에 대한 임베딩 추출
    input_image = input_image.to(device)
    with torch.no_grad():
        input_embedding = model(input_image).cpu().numpy()

    # Faiss 인덱스를 사용하여 입력 이미지에 대한 검색
    D, I = search_faiss_index(input_embedding, faiss_index)
    # 가장 가까운 이미지와 거리 출력
    print("new image")
    # 이미지 파일 경로와 라벨을 저장하는 리스트
    input_image_path = image_path 
    related_image_paths = []
    image_labels = []

    # 검색 결과를 반복하면서 이미지 경로와 라벨을 저장
    for i in range(len(I[0])):
        image_path = test_dataset.imgs[I[0][i]][0]
        image_label = test_labels[I[0][i]]
        related_image_paths.append(image_path)
        image_labels.append(image_label)
        print(f"Distance: {D[0][i]}, Image Index: {I[0][i]}, Label: {test_labels[I[0][i]]}")

    # 입력 이미지와 연관 이미지를 함께 표시
    show_input_and_related_images(input_image_path, related_image_paths, image_labels)

if __name__ == '__main__':
    main()