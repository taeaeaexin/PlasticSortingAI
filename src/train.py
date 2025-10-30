import torch
from torch import nn, optim # 신경망 래퍼, 옵티마이저
from torchvision import datasets, transforms, models # 이미지 머신 러닝
from torch.utils.data import DataLoader

# 전처리
transform_data = transforms.Compose([
    transforms.Resize((255, 255)),
    transforms.ToTensor(), # to Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(root="../dataset/train", transform=transform_data)
val_dataset = datasets.ImageFolder(root="../dataset/val", transform=transform_data)

train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False)

# ResNet18
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# loop
epochs = 10
val_sum = 0

for epoch in range(epochs):
    # train
    model.train()
    train_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # val
    model.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()

    val_sum += correct/len(val_dataset)

    print(f"[case {epoch + 1}] "
          f"훈련(train) 오차: {train_loss/len(train_loader):.4f}, "
          f"검증(val) 오차: {val_loss/len(val_loader):.4f}, "
          f"정확도: {correct/len(val_dataset):.2%}")

torch.save(model.state_dict(), "../model/plasticSortingAI.pth")

print(f"평균 정확도 : {val_sum / epochs:.2%}")