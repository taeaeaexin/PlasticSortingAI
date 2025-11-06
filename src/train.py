import torch
import time
import sys
from torch import nn, optim # 신경망 래퍼, 옵티마이저
from torchvision import datasets, transforms, models # 이미지 데이터 셋, 전처리, 모델
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader # 배치 단위 데이터 로딩

# 빠른 입출력
input = sys.stdin.readline

# 1. 데이터 전처리

# train : 데이터 전처리 + 증강
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")),  # 투명/팔레트 이미지를 3채널 RGB로 통일 (PIL 경고 제거)
    transforms.RandomHorizontalFlip(),          # 좌우 반전
    transforms.RandomRotation(10),              # +-10도 회전
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 색감 변화
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# val : 검증 데이터
transform_val = transforms.Compose([
    transforms.Resize((224, 224)), # 입력 크기 변경 255 -> 224
    transforms.Lambda(lambda img: img.convert("RGB")),  # 투명/팔레트 이미지를 3채널 RGB로 통일 (PIL 경고 제거)
    transforms.ToTensor(), # to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 2. 데이터 셋 구성
train_dataset = datasets.ImageFolder(root="../dataset/train", transform=transform_train)
val_dataset = datasets.ImageFolder(root="../dataset/val", transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False)

# 3. ResNet18 기반 모델 구성
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, 2)
)

# 4. 하드웨어 Apple GPU: mps
device = torch.device("mps" if torch.mps.is_available() else "cpu") # cuda -> mps (Apple GPU)
model = model.to(device)

# 5. loss 계산 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005) # 0.0005 -> 0.0001

# 6. 변수 세팅 (에폭, 정확도, 시간)
epochs = 10
val_sum = 0
start_time = time.time()

# 7. 학습 loop
best_acc = 0.0
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

    val_acc = correct / len(val_dataset)
    val_sum += val_acc

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "../model/best_model.pth")
        print(f"최고 정확도 갱신: {best_acc:.2%}")

    print(f"[Epoch{epoch + 1}] "
          f"훈련(train) 오차: {train_loss/len(train_loader):.4f}, "
          f"검증(val) 오차: {val_loss/len(val_loader):.4f}, "
          f"정확도: {val_acc:.2%}")

# 8. 학습 결과 저장
print()
torch.save(model.state_dict(), "../model/plasticSortingAI.pth")

end_time = time.time()
print(f"평균 정확도: {val_sum / epochs:.2%} / 최고 정확도 : {best_acc:.2%} / 소요 시간: {end_time - start_time:.2f}초", flush=True)