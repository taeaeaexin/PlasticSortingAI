# ⚙️ Plastic Sorting AI
> 플라스틱 이미지 분류 AI  
> 2025.10.28. ~ 

<br>

## 🛠️ 스택
- 언어 : Python 3.9
- 라이브러리 : PyTorch, TorchVision, PIL
- 모델 : ResNet18
- 데이터 처리 : ImageFolder, transforms
- 환경 : MacBook(Apple M4) / IntelliJ

<br>

## 🗂️ 데이터 셋
dataset/  
├─ train/  
│   ├─ PLASTIC/  
│   └─ NON_PLASTIC/  
└─ val/  
├─ PLASTIC/  
└─ NON_PLASTIC/  
- icrawler로 Google Image 수집 후 라벨링

<br>

## 📈 학습 결과
## 1. 프로토타입 모델
### 🔍 스펙
- 모델: ResNet18 (pretrained=True)
- 입력 크기: 255×255
- 정규화: mean/std = [0.5, 0.5, 0.5]
- 옵티마이저: Adam(lr=0.0005)
- 에폭: 10

### 📚 결과
![img.png](img/img.png)

<br>