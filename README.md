# ⚙️ Plastic Sorting AI
> 플라스틱 이미지 분류 AI  
> 2025.10.28. ~ 

## 💡 개요
플라스틱과 비플라스틱(유리, 스테인리스, 종이) 이미지를 분류하는 딥러닝 프로젝트입니다.  
Google image로 크롤링한 데이터를 라벨링하여 PyTorch, ResNet18 기반으로 학습, 검증을 구현했습니다.

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

## 📈 성능
| 버전 | 변경 사항                        |  평균 정확도  |   학습 시간   |
|------|------------------------------|:--------:|:---------:|
| Prototype | 기본 모델 (ResNet18, Adam 0.0005) |  84.78%  |     -     |
| V1 | 전처리 강화 (224x224, 정규화)        |  87.83%  |     -     |
| V2 | 데이터 증강 추가                    |  91.09%  | 65.92 sec |
| V3 | GPU(MPS) + Dropout           |  90.87%  | 22.49 sec |

## 🔍 상세
### 프로토타입
- 모델: ResNet18 (pretrained=True)
- 입력 크기: 255×255
- 정규화: mean/std = [0.5, 0.5, 0.5]
- 옵티마이저: Adam(lr=0.0005)
- 에폭: 10
![img.png](img/img.png)

<br>

### V1 (전처리 보완)
- 입력 크기: 255×255 -> 244x244
- 정규화: mean/std = [0.5, 0.5, 0.5] -> [0.485, 0.456, 0.406]/[0.229, 0.224, 0.225]
- 옵티마이저: Adam(lr=0.0005) -> 0.0001
![img.png](img/img1.png)

<br>

### V2 (학습 데이터 증강)
- 데이터 증강(train 데이터 변형)
![img_5.png](img/img_5.png)

<br>

### V3 (CPU -> GPU)
- cpu -> mps
- nn.Dropout(0.3) 추가
- 옵티마이저: Adam(lr=0.00005)
![img_7.png](img/img_7.png)
