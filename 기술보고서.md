# CCTV 기반 호스 누수 이상탐지 POC 기술보고서

## 1. 개요

### 1.1 과제 배경

Photo 장비 내 호스 및 연결부에서 누수가 발생할 경우 장비 이상, 주변 오염 및 추가적인 안전 문제로 이어질 가능성이 있다.

본 POC는 기존 CCTV 영상을 활용하여 별도의 센서를 추가하지 않고 다음과 같은 누수 이상을 자동으로 검출할 수 있는지 기술적 가능성을 검증하는 것을 목적으로 한다.

주요 검출 대상은 다음 세 가지이다.

| 구분 | 이상 형태 | 특징 |
|---|---|---|
| Droplet | 물방울 맺힘 | 누수 초기 단계, 크기가 작아 검출 난이도가 높음 |
| Leak | 지속적인 물줄기 | 연결부에서 물이 지속적으로 새는 형태 |
| Splash | 물 튐/분사 | 누수가 심해져 여러 방향으로 물줄기가 퍼지는 형태 |

본 과제의 가장 큰 제약조건은 **실제 누수 NG 데이터가 거의 존재하지 않는다는 점**이다.

따라서 일반적인 지도학습 기반 OK/NG Classification보다, 정상 데이터만을 이용하여 정상 분포를 학습하고 정상과 다른 특징을 이상으로 판단하는 **Unsupervised Anomaly Detection** 방식이 적합하다고 판단하였다.

---

# 2. POC 목표 및 검증 범위

본 POC의 목표는 양산 수준의 최종 검출률 확보가 아니라 다음 질문에 대한 기술적 검증이다.

> **“현재 CCTV 영상과 정상 데이터만으로 호스 누수 이상을 검출할 수 있는 구조를 만들 수 있는가?”**

이를 위해 다음 항목을 단계적으로 검증하였다.

| 검증 영역 | 검증 내용 |
|---|---|
| 검사 가능 상태 판정 | 현재 영상에서 호스가 정상적인 검사 위치에 존재하는지 |
| 위치 보정 | 호스 위치 및 각도 변화에 대응 가능한지 |
| 정상 모델 구축 | 실제 NG 없이 Normal 데이터만으로 모델 구축 가능한지 |
| 이상 검출 | Droplet / Leak / Splash에 anomaly score가 증가하는지 |
| 이상 위치 검출 | 모델이 실제 합성 이상 영역을 보고 있는지 |
| 영상 판정 | Frame 단위 판정을 시간축 Alarm으로 확장 가능한지 |

---

# 3. 초기 구상

초기에는 가장 단순한 구조를 고려하였다.

```text
┌──────────────┐
│ CCTV Frame   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Fixed ROI    │
│ Crop         │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Anomaly      │
│ Detection    │
└──────┬───────┘
       │
       ▼
   OK / NG
```

즉, CCTV 화면에서 호스 영역을 고정 ROI로 지정하고 해당 이미지를 이상탐지 모델에 입력하는 방식이었다.

그러나 실제 영상을 분석하면서 해당 구조에 몇 가지 문제가 있음을 확인하였다.

### 문제 1. 검사 대상 위치 변화

장비 움직임, 진동 및 시야 변화로 인해 호스의 위치와 각도가 일정하지 않았다.

```text
Frame A                  Frame B

┌────────────┐           ┌────────────┐
│   HOSE     │           │            │
│    │       │           │      HOSE  │
│    │       │           │       │    │
└────────────┘           └────────────┘

      동일 Fixed ROI 사용 시
               ↓

실제 대상 위치 변화 자체가 Anomaly로 인식될 가능성
```

---

### 문제 2. 검사 불가능 상태 존재

모든 프레임에서 호스가 정상적으로 위치하는 것은 아니었다.

예를 들어 다음 상태가 존재할 수 있다.

```text
Normal Position
      │
      ├── 이동
      ├── 부분 가림
      ├── 회전
      └── 시야 이탈
```

이런 프레임을 PatchCore에 그대로 입력하면 실제 누수가 아니라 **위치 변화 자체가 높은 anomaly score를 발생**시킬 가능성이 있다.

---

### 문제 3. 실제 NG 부족

지도학습 기반 모델을 적용하려면 충분한 수의 누수 NG가 필요하지만 실제 데이터 확보가 어려웠다.

따라서 최종 구조는 단순 Fixed ROI 방식에서 다음 구조로 변경하였다.

---

# 4. 최종 시스템 설계

## 4.1 전체 시스템 Architecture

```text
                         CCTV VIDEO
                              │
                              ▼
                    ┌─────────────────┐
                    │  Search ROI     │
                    │ LEFT / RIGHT    │
                    └────────┬────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │ Feature-based Matching│
                 │ SIFT/ORB + FLANN/BF   │
                 └───────────┬───────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Homography    │
                    │     RANSAC      │
                    └────────┬────────┘
                             │
                             ▼
                      ┌─────────────┐
                      │ Gate Logic  │
                      └──────┬──────┘
                             │
                   ┌─────────┴─────────┐
                   │                   │
                FAIL                  PASS
                   │                   │
                   ▼                   ▼
              검사 Skip        ┌──────────────┐
                               │ Alignment    │
                               │ H inverse    │
                               └──────┬───────┘
                                      │
                                      ▼
                               ┌──────────────┐
                               │ Anomaly ROI  │
                               │ Canonical    │
                               └──────┬───────┘
                                      │
                                      ▼
                               ┌──────────────┐
                               │ PatchCore    │
                               └──────┬───────┘
                                      │
                           ┌──────────┴──────────┐
                           │                     │
                     Image Score           Anomaly Map
                           │
                           ▼
                     Threshold
                           │
                           ▼
                      Raw OK / NG
                           │
                           ▼
                     Temporal Filter
                           │
                           ▼
                       FINAL ALARM
```

최종 구조의 핵심은 이상탐지 모델 이전에 **Gate + Alignment 단계가 추가된 것**이다.

---

# 5. 개발 및 검증 프로세스

본 POC는 총 02~14 단계로 구분하여 개발하였다.

| 단계 | Script | 목적 |
|---:|---|---|
| 02 | `02_build_gate_dataset.py` | 영상에서 Gate 검증 데이터 생성 |
| 03 | `03_label_gate_dataset.py` | 검사 가능/불가능 상태 수동 라벨링 |
| 04 | `04_calibrate_gate.py` | Gate Threshold Calibration |
| 05 | `05_evaluate_gate_val.py` | Gate Validation |
| 06 | `06_extract_normal_roi.py` | Alignment 후 Normal ROI 추출 |
| 07 | `07_select_anomaly_roi.py` | 실제 검사 영역 선정 |
| 08 | `08_train_patchcore.py` | LEFT/RIGHT PatchCore 학습 |
| 09 | `09_generate_synthetic_ng.py` | Procedural Synthetic NG 생성 |
| 10 | `10_evaluate_patchcore.py` | NG Detection 성능 평가 |
| 11 | `11_validate_anomaly_maps.py` | Anomaly Localization 평가 |
| 12 | `12_generate_asset_leak_ng.py` | Realistic Water Asset 합성 |
| 13 | `13_generate_anomaly_test_video.py` | Synthetic 이상 영상 생성 |
| 14 | `14_run_video_inference.py` | End-to-End Video Inference |

전체 개발 흐름은 다음과 같다.

```text
[Gate 구축]
02 Dataset 생성
      ↓
03 Manual Label
      ↓
04 Threshold Calibration
      ↓
05 Validation
      │
      ▼
[Normal Dataset 구축]
06 Alignment / ROI 추출
      ↓
07 Anomaly ROI 선정
      │
      ▼
[Anomaly Model]
08 PatchCore Training
      │
      ▼
[Synthetic Validation]
09 Synthetic 생성
      ↓
10 Detection 검증
      ↓
11 Localization 검증
      ↓
12 Realistic Asset Synthetic
      │
      ▼
[Video Validation]
13 Anomaly Test Video
      ↓
14 End-to-End Inference
```

---

# 6. 검사 가능 상태 판정: Gate

## 6.1 Feature-based Matching

Gate의 첫 단계는 기준 Template과 현재 CCTV Frame의 Search ROI 사이에서 동일한 객체를 찾는 것이다.

```text
Template Image                   Current Frame

      │                               │
      ▼                               ▼
Feature Detection               Feature Detection
      │                               │
      ▼                               ▼
Descriptor                      Descriptor
      │                               │
      └───────────┬───────────────────┘
                  │
                  ▼
          Descriptor Matching
                  │
                  ▼
            Lowe Ratio Test
                  │
                  ▼
             Good Matches
```

Feature Detector는 SIFT 또는 ORB를 사용할 수 있도록 구성하였다.

Matcher는 다음 두 방식을 지원한다.

```text
FLANN Matcher
BF Matcher
```

---

## 6.2 Lowe Ratio Test

각 특징점에 대해 가장 가까운 Descriptor 하나만 선택하지 않고 1순위와 2순위의 거리 비율을 비교하였다.

```text
                 Best Match Distance
Ratio = --------------------------------
              Second Match Distance
```

다음 조건을 만족하는 경우에만 Good Match로 인정하였다.

```text
Ratio < Ratio Threshold
```

이를 통해 모호한 Feature Matching을 제거하였다.

---

# 7. Homography

Good Match를 이용하여 Template 좌표와 현재 CCTV 좌표 사이의 Homography를 계산하였다.

수학적으로 다음 관계를 이용한다.

```text
          ┌             ┐
          │ h11 h12 h13 │
p'  ~  H  │ h21 h22 h23 │ p
          │ h31 h32 h33 │
          └             ┘
```

여기서:

```text
p  : Template 좌표
p' : CCTV Search ROI 좌표
H  : 3 × 3 Homography Matrix
```

이다.

Good Match에는 잘못 매칭된 Outlier가 포함될 수 있으므로 Homography 계산에는 RANSAC을 적용하였다.

```text
Good Matches
     │
     ▼
   RANSAC
     │
 ┌───┴────┐
 │        │
Inlier  Outlier
 │
 ▼
Homography
```

---

# 8. Gate 판정 로직

Homography 계산 성공만으로 검사 가능 상태를 판단하지 않고 다음 정보를 추가 평가하였다.

| Metric | 의미 |
|---|---|
| Good Matches | Ratio Test를 통과한 특징점 수 |
| Inliers | RANSAC에서 정상 대응점으로 인정된 Match |
| Inlier Ratio | Good Match 중 Inlier 비율 |
| Area Ratio | Template 대비 투영 Polygon 면적 |
| Rotation | Template 기준 회전 각도 |
| Status | Homography 계산 성공 여부 |

Gate 판정식은 개념적으로 다음과 같다.

```text
Gate PASS =
    Homography Success
    AND Inlier 조건 만족
    AND Area Ratio 범위 만족
    AND Rotation 범위 만족
```

---

## 8.1 LEFT Gate

최종 선정 조건:

```text
min_inliers      >= 6

0.8 <= area_ratio <= 1.2

abs(rotation)    <= 5°
```

Validation 결과:

| 실제 상태 / 예측 | Gate PASS | Gate FAIL |
|---|---:|---:|
| Visible | 197 | 3 |
| Not Visible | 0 | 160 |

즉:

```text
TP = 197
TN = 160
FP =   0
FN =   3
```

이었다.

FN 3건은 모두 localization 품질 자체가 불안정한 영상이었다.

---

## 8.2 RIGHT Gate

최종 선정 조건:

```text
min_inliers      >= 20

0.8 <= area_ratio <= 1.2

abs(rotation)    <= 10°
```

Validation 결과:

```text
TP = 186
TN = 172
FP =   0
FN =   2
```

RIGHT 역시 Not Visible 영상을 Gate PASS시키는 False Positive는 발생하지 않았다.

---

# 9. Homography Alignment

PatchCore에는 가능한 동일한 위치와 동일한 형태의 영상이 입력되어야 한다.

Matching으로 구한 Homography는 다음 방향의 변환이다.

```text
Template
   │
   │ H
   ▼
Current CCTV
```

반대로 현재 CCTV 영상을 Template 기준 좌표계로 복원하기 위해서는 역행렬을 이용한다.

```text
Current CCTV
     │
     │ H⁻¹
     ▼
Canonical Template
```

따라서 다음과 같은 위치 변화가 발생하더라도:

```text
Frame #1         Frame #2         Frame #3

  HOSE             HOSE             HOSE
   │                /                \
   │               /                  \

     │               │                │
     └───────────────┼────────────────┘
                     ▼
                 Alignment
                     │
                     ▼

             동일 Canonical ROI
```

PatchCore에는 최대한 동일한 구조의 ROI가 입력된다.

---

# 10. PatchCore 적용

## 10.1 PatchCore 선정 이유

본 과제에서는 실제 누수 NG 데이터 확보가 어렵다.

따라서 다음 형태의 접근이 필요하였다.

```text
많은 Normal Data
       +
거의 없는 NG Data
       ↓
Normal-only Anomaly Detection
```

PatchCore는 정상 이미지에서 Patch Feature를 추출하여 정상 Feature Memory를 구축하고, 입력 Patch와 가장 가까운 정상 Feature 사이의 거리를 이용하여 이상을 판정한다.

---

# 11. PatchCore 학습 구조

```text
Normal ROI Images
        │
        ▼
┌───────────────────┐
│ WideResNet50-2    │
│ ImageNet Pretrain │
└─────────┬─────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
 layer2       layer3
    │           │
    │       Interpolate
    │           │
    └─────┬─────┘
          ▼
   Feature Concatenate
          │
          ▼
     Patch Embedding
          │
          ▼
 Candidate Sampling
          │
          ▼
Approx. Greedy Coreset
          │
          ▼
      Memory Bank
```

Backbone:

```text
WideResNet50-2
```

Feature Layer:

```text
layer2
layer3
```

최종 Patch Embedding Dimension:

```text
1536
```

---

# 12. Memory Bank 구성

전체 Patch를 모두 Memory Bank에 저장할 경우 추론 비용이 크게 증가할 수 있다.

따라서 다음 방식으로 Patch 수를 줄였다.

```text
전체 Normal Patch
       │
       ▼
Priority Reservoir
       │
       ▼
6,000 Candidate Patch
       │
       ▼
Approximate Greedy Coreset
       │
       ▼
600 Representative Patch
       │
       ▼
Memory Bank
```

학습 결과는 다음과 같다.

| 항목 | LEFT | RIGHT |
|---|---:|---:|
| Normal Images | 580 | 549 |
| Total Patch Seen | 593,920 | 562,176 |
| Candidate Patch | 6,000 | 6,000 |
| Memory Patch | 600 | 600 |
| Embedding Dimension | 1,536 | 1,536 |

CPU POC 환경에서도 ROI별 학습시간이 약 1분 수준으로 확인되어 반복 실험이 가능한 수준이었다.

---

# 13. PatchCore 이상 판정

입력 영상의 각 Patch Feature를 다음과 같이 비교한다.

```text
                Memory Bank

                 M1
                 M2
Test Patch ────► M3
                 ...
                 Mn

        가장 가까운 정상 Patch 검색
```

Patch `p`의 anomaly distance는 다음과 같이 표현할 수 있다.

```text
d(p) = min || p - m ||
       m∈M
```

여기서 `M`은 Normal Memory Bank이다.

Image Score는 각 Patch의 anomaly distance 중 높은 값을 이용한다.

```text
S(image) = max d(p)
            p
```

따라서:

```text
정상과 유사한 영상
    ↓
Patch Distance 작음
    ↓
Score 낮음


정상에서 보지 못한 구조
    ↓
Patch Distance 큼
    ↓
Score 높음
```

형태로 동작한다.

---

# 14. Anomaly Map

Patch 단위 distance를 원래 Feature Grid 형태로 재배치하면 위치별 anomaly score를 얻을 수 있다.

```text
Patch Distance
      │
      ▼
┌────────────────┐
│ Feature Grid   │
│ 32 × 32        │
└───────┬────────┘
        │
        ▼
   Interpolation
        │
        ▼
┌────────────────┐
│ Anomaly Map    │
└────────────────┘
```

이를 원본 ROI 위에 Heatmap 형태로 표시하여 모델이 실제 어느 위치를 이상으로 보고 있는지 확인하였다.

---

# 15. Synthetic NG 전략

실제 NG 데이터가 없다는 문제를 해결하기 위해 Synthetic NG를 생성하였다.

그러나 Synthetic NG는 PatchCore Training에는 사용하지 않았다.

```text
                    DATA
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
   REAL NORMAL              SYNTHETIC NG
          │                     │
          ▼                     ▼
 PatchCore Training        Validation Only
                                │
                     ┌──────────┼──────────┐
                     ▼          ▼          ▼
                  Detection   Threshold Localization
```

이는 모델이 Synthetic 생성 패턴 자체를 학습하는 것을 방지하기 위한 설계이다.

---

# 16. Synthetic NG 단계별 고도화

## 16.1 Easy

초기에는 이상탐지 가능성 자체를 확인하기 위해 비교적 눈에 띄는 변화를 생성하였다.

```text
droplet
wet_spot
thin_stream
```

목적:

```text
“물과 관련된 구조적 변화가 PatchCore Score에 반영되는가?”
```

를 확인하는 것이었다.

---

## 16.2 Medium

Easy보다 색상과 크기 차이를 줄여 정상 영상과의 차이를 낮추었다.

```text
Easy
 │
 │ Contrast 감소
 │ Size 감소
 │ Color Difference 감소
 ▼
Medium
```

특히 `wet_spot`은 육안으로도 정상 영상과 구분하기 어려운 수준까지 차이가 감소하였다.

이를 통해 단순 색 변화보다 물방울/물줄기처럼 **구조적인 영상 변화가 본 과제에서 더 적절한 이상 정의**라고 판단하였다.

---

# 17. Realistic Asset Synthetic

최종적으로 배경이 제거된 RGBA 형태의 실제 물 이미지를 이용하였다.

```text
droplet_xx.png
leak_xx.png
splash_xx.png
```

Asset은 하나를 그대로 합성하지 않고 다음 augmentation을 수행하였다.

```text
               Water PNG
                   │
                   ▼
          Transparent Crop
                   │
                   ▼
               Resize
                   │
             ┌─────┴─────┐
             ▼           ▼
          Rotation      Flip
             │           │
             └─────┬─────┘
                   ▼
               Opacity
                   │
                   ▼
                 Blur
                   │
                   ▼
              Placement
                   │
                   ▼
             Alpha Blend
                   │
                   ▼
          Realistic Synthetic
```

이를 통해 소수의 Asset으로도 다양한 형태의 Synthetic NG를 생성할 수 있도록 하였다.

---

# 18. Validation Dataset

Real Normal Validation 데이터는 다음과 같다.

| ROI | Normal Validation |
|---|---:|
| LEFT | 197 |
| RIGHT | 186 |

PatchCore 정상 Score 분포는 다음과 같았다.

### LEFT

| Metric | Score |
|---|---:|
| Mean | 15.21 |
| P50 | 15.10 |
| P95 | 16.07 |
| P99 | 18.32 |
| Max | 19.50 |

### RIGHT

| Metric | Score |
|---|---:|
| Mean | 13.71 |
| P50 | 13.63 |
| P95 | 14.47 |
| P99 | 15.39 |
| Max | 18.95 |

Train과 Validation의 score 분포가 유사하여 Normal 데이터 분포가 크게 달라지지 않음을 확인하였다.

---

# 19. Realistic Synthetic NG 결과

## 19.1 LEFT

| Class | Min | Median | Mean |
|---|---:|---:|---:|
| Normal | 14.31 | 15.10 | 15.21 |
| Droplet | 18.90 | 27.97 | 27.63 |
| Leak | 23.62 | 26.28 | 26.42 |
| Splash | 26.75 | 29.53 | 29.70 |

분포를 개념적으로 표현하면 다음과 같다.

```text
Score →

Normal
█████████████████████
14              18   19.50

Droplet
                    █████████████████████████
                   18.90               33.43

Leak
                          ███████████████
                         23.62        29.67

Splash
                              █████████████
                             26.75       31.89
```

Leak 및 Splash는 Normal 영역과 명확하게 분리되었다.

Droplet은 가장 낮은 Score 일부가 Normal Maximum과 중첩되었다.

Normal Max를 Threshold로 사용하면:

```text
Threshold ≈ 19.50

Normal FP = 0

Synthetic
89 / 90 Detect

Recall = 98.9 %
```

이었다.

---

# 20. RIGHT Realistic Synthetic 결과

| Class | Min | Median | Mean |
|---|---:|---:|---:|
| Normal | 12.86 | 13.63 | 13.71 |
| Droplet | 15.50 | 23.67 | 23.17 |
| Leak | 20.55 | 23.33 | 23.71 |
| Splash | 21.05 | 28.06 | 27.37 |

Score 분포:

```text
Score →

Normal
██████████████████████
12              15       18.95

Droplet
          ███████████████████████████
         15.50                 27.68

Leak
                        ███████████████
                       20.55        28.71

Splash
                         ██████████████████████
                        21.05              32.40
```

RIGHT에서도 Leak 및 Splash는 Normal과 명확하게 분리되었다.

반면 Droplet은 Normal과 일부 중첩되는 영역이 존재하였다.

Normal Max 기준:

```text
Threshold ≈ 18.95

Normal FP = 0

Synthetic
85 / 90 Detect

Recall = 94.4 %
```

이었다.

Normal P99를 Threshold로 낮추면:

```text
Threshold ≈ 15.39

Synthetic Recall = 100 %

Normal FP = 2 / 186
```

으로 변화하였다.

즉 다음 Trade-off가 존재한다.

```text
Threshold ↑
    │
    ├─ False Positive ↓
    └─ Tiny Droplet Miss ↑


Threshold ↓
    │
    ├─ Tiny Droplet Recall ↑
    └─ False Positive ↑
```

---

# 21. Anomaly Localization 검증

단순히 Image Score만 높아지는 것이 아니라 실제 이상 영역을 모델이 보고 있는지 확인하였다.

Synthetic 생성 시 실제 이상 영역 Mask를 함께 저장하였다.

```text
Synthetic Image
      │
      ├─────────────► PatchCore
      │                  │
      │                  ▼
      │              Anomaly Map
      │
      ▼
Ground Truth Mask
      │
      └──────────── Comparison ────────┐
                                       ▼
                                 Localization Metric
```

Easy Synthetic 기준 Anomaly Map Peak가 실제 Synthetic Mask 내부에 위치한 비율은:

| ROI | Peak Hit Rate |
|---|---:|
| LEFT | 91.1% |
| RIGHT | 88.9% |

이었다.

Thin Stream처럼 폭이 매우 작은 이상은 Feature Grid 및 CNN Receptive Field 영향으로 Peak Localization이 상대적으로 낮았다.

그러나 Top 5~10% 이상 영역까지 확대하면 대부분 실제 이상 영역을 포함하였다.

---

# 22. Frame 단위 판정

각 ROI의 PatchCore Score에 Threshold를 적용하여 Raw 판정을 생성한다.

```text
PatchCore Score
       │
       ▼
Score >= Threshold ?
       │
   ┌───┴───┐
   │       │
  YES      NO
   │       │
   ▼       ▼
  NG       OK
```

단, 하나의 Frame NG를 즉시 Alarm으로 사용하지 않았다.

---

# 23. Temporal Filter

CCTV 기반 시스템에서는 순간적인 영상 변화가 존재할 수 있으므로 최근 판정 결과를 이용한 Temporal Voting 구조를 적용하였다.

POC 기준:

```text
Window Size = 5
Required NG = 3
```

예를 들어:

```text
Frame       t-4   t-3   t-2   t-1   t
             │     │     │     │    │
Result       OK    NG    NG    NG   OK

NG Count = 3 / 5
                 │
                 ▼
              ALARM
```

수식으로 표현하면:

```text
               N
Alarm = 1  if  Σ NGi >= M
              i=1
```

POC에서는:

```text
N = 5
M = 3
```

이다.

이를 통해 단일 Frame의 순간적인 이상 Score가 즉시 Alarm으로 이어지는 것을 방지하였다.

---

# 24. Gate와 Temporal Filter의 관계

Gate FAIL은 `OK` 판정으로 처리하지 않는다.

```text
Frame
  │
  ▼
Gate
  │
  ├─ FAIL ─────► 검사하지 않음
  │
  └─ PASS
       │
       ▼
    PatchCore
       │
       ▼
  Temporal Vote 반영
```

Gate FAIL이 연속으로 발생하면 기존 Temporal History를 초기화하도록 구성하였다.

이는 검사 대상이 사라졌다가 다시 나타났을 때 이전 NG 상태가 새 검사 대상에 영향을 주는 것을 방지한다.

---

# 25. Synthetic Anomaly Test Video

이미지 단위 검증 이후 전체 영상 Pipeline 검증을 위해 Synthetic Anomaly Video 생성 구조를 구현하였다.

검증 시나리오는 누수가 점점 심해지는 상황을 재현하도록 설계하였다.

```text
Time
 │
 ├── 0 ~ 10 sec
 │       NORMAL
 │
 ├── 10 sec
 │       │
 │       ▼
 │    DROPLET
 │       │
 │       │ gradually increase
 │       ▼
 │     LEAK
 │       │
 │       │ increase
 │       ▼
 │    SPLASH
 │
 └── 35 sec
```

즉:

```text
NORMAL
   ↓
DROPLET
   ↓
LEAK
   ↓
SPLASH
```

순으로 누수 Severity가 증가한다.

---

# 26. Synthetic Asset의 영상 좌표 적용

단순히 원본 CCTV Pixel 위치에 물 이미지를 고정시키면 호스가 움직일 경우 누수가 공중에 떠 있는 형태가 될 수 있다.

따라서 Synthetic Asset을 Canonical Template 좌표에 먼저 배치하고 현재 Frame의 Homography를 이용해 CCTV 좌표로 투영한다.

```text
Canonical Template

        Water Asset
             │
             ▼
      [ Hose + Water ]
             │
             │ H
             ▼
Current CCTV Frame

      움직인 Hose 위치에
      Water도 함께 이동
```

이를 통해 실제 객체의 움직임을 따라가는 Synthetic Video를 생성할 수 있도록 설계하였다.

---

# 27. End-to-End Video Inference

최종 14단계 Pipeline은 다음 순서로 처리한다.

```text
Video Input
     │
     ▼
Target FPS Sampling
     │
     ▼
LEFT / RIGHT Search ROI
     │
     ▼
Feature Matching
     │
     ▼
Homography
     │
     ▼
Gate
     │
 ┌───┴────┐
 │        │
FAIL     PASS
 │        │
Skip      ▼
       Alignment
          │
          ▼
      Anomaly ROI
          │
          ▼
       PatchCore
          │
     ┌────┴────┐
     ▼         ▼
   Score    Heatmap
     │
     ▼
 Threshold
     │
     ▼
 Raw NG
     │
     ▼
 Temporal Filter
     │
     ▼
 Final Alarm
```

POC에서는 CPU 환경을 고려하여 약 2 FPS inference를 기준으로 설계하였다.

---

# 28. 결과 기록 및 Visualization

영상 추론 결과는 다음 형태로 저장한다.

```text
result.csv
summary.json
event_results.csv
annotated.mp4
```

### `result.csv`

Frame 또는 추론 시점별 상세 결과를 저장한다.

예:

```text
timestamp
frame_idx

left_gate_pass
left_score
left_raw_ng
left_alarm

right_gate_pass
right_score
right_raw_ng
right_alarm

gate_ms
patchcore_ms
total_inference_ms
```

---

# 29. Annotated Video

`annotated.mp4`는 단순 판정 결과뿐만 아니라 모델의 동작 과정을 확인할 수 있도록 구성하였다.

```text
┌──────────────────────────────────────────┬────────────────────┐
│                                          │ SYSTEM STATUS      │
│              CCTV FRAME                  │                    │
│                                          │ LEFT               │
│      ┌───────────┐   ┌───────────┐       │ Gate     PASS      │
│      │ LEFT ROI  │   │ RIGHT ROI │       │ Score    15.2      │
│      └───────────┘   └───────────┘       │ Threshold 19.5     │
│                                          │ Raw      OK        │
│ Homography Polygon                       │ Vote     0 / 5     │
│                                          │ Alarm    OFF       │
│                                          │                    │
│                                          │ RIGHT              │
│                                          │ Gate     PASS      │
│                                          │ Score    24.3      │
│                                          │ Threshold 18.95    │
│                                          │ Raw      NG        │
│                                          │ Vote     3 / 5     │
│                                          │ Alarm    ON        │
│                                          │                    │
│                                          │ ROI | Heatmap      │
└──────────────────────────────────────────┴────────────────────┘
```

Synthetic Test Video에서는 실제 Event 정보를 알고 있으므로 다음과 같이 Ground Truth도 표시한다.

```text
GT   : RIGHT LEAK
PRED : ALARM
```

이를 통해 이상 발생 시점과 Alarm 발생 시점 사이의 Detection Delay도 평가할 수 있다.

---

# 30. POC 정량 결과 요약

## Gate

| ROI | TP | TN | FP | FN |
|---|---:|---:|---:|---:|
| LEFT | 197 | 160 | 0 | 3 |
| RIGHT | 186 | 172 | 0 | 2 |

Not Visible 상태가 Gate PASS되는 사례는 Validation Dataset에서 발생하지 않았다.

---

## Realistic Synthetic Detection

Normal Maximum을 Threshold로 적용한 결과:

| ROI | Normal FP | Synthetic NG | Detect | Recall |
|---|---:|---:|---:|---:|
| LEFT | 0 | 90 | 89 | 98.9% |
| RIGHT | 0 | 90 | 85 | 94.4% |

주요 이상별 결과는 다음과 같이 정리할 수 있다.

| Defect | 결과 |
|---|---|
| Leak | 정상 영역과 명확한 Score Separation |
| Splash | 정상 영역과 명확한 Score Separation |
| Droplet | 대부분 검출 가능 |
| Tiny Droplet | 일부 Normal Score와 Overlap |

---

## Localization

| ROI | Peak Hit Rate |
|---|---:|
| LEFT | 91.1% |
| RIGHT | 88.9% |

PatchCore가 단순히 Image Score만 상승시키는 것이 아니라 실제 이상 위치의 Feature 변화에 반응하는 것을 확인하였다.

---

# 31. POC를 통해 확인한 기술적 가능성

### 31.1 검사 시점 선별 가능

Feature Matching과 Homography 기반 Gate를 통해 검사 가능한 프레임과 검사하기 어려운 프레임을 분리할 수 있었다.

### 31.2 위치 변화 대응 가능

Homography Alignment를 통해 위치 및 각도 변화의 영향을 감소시킬 수 있었다.

### 31.3 NG 없이 모델 구축 가능

정상 데이터만으로 PatchCore Memory Bank를 생성하고 이상 Score를 계산할 수 있었다.

### 31.4 주요 누수 형태 검출 가능

Leak과 Splash는 정상 Score 분포에서 명확하게 분리되었다.

### 31.5 이상 위치 설명 가능

Anomaly Map을 통해 어느 영역이 이상으로 판단되었는지 시각적으로 확인할 수 있었다.

### 31.6 영상 Alarm 구조로 확장 가능

단일 Frame 판정을 Temporal Voting 구조와 결합하여 실제 CCTV Monitoring 형태로 확장 가능한 Pipeline을 구축하였다.

---

# 32. 기술적 한계

현재 POC 결과에는 다음과 같은 한계가 존재한다.

### 실제 NG 부족

현재 Detection 성능은 실제 누수 NG가 아닌 Synthetic NG를 중심으로 평가하였다.

따라서:

```text
Synthetic Performance
        ≠
Real-world Performance
```

일 수 있다.

---

### 작은 Droplet

작은 초기 물방울은 PatchCore Feature Resolution에 비해 크기가 작기 때문에 정상 Score와 일부 Overlap이 발생하였다.

향후 검토 항목:

```text
Higher Resolution Input
       +
Patch Feature Layer 조정
       +
ROI 확대
       +
Normal Dataset 확대
```

등이 있다.

---

### 장기간 환경 변화 미검증

현재 POC에서는 다음 변화에 대한 충분한 장기 검증이 이루어지지 않았다.

```text
조명 변화
렌즈 오염
카메라 흔들림
장비 상태 변화
호스 교체
테이프 형상 변화
노후화
```

---

### Threshold 미확정

현재 Threshold는 Validation Normal과 Synthetic NG 기반으로 선정한 후보이다.

실제 운영 Threshold는 다음 두 항목을 동시에 고려해야 한다.

```text
NG Detection Recall
        ↑

False Alarm / Hour
        ↓
```

---

# 33. 향후 고도화 방향

향후 개발은 다음 구조로 진행하는 것이 적절하다.

```text
현재 POC
   │
   ▼
장기간 Normal 영상 수집
   │
   ▼
Normal Distribution 확대
   │
   ▼
실제 NG 확보
   │
   ▼
Synthetic ↔ Real NG 비교
   │
   ▼
Threshold / Temporal Rule 최적화
   │
   ▼
장기 False Alarm 검증
   │
   ▼
Jetson 실시간 운영 최적화
   │
   ▼
현장 Pilot
```

주요 고도화 항목은 다음과 같다.

| 영역 | 고도화 내용 |
|---|---|
| Dataset | 계절/조명/장비 상태별 Normal 확대 |
| NG | 실제 Droplet / Leak / Splash 데이터 확보 |
| Gate | Localization Failure Case 추가 학습/보완 |
| PatchCore | Input Size, Layer, Backbone 비교 |
| Threshold | 실제 Normal 장기 분포 기반 결정 |
| Temporal | 3/5 외 다양한 Rule 비교 |
| Runtime | Jetson GPU 최적화 |
| Monitoring | Alarm/영상/Score History 저장 |
| Feedback | 현장 확인 결과를 재학습 Dataset으로 환류 |

---

# 34. 최종 결론

본 POC에서는 CCTV 기반 호스 누수 검출 시스템을 다음 세 단계로 분리하여 검증하였다.

```text
① 검사 가능한가?
       │
       ▼
Feature Matching + Gate


② 같은 위치를 비교하고 있는가?
       │
       ▼
Homography Alignment


③ 정상과 다른가?
       │
       ▼
PatchCore
```

그리고 최종적으로:

```text
Gate
  +
Alignment
  +
PatchCore
  +
Temporal Filter
        │
        ▼
CCTV Leak Monitoring Pipeline
```

형태의 End-to-End 구조를 구축하였다.

정량 검증 결과 Gate 단계에서는 LEFT와 RIGHT 모두 Validation False Positive가 발생하지 않았으며, Realistic Synthetic NG에서는 LEFT 98.9%, RIGHT 94.4%의 Recall을 Normal FP 0 조건에서 확인하였다.

특히 지속적인 `Leak`과 강한 `Splash`는 Normal Score 영역과 명확하게 분리되었으며, 작은 `Droplet`은 상대적으로 검출 난이도가 높은 주요 고도화 대상으로 확인되었다.

따라서 본 POC의 결론은 다음과 같다.

> **CCTV 영상과 정상 데이터만을 이용한 호스 누수 이상탐지 Pipeline의 기술적 구현 가능성을 확인하였다.**

현재 결과는 즉시 양산 적용 가능한 최종 성능을 의미하지 않는다.

그러나 검사 상태 판별, 위치 정렬, 정상 기반 이상탐지, 이상 위치 Visualization, 시간축 Alarm까지 이어지는 전체 기술 구조를 검증하였으며, 실제 NG 데이터 및 장기간 운영 데이터를 추가 확보할 경우 현장 적용 수준으로 고도화할 수 있는 기반을 확보하였다.

---

# 35. 기술 구조 요약

```text
┌────────────────────────────────────────────────────────────────┐
│                         CCTV VIDEO                             │
└──────────────────────────────┬─────────────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Search ROI          │
                    │ LEFT / RIGHT        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Matching    │
                    │ SIFT / ORB          │
                    │ FLANN / BF          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ RANSAC Homography   │
                    └──────────┬──────────┘
                               │
                               ▼
                       ┌──────────────┐
                       │ Gate         │
                       └──────┬───────┘
                              │
                   ┌──────────┴──────────┐
                   │                     │
                  FAIL                  PASS
                   │                     │
                   ▼                     ▼
              Inference Skip      Homography Alignment
                                         │
                                         ▼
                                  Canonical Anomaly ROI
                                         │
                                         ▼
                                      PatchCore
                                         │
                             ┌───────────┴────────────┐
                             │                        │
                             ▼                        ▼
                         Image Score             Anomaly Map
                             │
                             ▼
                         Threshold
                             │
                             ▼
                          Raw NG
                             │
                             ▼
                      Temporal Voting
                          3 / 5
                             │
                             ▼
                        FINAL ALARM
```

---

# 참고 기술

- OpenCV Feature Matching / Homography / RANSAC
- SIFT / ORB Feature Descriptor
- FLANN / Brute Force Matcher
- PatchCore: Towards Total Recall in Industrial Anomaly Detection
- WideResNet50-2 ImageNet Pretrained Backbone
- Coreset Sampling
- K-Nearest Normal Patch Distance
- Temporal Voting
