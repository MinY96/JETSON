# Jetson 기반 CCTV 실시간 CST 이상탐지 시스템 구축 기술보고서

**프로젝트명:** RMV CST(Cassette) 실시간 이상탐지 시스템  
**최종 운영 버전:** `final_cst_inference_operation_v6_3.py`  
**운영 플랫폼:** NVIDIA Jetson / Ubuntu / Python / OpenCV / PyTorch / TensorRT  
**최종 현장 검증:** 2026-08-25 15:00, 약 1시간

---

## 1. 보고서 요약

본 프로젝트는 제조 현장의 CCTV 영상을 활용하여 좌·우 CST(Cassette)의 상태를 실시간으로 판정하고, 작업 오류 및 Wafer 잔류와 같은 이상 상황을 자동 검출하기 위해 구축하였다.

초기에는 Empty Reference 기반의 CST 존재 여부와 색상 Rule, Wafer 분류 모델을 개별적으로 적용하였으나, 실제 현장에서는 CST 투입·회전·제거·교체 과정에서 순간적인 영상 변화가 발생하여 단일 프레임 판정만으로 안정적인 운영이 어려웠다.

이를 개선하기 위해 최종 v6.3에서는 다음 구조를 적용하였다.

- READY/NOT_READY 전용 MobileNetV3-Small TensorRT 모델을 통한 추론 가능 상태 판단
- Wafer MobileNetV3-Small 재학습 모델 및 4/5 Temporal Voting 적용
- Pink 1차 판정 + Pink/Black 2차 검증의 Cascade Rule 적용
- Empty Reference를 CST 실제 제거 이벤트 확인용 독립 신호로 유지
- Cycle State Machine을 통한 WAIT_READY → RIGHT_PRECHECK → COLOR_EVALUATE → LOCKED → CHANGEOVER 상태 관리
- WARNING/ABNORMAL 상태 Lock 및 PATLITE 경광등/부저 연동
- 일 단위 로그, 이벤트 영상, 재학습용 프레임 자동 수집
- systemd 기반 24시간 자동 실행 및 Thermal Watchdog 적용

2026-08-25 현장 검증 결과, 색상 불일치 2개 조합, 우측 CST Slot 1~5 Wafer 낱장 검출, 실제 생산 동작 Cycle 및 기존 오검출 재현 여부를 확인하였으며 주요 테스트 항목에서 정상 동작을 확인하였다.

---

# 2. 프로젝트 배경 및 목적

## 2.1 배경

CST 작업 과정에서는 좌측과 우측 CST의 상태 및 내부 Wafer 잔류 여부를 작업자가 확인해야 한다.

특히 다음과 같은 상황은 작업 오류 또는 후속 공정 문제로 이어질 가능성이 있어 자동 감시가 필요하였다.

1. 좌·우 CST의 종류 또는 색상 조합이 일치하지 않는 경우
2. 작업 완료 후 우측 CST 내부에 Wafer가 남아 있는 경우
3. CST가 완전히 안착되지 않은 상태에서 판정이 수행되는 경우
4. CST 제거·교체 과정의 순간 영상이 신규 CST로 잘못 인식되는 경우
5. 순간 오검출이 즉시 WARNING 또는 ABNORMAL 알람으로 이어지는 경우

기존 CCTV를 활용하면서 별도의 고가 Vision 설비를 추가하지 않고 Edge 장비에서 실시간 처리할 수 있는 시스템 구축을 목표로 하였다.

## 2.2 구축 목적

본 시스템의 핵심 목적은 다음과 같다.

- CCTV 영상에서 LEFT / RIGHT CST 상태 실시간 감시
- CST가 실제 판정 가능한 상태인지 READY/NOT_READY 판별
- 우측 CST 내부 Wafer 잔류 여부 자동 판정
- 좌·우 CST 색상 일치 여부 판정
- 이상 발생 시 작업자에게 경광등 및 부저로 즉시 알림
- CST 교체 과정에서 이전 판정을 안정적으로 유지하고 신규 Cycle 자동 인식
- 24시간 무인 운영을 위한 서비스 자동 실행 및 열 관리
- 로그/영상/프레임 자동 수집을 통한 지속적인 모델 개선 기반 확보

---

# 3. 시스템 구성

## 3.1 전체 구성

```text
CCTV / HDMI Capture
        │
        ▼
1920 × 1080 Frame
        │
        ├───────────────┐
        ▼               ▼
   LEFT CST ROI     RIGHT CST ROI
        │               │
        ├──── READY / NOT_READY Model ────┐
        │                                  │
        ├──── Empty Reference 비교 ────────┤
        │                                  │
        │                      RIGHT Wafer Model
        │                           + 4/5 Voting
        │                                  │
        └──── Pink 2-Stage Rule ───────────┘
                         │
                         ▼
                 Cycle State Machine
                         │
        ┌────────────────┼─────────────────┐
        ▼                ▼                 ▼
      NORMAL          ABNORMAL          WARNING
        │                │                 │
        └────────── PATLITE Alarm ─────────┘
                         │
                         ▼
              CSV / Event Video / Frame
```

## 3.2 입력 영상 및 ROI

운영 입력은 HDMI-USB Capture 기반 실시간 영상이며 기본 설정은 다음과 같다.

| 항목 | 설정 |
|---|---|
| 입력 모드 | Capture |
| 해상도 | 1920 × 1080 |
| FPS | 30 |
| FOURCC | MJPG |
| LEFT ROI | `(3, 85, 300, 410)` |
| RIGHT ROI | `(1210, 240, 540, 600)` |

ROI는 전체 영상에서 CST 판정에 필요한 영역만 추출하여 모델 추론량과 불필요한 배경 영향을 줄이도록 구성하였다.

## 3.3 Edge 운영 환경

운영 시스템은 Jetson 환경에서 모든 판정을 Local로 수행한다.

주요 사용 기술은 다음과 같다.

- Python
- OpenCV
- PyTorch / torchvision
- TensorRT
- NumPy
- PIL
- PyUSB
- systemd

READY 및 Wafer 모델은 TensorRT FP16 Engine으로 배포하여 Edge GPU 환경에서 실시간 추론하도록 구성하였다.

---

# 4. 핵심 판정 기술

## 4.1 Empty Reference 기반 CST 제거 판정

Empty 판정은 AI 분류 모델이 아닌 Reference Image 비교 방식을 사용한다.

Empty Reference는 실제 라인에서 CST가 없는 상태의 LEFT / RIGHT ROI를 여러 장 저장하여 구성한다.

### 전처리

```text
ROI
 ↓
Grayscale
 ↓
Center Crop
 ↓
Gaussian Blur
 ↓
Reference Image 비교
```

현재 프레임과 Empty Reference 간 비교 시 평균 밝기를 맞춰 조명 차이를 보정한 뒤 절대 차분을 계산한다.

3 미만의 미세한 픽셀 차이는 노이즈로 간주하여 제거하고, 각 Reference와의 평균 차분 값 중 가장 작은 값을 Empty Score로 사용한다.

```text
Empty Score = min(mean(abs(Current - Reference_i)))
```

운영 Threshold:

| 구분 | Threshold |
|---|---:|
| LEFT | 16.7258 |
| RIGHT | 26.9675 |

Score가 Threshold 이하이면 Empty로 판단한다.

### Empty 판정의 역할

최종 v6.3에서는 Empty 판정을 READY 판정과 분리하였다.

- READY: 현재 CST가 추론 가능한 정상 안착 상태인가?
- EMPTY: CST가 실제로 제거되었는가?

즉, 손/팔/회전/일시적 가림에 의해 READY 모델이 NOT_READY를 출력하더라도 Empty가 아니면 CST가 제거된 것으로 판단하지 않는다.

이 구조를 통해 CST 교체 과정의 오인식을 줄였다.

---

# 5. READY / NOT_READY 모델

## 5.1 적용 목적

초기 시스템은 Empty의 역값을 사실상 CST 존재 상태로 활용하였으나, 실제 현장에서는 다음 문제가 있었다.

- CST가 회전 중인 경우
- 완전히 안착되기 전의 중간 상태
- 작업자 손/팔 또는 장비가 ROI를 가리는 경우
- CST는 존재하지만 모델 판정을 수행하기 적합하지 않은 경우

따라서 CST 존재 여부와 "판정 가능한 자세"를 분리하기 위해 READY/NOT_READY 전용 모델을 추가하였다.

## 5.2 모델 구성

- Architecture: MobileNetV3-Small
- Class: `ready`, `not_ready`
- Input: LEFT / RIGHT CST ROI
- Input Size: 224 × 224
- Deployment: TensorRT FP16
- Positive Label: `ready`
- 운영 Confidence Threshold: 0.5

전처리는 ImageNet 기반 MobileNet 계열과 동일한 Normalize 설정을 적용한다.

```text
Resize 224×224
→ ToTensor
→ Normalize
   mean = [0.485, 0.456, 0.406]
   std  = [0.229, 0.224, 0.225]
```

## 5.3 READY 안정화

AI 모델의 단일 프레임 결과를 바로 상태로 사용하지 않고 Hold 방식을 적용하였다.

운영값:

```text
ready_check_interval = 1
ready_hold_frames    = 5
ready_stable_frames  = 10
```

동일 READY/NOT_READY 결과가 일정 프레임 이상 유지되어야 stable state를 변경하며, 좌·우가 모두 READY인 상태도 일정 기간 유지된 후 색상 판정 단계로 이동한다.

---

# 6. Wafer 잔류 검출 모델

## 6.1 적용 목적

Wafer 검출은 우측 CST에 Wafer가 남아 있는지를 확인하기 위한 핵심 안전 판정이다.

Class는 다음 두 가지로 구성한다.

```text
no_wafer
wafer_exist
```

## 6.2 모델 선정

Wafer 분류 모델은 MobileNetV3-Small을 사용하였다.

MobileNetV3-Small을 선정한 주요 이유는 다음과 같다.

- Jetson과 같은 Edge Device에서 상대적으로 작은 연산량
- 분류 성능 대비 경량 구조
- TensorRT 변환 및 FP16 최적화 용이
- ImageNet Pretrained Weight를 활용한 Transfer Learning 가능
- 제한된 제조 현장 Dataset에서도 Feature 재사용 가능

학습 파이프라인은 torchvision의 ImageNet pretrained MobileNetV3-Small을 기반으로 마지막 Classifier를 Wafer 2-Class 문제에 맞게 변경하는 Transfer Learning 구조로 구현하였다.

## 6.3 학습 안정화

운영 모델 학습 파이프라인에는 다음 방법을 적용하였다.

- ImageNet Pretrained Weight
- Class imbalance 보정용 Class Weight
- Hard Example / Dataset Manifest 기반 Sample Weight
- AdamW Optimizer
- Validation 기반 Best Model 선정
- Wafer Recall 우선 Threshold 탐색
- Test Dataset 별도 평가
- ONNX Export
- TensorRT FP16 Engine 변환

운영 학습 스크립트 기본 설정은 다음과 같이 구성하였다.

```text
Epoch       : 30
Batch Size  : 32
Learning Rate: 1e-4
ONNX Opset  : 17
```

운영 Threshold는 Validation에서 Wafer Recall 목표를 우선 만족하는 Threshold 중 False Positive를 줄일 수 있는 값을 탐색한다.

선택된 원본 Threshold는 Deployment Model의 Logit에 반영하고, 최종 운영 프로그램에서는 공통적으로 `wafer_conf_threshold=0.5`를 사용할 수 있도록 구성하였다.

## 6.4 Temporal Voting

단일 프레임 Wafer 오검출로 즉시 WARNING이 발생하지 않도록 M/N Temporal Voting을 적용하였다.

운영 설정:

```text
Wafer Check Interval : 2 frame
Vote Window          : 5 inference
WARNING              : wafer_exist 4 / 5 이상
NO_WAFER PASS        : no_wafer 4 / 5 이상
Confidence Threshold : 0.5
```

즉, 최근 5회의 추론 중 4회 이상 동일 방향의 결과가 누적되어야 최종 상태를 확정한다.

이는 순간적인 빛 반사, 작업자 동작, CST 이동 등에 의한 단일 프레임 오검출을 줄이기 위한 구조이다.

---

# 7. Pink / Non-Pink 색상 판정

## 7.1 목적

LEFT와 RIGHT CST의 색상 조합이 동일한지 확인하기 위한 판정이다.

AI Classification 대신 비교적 명확한 색상 특징을 이용한 Rule 방식으로 구현하였다.

## 7.2 1차 Pink / Non-Pink 판정

ROI에 Center Crop 및 Gaussian Blur를 적용한 후 RGB 채널 차이를 계산한다.

주요 Feature:

```text
R-G median / mean
R-B median / mean
B-G median / mean
```

Pink 색상의 R Channel 우세 특성을 활용하면서 전체 밝기 변화 영향을 줄이기 위한 방식이다.

현재 운영 1차 Rule:

| ROI | Feature | Rule |
|---|---|---|
| LEFT | `rg_diff_median` | `>= 8.5` |
| RIGHT | `rg_diff_median` | `>= 14.0` |

## 7.3 2차 Pink / Black Verifier

현장 데이터에서 어두운 Black CST가 특정 조명 조건에서 1차 Pink Rule을 만족하는 경우를 보완하기 위해 2차 검증을 추가하였다.

1차에서 Pink 후보로 판정된 경우에만 절대 밝기/색상 기반 독립 Feature를 사용하여 재검증한다.

현재 운영 Rule:

| ROI | Feature | Rule |
|---|---|---|
| LEFT | `r_median` | `>= 47.0` |
| RIGHT | `r_median` | `>= 68.5` |

최종 구조:

```text
Primary Rule
     │
     ├─ NON_PINK → 최종 NON_PINK
     │
     └─ PINK Candidate
              │
              ▼
       Pink / Black Verifier
              │
         ┌────┴─────┐
         ▼          ▼
       PINK       BLACK
                    │
                    ▼
                NON_PINK
```

이를 통해 단순 Color Threshold보다 Black 오검출에 대한 안정성을 높였다.

---

# 8. 최종 Cycle State Machine

단순 프레임 판정이 아니라 실제 제조 작업 흐름을 반영하기 위해 Cycle State Machine을 적용하였다.

## 8.1 전체 상태

```text
WAIT_READY
    ↓
RIGHT_PRECHECK
    ↓
COLOR_EVALUATE
    ↓
LOCKED
    ↓
CHANGEOVER
    ↓
RIGHT_PRECHECK (New Cycle)
```

## 8.2 WAIT_READY

LEFT / RIGHT CST의 READY 및 Empty 상태를 지속 감시한다.

어느 한쪽 CST라도 READY가 확인되면 `RIGHT_PRECHECK` 단계로 이동한다.

## 8.3 RIGHT_PRECHECK

RIGHT CST가 READY일 때 Wafer 모델을 실행한다.

```text
wafer_exist 4/5
   → WARNING Lock

no_wafer 4/5
   → Precheck PASS
```

RIGHT CST가 NOT_READY가 되면 기존 Voting 결과를 초기화하여 이전 판정이 신규 상태에 영향을 주지 않도록 한다.

## 8.4 COLOR_EVALUATE

다음 조건이 모두 만족되면 색상을 1회 판정한다.

- RIGHT Wafer Precheck = NO_WAFER
- LEFT READY
- RIGHT READY
- 양쪽 READY Stable 조건 만족

LEFT / RIGHT의 Pink 2-stage 결과를 비교한다.

```text
LEFT == RIGHT
→ NORMAL

LEFT != RIGHT
→ ABNORMAL
```

## 8.5 LOCKED

NORMAL / ABNORMAL / WARNING 결과가 확정되면 해당 Cycle 동안 결과를 Lock한다.

Lock 이후에는 Wafer/Pink 모델을 반복 실행하지 않으며 CST의 실제 제거 이벤트만 감시한다.

이 구조를 통해 작업 중 CST의 순간 자세 변화가 이미 확정된 결과를 바꾸는 것을 방지하였다.

## 8.6 CHANGEOVER

CST 교체 여부는 다음 두 신호의 조합으로 판단한다.

```text
REMOVED
= EMPTY=True AND READY=False

REPLACED
= REMOVED 경험 후
  READY=True AND EMPTY=False
```

### NORMAL

LEFT 또는 RIGHT 중 한쪽이라도

```text
REMOVED → REPLACED
```

가 완료되면 현재 좌·우 조합이 새로운 Pair로 변경된 것으로 판단하고 신규 Cycle을 시작한다.

### WARNING

WARNING의 원인은 RIGHT CST 내부 Wafer이므로 LEFT 상태 변화는 WARNING 해제 조건으로 사용하지 않는다.

```text
RIGHT REMOVED
→ CHANGEOVER
→ PATLITE OFF
→ RIGHT REPLACED
→ New Cycle
```

### ABNORMAL / UNKNOWN

LEFT 또는 RIGHT 중 한쪽 이상이 교체되면 다시 전체 판정을 수행한다.

---

# 9. 최종 판정 및 알람 정책

| 상태 | 조건 | PATLITE |
|---|---|---|
| NORMAL | Wafer 없음 + 좌/우 색상 일치 | OFF |
| ABNORMAL | Wafer 없음 + 좌/우 색상 불일치 | ORANGE |
| WARNING | RIGHT CST Wafer 존재 | RED |
| IDLE / CHECKING | 판정 대기/교체 중 | OFF |

WARNING은 우측 Wafer 잔류에 대한 높은 우선순위 이상으로 관리한다.

ABNORMAL/WARNING 상태에서 문제 CST가 실제 제거되어 CHANGEOVER에 진입하면 이전 Lock 정보는 로그상 유지하되 작업자 Overlay는 IDLE로 변경하고 PATLITE도 즉시 OFF한다.

---

# 10. 운영 Overlay

운영 Overlay에는 작업자가 즉시 판단해야 하는 정보만 표시한다.

주요 표시 정보:

- LEFT READY / NOT READY / EMPTY
- RIGHT READY / NOT READY / EMPTY
- RIGHT Wafer 상태
- LEFT / RIGHT Pink 상태
- 최종 STATUS
- 작업자 안내 Message

TensorRT 내부 Probability, Voting Count, Empty Score 등의 상세 Debug 값은 화면에서 제거하고 CSV Log에 기록한다.

이를 통해 현장 작업자가 복잡한 모델 정보를 해석하지 않고도 현재 상태를 즉시 확인할 수 있도록 구성하였다.

---

# 11. PATLITE 경광등 연동

PATLITE LR6-USB와 USB 통신하여 이상 상태를 작업자에게 전달한다.

```text
WARNING
→ RED LED
→ High Buzzer

ABNORMAL
→ YELLOW/ORANGE LED
→ Low Buzzer

NORMAL / IDLE
→ OFF
```

운영 중 USB 연결이 끊기더라도 추론 프로그램 자체는 계속 동작하며 설정된 주기로 PATLITE 재연결을 시도한다.

재연결 성공 시 현재 판정 상태를 즉시 다시 적용한다.

이 구조를 통해 PATLITE USB의 일시적인 통신 장애가 전체 Vision System 장애로 확산되는 것을 방지하였다.

---

# 12. Logging 및 데이터 자동 수집

## 12.1 Daily CSV

운영 로그는 일 단위 CSV로 저장한다.

주요 기록 항목:

- Timestamp
- Frame Index
- Cycle State
- Inference Time
- READY 상태 및 Score
- Empty 상태 및 Score
- Changeover 상태
- 최종 Status
- Pink 판정
- Wafer Probability
- Voting 결과
- Alarm Action

프로세스가 비정상 종료되더라도 데이터 유실을 최소화하기 위해 매 기록 시 Flush한다.

## 12.2 Event Clip

WARNING 또는 ABNORMAL 발생 시 이상 발생 시점의 전·후 Frame을 포함한 MP4 Clip을 자동 저장한다.

이를 통해 사후 원인 분석 및 현장 VOC 검토가 가능하다.

## 12.3 Auto Frame Collector

향후 모델 고도화를 위해 운영 중 Frame을 자동 수집한다.

```text
auto_frames/
├── random/
│   └── YYYYMMDD/
│       ├── full
│       ├── left
│       ├── right
│       └── meta
└── review_queue/
    ├── WARNING
    ├── ABNORMAL
    ├── UNKNOWN
    └── INCOMPLETE
```

Random Sampling은 전체 운영 분포를 확보하기 위한 용도이며, Review Queue는 이상 또는 난이도 높은 Case를 우선 검수하기 위한 용도이다.

디스크 용량 제한과 최소 여유 공간 조건을 적용하여 장기간 운영 중 저장 데이터가 시스템을 가득 채우지 않도록 구성하였다.

---

# 13. 24시간 운영 및 Thermal Watchdog

## 13.1 systemd 서비스

운영 프로그램은 systemd Service로 등록하여 Jetson 재부팅 시 자동 실행된다.

```text
rmv-cst-inference.service
rmv-cst-thermal.service
```

Inference Service는 최종 `final_cst_inference_operation_v6_3.py`를 실행하며, Thermal Service는 Jetson 온도 및 Inference Service 상태를 감시한다.

## 13.2 Thermal 정책

| 온도 | 동작 |
|---|---|
| `< 75°C` | 정상 운영 |
| `>= 75°C` | Warning Log |
| `>= 82°C` | Inference Service 중지 / Cooling |
| `<= 75°C` | Inference Service 재시작 |
| `>= 88°C` 60초 지속 | Jetson Reboot |

AI 추론 장비의 장시간 GPU 부하에 따른 과열을 방지하고, Cooling 후 자동 복귀할 수 있도록 구성하였다.

---

# 14. 주요 고도화 내용

최종 v6.3까지 진행한 핵심 개선 사항은 다음과 같다.

| 구분 | 초기 | 최종 v6.3 |
|---|---|---|
| CST 상태 | Empty 중심 | READY와 EMPTY 독립 관리 |
| Wafer 판정 | 단일/순간 판정 | MobileNetV3 + 4/5 Temporal Voting |
| 색상 판정 | 단일 Pink Rule | Pink Primary + Pink/Black Verifier |
| Cycle 관리 | 단순 상태 판정 | 5단계 State Machine |
| WARNING | 순간 상태 영향 가능 | RIGHT 제거 전까지 Lock |
| CST 교체 | 조합에 따라 재판정 불안정 | REMOVED→REPLACED 명시적 추적 |
| 알람 | 결과 중심 | CHANGEOVER 진입 즉시 해제 |
| 운영 | Script 실행 | systemd 자동 실행 |
| 장애 대응 | 수동 | PATLITE 자동 재연결 + Thermal Watchdog |
| 데이터 확보 | 수동 수집 | Random/Review Queue 자동 수집 |

---

# 15. 현장 검증 결과

## 15.1 테스트 정보

| 항목 | 내용 |
|---|---|
| 테스트 일시 | 2026-08-25 15:00 |
| 소요 시간 | 약 1시간 |
| 대상 | 최종 v6.3 운영 시스템 |

## 15.2 기능 테스트

### 색상 불일치

1. LEFT Pink / RIGHT Black
   - 색상 불일치 정상 검출
   - ABNORMAL 동작 확인

2. LEFT Black / RIGHT Pink
   - 색상 불일치 정상 검출
   - ABNORMAL 동작 확인

### Wafer 잔류

3. LEFT Any / RIGHT Wafer 낱장
   - Slot 1 ~ Slot 5 각각 테스트
   - 모두 정상 검출
   - WARNING 동작 확인

## 15.3 실제 작업 Cycle

실제 작업 조건에서 다음 Cycle을 수행하였다.

```text
Pink CST       : 3 Cycle
Black-Gray CST : 1 Cycle
```

전체 Cycle에서 정상 동작을 확인하였다.

## 15.4 기존 오검출 개선 검증

기존 시스템에서는 작업 완료 후 LEFT CST를 먼저 제거할 경우, 아직 제거되지 않은 RIGHT 작업완료 CST를 Wafer 잔류 상태로 판단하여 WARNING이 발생하는 사례가 있었다.

v6.3 현장 테스트에서 동일 상황을 재현하여 확인한 결과 기존 WARNING 오검출 현상은 재현되지 않았다.

이는 다음 개선의 복합 결과이다.

- READY Gate
- RIGHT Precheck
- Wafer Temporal Voting
- WARNING Lock
- CST REMOVED / REPLACED 상태 추적
- Cycle State Machine

## 15.5 운영 기능

- Inference Service 정상 동작
- Thermal Service 정상 동작
- Jetson 자동 서비스 운영 확인
- PATLITE 경광등 정상 동작

현장 검증 결과 최종 운영 버전 적용이 가능하다고 판단하였으며 지속 운영으로 전환하였다.

---

# 16. 프로젝트 성과

## 16.1 기술적 성과

### 1) CCTV 기반 Edge AI 시스템 구현

기존 CCTV 영상을 활용하면서 서버 전송 없이 Jetson 내부에서 실시간 분석하는 Edge AI 구조를 구축하였다.

### 2) AI와 Rule 기반 알고리즘의 혼합

모든 판정을 하나의 AI 모델로 해결하지 않고 문제 특성에 따라 적합한 기술을 분리 적용하였다.

```text
READY       → Deep Learning
Wafer       → Deep Learning + Temporal Voting
Pink        → Color Feature Rule
Empty       → Reference Image Difference
Cycle       → State Machine
Alarm       → Deterministic Rule
```

이를 통해 설명 가능성과 유지보수성을 확보하였다.

### 3) 프레임 판정에서 공정 Cycle 판정으로 확장

단일 Frame의 Class Prediction을 직접 알람으로 연결하지 않고, 실제 작업 순서를 State Machine으로 모델링하였다.

이는 제조 현장 Vision System에서 중요한 안정성 개선 요소이다.

### 4) 운영 자동화

- systemd 자동 실행
- Thermal Watchdog
- PATLITE 자동 재연결
- Daily Log
- Event Clip
- Auto Frame Collection

까지 포함하여 모델 개발이 아닌 실제 24시간 운영 가능한 시스템 형태로 완성하였다.

### 5) 지속 개선 구조 확보

운영 중 Random Frame 및 Review Queue를 지속적으로 축적하므로 향후 오검출/미검출 사례를 기반으로 READY/Wafer 모델을 반복 개선할 수 있는 구조를 확보하였다.

---

# 17. 운영 및 향후 관리 방향

최종 시스템은 v6.3 기준으로 지속 운영한다.

향후 유지관리 방향은 다음과 같다.

1. 제조팀 VOC 접수
2. 오검출/미검출 Event 및 Frame 확인
3. Review Dataset 구성
4. READY/Wafer 모델 재학습
5. Pink Rule 재보정
6. Offline Video Replay 검증
7. 신규 Model Version 생성
8. `models_current` Symbolic Link 교체
9. 현장 재검증 후 운영 반영

모델 파일은 `models_current` Symbolic Link를 통해 참조하도록 구성되어 있어 향후 모델 Version Update 시 운영 코드의 경로 변경을 최소화할 수 있다.

---

# 18. 결론

본 프로젝트를 통해 CCTV 영상과 Jetson Edge Device를 활용한 CST 실시간 이상탐지 시스템을 구축하였다.

최종 시스템은 단순 AI Classification이 아닌 READY 판정, Empty Reference, Wafer Detection, Pink Cascade Rule, Temporal Voting, Cycle State Machine을 결합하여 실제 제조 작업 흐름을 반영하였다.

특히 최종 v6.3에서는 기존 현장 오검출의 주요 원인이었던 CST 교체 과정과 순간 상태 변화를 명시적으로 관리하도록 개선하였으며, 현장 테스트에서 색상 불일치, Slot별 Wafer 잔류, 실제 작업 Cycle 및 기존 Warning 오검출 Case에 대한 정상 동작을 확인하였다.

또한 PATLITE 알람, Event 기록, 자동 데이터 수집, systemd 서비스 및 Thermal Watchdog을 적용하여 24시간 지속 운영이 가능한 형태로 구축하였다.

향후에는 운영 중 발생하는 VOC와 자동 수집 데이터를 기반으로 모델 및 Rule을 지속 개선하여 시스템의 안정성을 추가 확보할 예정이다.

---

# Appendix A. 최종 운영 주요 Parameter

| Parameter | Value |
|---|---|
| Input Resolution | 1920 × 1080 |
| Capture FPS | 30 |
| LEFT ROI | `(3, 85, 300, 410)` |
| RIGHT ROI | `(1210, 240, 540, 600)` |
| LEFT Empty Threshold | 16.7258 |
| RIGHT Empty Threshold | 26.9675 |
| READY Confidence | 0.5 |
| READY Hold | 5 |
| READY Stable | 10 |
| Wafer Confidence | 0.5 |
| Wafer Check Interval | 2 |
| Wafer Vote Window | 5 |
| Wafer WARNING Vote | 4 / 5 |
| Wafer NO_WAFER Vote | 4 / 5 |
| LEFT Pink Primary | `rg_diff_median >= 8.5` |
| RIGHT Pink Primary | `rg_diff_median >= 14.0` |
| LEFT Pink/Black Verify | `r_median >= 47.0` |
| RIGHT Pink/Black Verify | `r_median >= 68.5` |
| PATLITE Reconnect | 5 sec |
| Auto Frame Max | 5 GB |

---

# Appendix B. 최종 운영 파일 구조 예시

```text
/home/fab1inno/rmv_cst/
├── src/
│   ├── final_cst_inference_operation_v6_3.py
│   └── thermal_watchdog.py
│
├── models_current -> models_YYYYMMDD/
│   ├── ready_model/
│   │   ├── cst_ready_mobilenetv3_small_fp16.engine
│   │   └── cst_ready_class_meta.json
│   │
│   ├── wafer_model_v2/
│   │   ├── wafer_mobilenetv3_small_fp16.engine
│   │   └── wafer_class_meta.json
│   │
│   └── pink_rule/
│       └── pink_two_stage_thresholds.json
│
├── reference_empty/
│   ├── left/
│   └── right/
│
├── logs/
├── history/
├── event/
└── auto_frames/
```

---

# Appendix C. 운영 서비스

```text
rmv-cst-inference.service
└── final_cst_inference_operation_v6_3.py

rmv-cst-thermal.service
└── thermal_watchdog.py
    └── rmv-cst-inference.service 감시
```

Jetson 재부팅 시 두 Service가 자동 실행되도록 `enabled` 상태로 운영한다.
