# CCTV Hose Leak Detection - 01~14 실행 명령어 정리

> 기준: Windows PowerShell  
> 실행 위치: `HoseLeakInspection` 프로젝트 루트  
> Python 가상환경이 있다면 먼저 활성화 후 실행

---

## 전체 실행 순서

```text
01_split_videos.py
    ↓
02_build_gate_dataset.py
    ↓
03_label_gate_dataset.py
    ↓
04_calibrate_gate.py
    ↓
05_evaluate_gate_val.py
    ↓
06_extract_normal_roi.py
    ↓
07_select_anomaly_roi.py
    ↓
08_train_patchcore.py
    ↓
09_generate_synthetic_ng.py
    ↓
10_evaluate_patchcore.py
    ↓
11_validate_anomaly_maps.py
    ↓
12_generate_asset_leak_ng.py
    ↓
13_generate_anomaly_test_video.py
    ↓
14_run_video_inference.py
```

---

# 01. Video Train / Val / Test 분리

**스크립트**

```text
scripts/01_split_videos.py
```

**목적**

원본 CCTV 영상을 Video 단위로 Train / Validation / Test로 분리한다.

현재 POC에서는 18개 영상을 다음과 같이 분리하였다.

```text
Video 01 ~ 10 → Train
Video 11 ~ 14 → Validation
Video 15 ~ 18 → Test
```

**대표 실행**

```powershell
python scripts\01_split_videos.py
```

**결과 구조**

```text
data/videos/
├─ train/
├─ val/
└─ test/
```

---

# 02. Gate Dataset 생성

**스크립트**

```text
scripts/02_build_gate_dataset.py
```

**목적**

Train 또는 Validation 영상에서 일정 FPS로 프레임을 추출하고, LEFT / RIGHT Template Matching 결과와 Gate 관련 metric을 저장한다.

**기본 실행**

```powershell
python scripts\02_build_gate_dataset.py
```

**경로를 직접 지정하는 경우**

```powershell
python scripts\02_build_gate_dataset.py `
    --videos-dir data\videos\train `
    --config config\gate.yaml `
    --output-dir data\gate_dataset\train
```

Validation Dataset 생성 시:

```powershell
python scripts\02_build_gate_dataset.py `
    --videos-dir data\videos\val `
    --config config\gate.yaml `
    --output-dir data\gate_dataset\val
```

**주요 출력**

```text
data/gate_dataset/train/
├─ left/
│  ├─ images/
│  ├─ metadata.csv
│  └─ summary.json
└─ right/
   ├─ images/
   ├─ metadata.csv
   └─ summary.json
```

---

# 03. Gate Dataset 수동 라벨링

**스크립트**

```text
scripts/03_label_gate_dataset.py
```

**목적**

Gate Dataset 이미지를 직접 확인하여:

- `visible`
- `not_visible`
- `ignore`
- localization `good / bad / unknown`

을 라벨링한다.

**LEFT 대표 실행**

```powershell
python scripts\03_label_gate_dataset.py `
    --roi left `
    --sample-per-status 150
```

**RIGHT 대표 실행**

```powershell
python scripts\03_label_gate_dataset.py `
    --roi right `
    --sample-per-status 150
```

**주요 옵션**

```text
--dataset-dir
--roi {left,right}
--status
--sample-per-status
--seed
--unlabeled-only
--max-width
--max-height
--save-every
```

`--sample-per-status 0`이면 전체 이미지를 대상으로 한다.

**기본 Dataset**

```text
data/gate_dataset/train
```

**GUI 주요 Key**

```text
1            visible
2            not_visible
3            ignore

G            localization good
B            localization bad
U            localization unknown

0            label clear

A / ←        이전 이미지
D / → / Space 다음 이미지

W            저장
Q / ESC      저장 후 종료
```

라벨 결과는 각 ROI의 `metadata.csv`에 기록된다.

---

# 04. Gate Threshold Calibration

**스크립트**

```text
scripts/04_calibrate_gate.py
```

**목적**

수동 라벨링한 Train Gate Dataset을 이용하여 Gate threshold 후보를 계산한다.

**기본 실행**

```powershell
python scripts\04_calibrate_gate.py
```

**전체 옵션 지정 예시**

```powershell
python scripts\04_calibrate_gate.py `
    --roi all `
    --dataset-dir data\gate_dataset\train `
    --output-dir outputs\gate_calibration `
    --target-recall 0.98 `
    --top-k 10
```

ROI별 실행:

```powershell
python scripts\04_calibrate_gate.py --roi left
```

```powershell
python scripts\04_calibrate_gate.py --roi right
```

**주요 출력**

```text
outputs/gate_calibration/
├─ left/
└─ right/
```

---

# 05. Gate Validation 평가

**스크립트**

```text
scripts/05_evaluate_gate_val.py
```

**목적**

04단계에서 결정한 Gate 조건을 Validation Dataset에 적용하여 TP / TN / FP / FN을 평가한다.

**기본 실행**

```powershell
python scripts\05_evaluate_gate_val.py
```

**전체 옵션 지정 예시**

```powershell
python scripts\05_evaluate_gate_val.py `
    --roi all `
    --dataset-dir data\gate_dataset\val `
    --gate-dir outputs\gate_calibration `
    --output-dir outputs\gate_validation
```

ROI별 실행:

```powershell
python scripts\05_evaluate_gate_val.py --roi left
```

```powershell
python scripts\05_evaluate_gate_val.py --roi right
```

---

# 06. Normal ROI 추출

**스크립트**

```text
scripts/06_extract_normal_roi.py
```

**목적**

Gate를 통과하고 localization이 정상인 프레임을 Homography Alignment 후 고정 Anomaly ROI로 Crop한다.

PatchCore 학습/검증용 정상 데이터를 생성한다.

## Train 추출

```powershell
python scripts\06_extract_normal_roi.py `
    --split train `
    --save-aligned
```

## Validation 추출

```powershell
python scripts\06_extract_normal_roi.py `
    --split val `
    --save-aligned
```

ROI를 지정하려면:

```powershell
python scripts\06_extract_normal_roi.py `
    --split train `
    --roi left `
    --save-aligned
```

**주요 옵션**

```text
--split {train,val}
--roi {all,left,right}
--config config/normal_roi.yaml
--dataset-root data/gate_dataset
--gate-dir outputs/gate_calibration
--output-root data/normal_roi
--save-aligned
```

**주요 출력**

```text
data/normal_roi/
├─ train/
│  ├─ left/
│  │  ├─ images/
│  │  └─ aligned/
│  └─ right/
└─ val/
   ├─ left/
   └─ right/
```

---

# 07. Anomaly ROI 선택

**스크립트**

```text
scripts/07_select_anomaly_roi.py
```

**목적**

정렬된 Canonical 이미지에서 PatchCore가 실제로 검사할 최종 Anomaly ROI를 GUI로 선택한다.

## LEFT

```powershell
python scripts\07_select_anomaly_roi.py --roi left
```

## RIGHT

```powershell
python scripts\07_select_anomaly_roi.py --roi right
```

**기본 설정**

```text
--split train
--config config/normal_roi.yaml
--normal-root data/normal_roi
--sample-count 30
```

**GUI Key**

```text
R       ROI 재선택
A / D   이전 / 다음 이미지
S       저장
Q       종료
```

최종 ROI는 다음 파일에 저장된다.

```text
config/normal_roi.yaml
```

> ROI를 수정한 경우 06단계 Normal ROI 추출을 다시 실행한다.

---

# 08. PatchCore 학습

**스크립트**

```text
scripts/08_train_patchcore.py
```

**목적**

Real Normal ROI만 사용하여 LEFT / RIGHT PatchCore 모델을 학습한다.

## LEFT

```powershell
python scripts\08_train_patchcore.py `
    --roi left `
    --device cpu `
    --save-maps
```

## RIGHT

```powershell
python scripts\08_train_patchcore.py `
    --roi right `
    --device cpu `
    --save-maps
```

## LEFT + RIGHT 전체

```powershell
python scripts\08_train_patchcore.py `
    --roi all `
    --device cpu
```

**주요 옵션**

```text
--config config/patchcore.yaml
--roi {left,right,all}
--device {auto,cpu,cuda}
--save-maps
```

**주요 출력**

```text
models/
├─ patchcore_left/
│  ├─ patchcore.pt
│  ├─ summary.json
│  ├─ train_scores.csv
│  ├─ val_scores.csv
│  └─ val_maps/
└─ patchcore_right/
```

---

# 09. Procedural Synthetic NG 생성

**스크립트**

```text
scripts/09_generate_synthetic_ng.py
```

**목적**

정상 Validation ROI에 procedural 방식의 Synthetic NG를 생성한다.

초기 검증 종류:

```text
droplet
wet_spot
thin_stream
```

## Medium Synthetic 생성 예시

```powershell
python scripts\09_generate_synthetic_ng.py `
    --roi all `
    --output-root data\synthetic_ng\medium
```

Easy를 별도로 생성한다면:

```powershell
python scripts\09_generate_synthetic_ng.py `
    --roi all `
    --output-root data\synthetic_ng\easy
```

**출력 예시**

```text
data/synthetic_ng/medium/val/
├─ left/
│  ├─ images/
│  ├─ aligned/
│  ├─ masks/
│  └─ metadata.csv
└─ right/
```

---

# 10. PatchCore Synthetic NG 성능 평가

**스크립트**

```text
scripts/10_evaluate_patchcore.py
```

**목적**

Normal Validation score와 Synthetic NG score를 비교하여:

- Score 분포
- Threshold 후보
- Precision / Recall
- FP / FN
- defect type별 검출률

을 평가한다.

## Medium 평가

```powershell
python scripts\10_evaluate_patchcore.py `
    --roi all `
    --device cpu `
    --batch-size 2 `
    --synthetic-root data\synthetic_ng\medium `
    --output-root outputs\patchcore_validation\medium
```

## Realistic 평가

```powershell
python scripts\10_evaluate_patchcore.py `
    --roi all `
    --device cpu `
    --batch-size 2 `
    --synthetic-root data\synthetic_ng\realistic `
    --output-root outputs\patchcore_validation\realistic
```

**주요 옵션**

```text
--patchcore-config
--synthetic-root
--output-root
--roi {all,left,right}
--device
--batch-size
```

**주요 출력**

```text
all_scores.csv
score_statistics.csv
threshold_search.csv
threshold_candidates.csv
detection_by_type.csv
normal_false_positive.csv
synthetic_missed.csv
score_distribution.png
summary.json
```

---

# 11. Anomaly Map Localization 검증

**스크립트**

```text
scripts/11_validate_anomaly_maps.py
```

**목적**

PatchCore anomaly map과 Synthetic GT mask를 비교하여 모델이 실제 이상 위치를 보고 있는지 검증한다.

## Medium

```powershell
python scripts\11_validate_anomaly_maps.py `
    --roi all `
    --device cpu `
    --batch-size 2 `
    --save-overlays `
    --synthetic-root data\synthetic_ng\medium `
    --output-root outputs\anomaly_map_validation\medium
```

## Realistic

```powershell
python scripts\11_validate_anomaly_maps.py `
    --roi all `
    --device cpu `
    --batch-size 2 `
    --save-overlays `
    --synthetic-root data\synthetic_ng\realistic `
    --output-root outputs\anomaly_map_validation\realistic
```

**주요 확인 Metric**

```text
peak_hit
mean_inside / mean_outside
inside / outside ratio
top1 precision / recall
top5 precision / recall
top10 precision / recall
```

---

# 12. PNG Asset 기반 Realistic Synthetic NG 생성

**스크립트**

```text
scripts/12_generate_asset_leak_ng.py
```

**목적**

배경이 제거된 RGBA PNG Asset을 정상 영상에 합성하여 실제 누수와 유사한 Synthetic NG를 생성한다.

Asset 종류:

```text
droplet
leak
splash
```

**대표 실행**

```powershell
python scripts\12_generate_asset_leak_ng.py `
    --roi all
```

설정과 출력 경로를 직접 지정하려면:

```powershell
python scripts\12_generate_asset_leak_ng.py `
    --roi all `
    --config config\synthetic_ng_realistic.yaml `
    --normal-roi-config config\normal_roi.yaml `
    --output-root data\synthetic_ng\realistic
```

**Asset 경로**

```text
data/synthetic_assets/water/
├─ droplet/
│  ├─ droplet_01.png
│  ├─ droplet_02.png
│  └─ droplet_03.png
├─ leak/
└─ splash/
```

**Synthetic 출력**

```text
data/synthetic_ng/realistic/val/
├─ left/
└─ right/
```

생성 후 다시 10 / 11단계를 실행한다.

---

# 13. Synthetic Anomaly Test Video 생성

**스크립트**

```text
scripts/13_generate_anomaly_test_video.py
```

**목적**

정상 Validation CCTV 영상에:

```text
Normal
  ↓
Droplet
  ↓
Leak
  ↓
Splash
```

형태로 시간축을 가진 Synthetic 누수를 합성하여 End-to-End 검증 영상을 생성한다.

**대표 실행**

```powershell
python scripts\13_generate_anomaly_test_video.py `
    --source-video data\videos\val\VIDEO_NAME.mp4 `
    --config config\anomaly_test_video.yaml `
    --output data\videos\anomaly_test\anomaly_test.mp4
```

현재 시나리오 예시:

```text
10 sec
  ↓
droplet
  ↓
leak
  ↓
splash
  ↓
35 sec
```

**주요 출력**

```text
data/videos/anomaly_test/
├─ anomaly_test.mp4
└─ anomaly_test.events.json
```

`events.json`은 14단계에서 Ground Truth Event / Time To Detect 계산에 사용한다.

---

# 14. End-to-End Video Inference

**스크립트**

```text
scripts/14_run_video_inference.py
```

**목적**

최종 영상 처리 Pipeline을 실행한다.

```text
Video
 ↓
Gate
 ↓
Homography Alignment
 ↓
Anomaly ROI
 ↓
PatchCore
 ↓
Threshold
 ↓
Temporal Filter
 ↓
Alarm
```

---

## Synthetic Anomaly Video 평가

```powershell
python scripts\14_run_video_inference.py `
    --video data\videos\anomaly_test\anomaly_test.mp4 `
    --config config\video_inference.yaml `
    --output-root outputs\video_inference
```

`anomaly_test.events.json`은 동일 경로에 있으면 자동으로 읽는다.

**출력**

```text
outputs/video_inference/anomaly_test/
├─ result.csv
├─ summary.json
├─ event_results.csv
└─ annotated.mp4
```

---

## 실제 Test 영상 전체 평가

```powershell
python scripts\14_run_video_inference.py `
    --input-dir data\videos\test `
    --config config\video_inference.yaml `
    --output-root outputs\video_inference\test
```

**주요 확인 결과**

```text
Gate pass rate

Normal Score
- p50
- p95
- p99
- max

Raw NG count
Temporal Alarm count
False Alarm / hour

Inference Time
- mean
- p95
- max
```

---

# 자주 사용하는 전체 실행 흐름

## Gate 구축

```powershell
python scripts\02_build_gate_dataset.py `
    --videos-dir data\videos\train `
    --config config\gate.yaml `
    --output-dir data\gate_dataset\train

python scripts\03_label_gate_dataset.py `
    --roi left `
    --sample-per-status 150

python scripts\03_label_gate_dataset.py `
    --roi right `
    --sample-per-status 150

python scripts\04_calibrate_gate.py --roi all
```

---

## Gate Validation

```powershell
python scripts\02_build_gate_dataset.py `
    --videos-dir data\videos\val `
    --config config\gate.yaml `
    --output-dir data\gate_dataset\val

python scripts\05_evaluate_gate_val.py --roi all
```

---

## PatchCore Dataset / Training

```powershell
python scripts\06_extract_normal_roi.py `
    --split train `
    --save-aligned

python scripts\06_extract_normal_roi.py `
    --split val `
    --save-aligned

python scripts\07_select_anomaly_roi.py --roi left

python scripts\07_select_anomaly_roi.py --roi right

python scripts\08_train_patchcore.py `
    --roi all `
    --device cpu
```

---

## Realistic Synthetic Validation

```powershell
python scripts\12_generate_asset_leak_ng.py `
    --roi all

python scripts\10_evaluate_patchcore.py `
    --roi all `
    --device cpu `
    --batch-size 2 `
    --synthetic-root data\synthetic_ng\realistic `
    --output-root outputs\patchcore_validation\realistic

python scripts\11_validate_anomaly_maps.py `
    --roi all `
    --device cpu `
    --batch-size 2 `
    --save-overlays `
    --synthetic-root data\synthetic_ng\realistic `
    --output-root outputs\anomaly_map_validation\realistic
```

---

## 최종 Video Test

```powershell
python scripts\13_generate_anomaly_test_video.py `
    --source-video data\videos\val\VIDEO_NAME.mp4 `
    --config config\anomaly_test_video.yaml `
    --output data\videos\anomaly_test\anomaly_test.mp4

python scripts\14_run_video_inference.py `
    --video data\videos\anomaly_test\anomaly_test.mp4 `
    --config config\video_inference.yaml `
    --output-root outputs\video_inference

python scripts\14_run_video_inference.py `
    --input-dir data\videos\test `
    --config config\video_inference.yaml `
    --output-root outputs\video_inference\test
```

---

# 핵심 Config 파일

```text
config/
├─ gate.yaml
├─ normal_roi.yaml
├─ patchcore.yaml
├─ synthetic_ng.yaml
├─ synthetic_ng_realistic.yaml
├─ anomaly_test_video.yaml
└─ video_inference.yaml
```

> 최종 14단계에서는 별도의 `roi.yaml`을 사용하지 않는다.  
> Search ROI는 `video_inference.yaml`의 `gate.left.search_roi`, `gate.right.search_roi`에 직접 지정한다.

