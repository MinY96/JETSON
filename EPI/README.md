좋아. 지금까지는 구현과 디버깅을 빠르게 진행하면서 **Blade motion 검출과 wafer 간격 측정 문제가 조금 섞인 상태**였습니다. 다시 정리하면, 이 프로젝트의 본질은 단순한 Blade 동작 검출이 아니라 **CST에 Wafer가 정상적인 위치로 안착했는지를 CCTV만으로 자동 확인하는 것**입니다.

## 1. 프로젝트 개요

### 프로젝트명
**CCTV 기반 CST Wafer 안착 상태 자동 검사**

기존 CCTV 영상을 이용해 CST와 Blade의 동작을 분석하고, 공정이 끝난 Wafer가 CST로 복귀하는 순간 **Wafer가 상·하 Wafer 사이에 정상적인 간격으로 안착했는지 자동 판정**하는 것이 목표입니다.

별도의 센서나 추가 카메라 없이 현재 설치된 고정 CCTV를 그대로 활용하는 방향입니다.

### 실제 장비 동작

CST에는 최대 **25장의 Wafer**가 들어가지만 CCTV에는 전체가 한 번에 보이지 않습니다. CST가 상하로 움직이기 때문에 영상에서는 대략 일부 Wafer만 보이고, Wafer는 정면이 아니라 옆면으로 보여 **얇은 수평선 형태**로 나타납니다.

Blade 역시 정면 CCTV에서 수평 구조로 보이며, Wafer보다 폭이 짧고 약간 두껍습니다.

---

# 2. 최종 목적

최종적으로 검출하려는 장면은 **Type 2**, 즉 공정이 완료된 Wafer가 Chamber에서 CST로 돌아오는 과정입니다.

실제 동작 순서는 다음과 같이 정리할 수 있습니다.

```text
공정 완료 Wafer
      ↓
Blade가 Wafer를 가지고 CST 방향으로 진입
      ↓
Blade 완전 진입
      ↓
CST가 소폭 상승
      ↓
Wafer가 CST 홈에 안착
      ↓
CST 상승 종료
      ↓
CST STOP
      ↓
★ GAP 측정 시점 ★
      ↓
Blade 후퇴
```

핵심은 **CST가 상승을 끝낸 직후이고 Blade가 아직 빠지기 전인 순간**입니다.

이 한 프레임에서 다음 구조를 측정합니다.

```text
──────────────  Upper Wafer
       ↑
       │ d_top
       ↓
══════════════  Blade Center
       ↑
       │ d_bottom
       ↓
──────────────  Lower Wafer
```

계산값은

```python
d_top = blade_center_y - upper_wafer_y
d_bottom = lower_wafer_y - blade_center_y

total_gap = d_top + d_bottom

top_ratio = d_top / total_gap
bottom_ratio = d_bottom / total_gap
```

입니다.

예를 들어

```text
Upper ↔ Blade = 12 px
Blade ↔ Lower = 13 px

→ 48 : 52
→ 약 5 : 5
```

와 같은 결과를 얻는 것이 목적입니다.

중요한 것은 **절대 pixel 거리보다는 상/하 간격의 상대 비율**입니다. 카메라 설치거리나 영상 크기가 조금 변해도 상대 비율은 비교적 안정적으로 사용할 수 있기 때문입니다.

---

# 3. 전체 문제를 3단계로 분리

앞으로는 이 세 문제를 완전히 분리해서 개발하는 게 좋습니다.

```text
[STEP 1]
Type 2 Event Detection
긴 CCTV 영상에서
CST로 Wafer가 돌아오는 장면 검출
        ↓
[STEP 2]
Measurement Timing Detection
Type 2 clip에서
CST UP → STOP 순간 검출
        ↓
[STEP 3]
Wafer Gap Measurement
해당 프레임에서
Upper Wafer / Blade / Lower Wafer 검출
        ↓
Gap / Ratio 계산
```

이렇게 분리하는 것이 현재 가장 중요합니다.

---

# 4. STEP 1 — 긴 CCTV 영상에서 Type 2 검출

원본은 대부분 움직임이 없는 긴 CCTV 영상입니다.

관찰된 주요 움직임은 크게 네 종류였습니다.

| Type | 동작 |
|---|---|
| Type 1 | CST → Chamber로 Wafer 반출 |
| **Type 2** | **Chamber → CST로 공정 완료 Wafer 복귀** |
| Type 3 | Blade 진입 전 위치 조정 |
| Type 4 | CST 이동 |

기존 Classical CV 기반 motion detector를 적용했을 때 검출된 결과는:

```text
Type 1 : 25
Type 2 : 13
Type 3 : 17
Type 4 : 1

Total : 56 clips
```

특히 Type 1은 실제 25장의 Wafer 반출 동작을 모두 검출했습니다.

반면 우리가 필요한 **Type 2는 현재 13개만 확보된 상태**입니다.

### 사용한 방법

Blade가 카메라 쪽으로 움직이므로 단순한 X/Y Optical Flow보다 **크기가 확대/축소되는 움직임**을 사용했습니다.

```text
Blade 접근
→ 영상에서 확대
→ EXPANDING

Blade 후퇴
→ 영상에서 축소
→ SHRINKING
```

Farneback Dense Optical Flow를 사용하고 radial flow를 계산했습니다.

다만 Type 2에서는 Blade가 Wafer를 들고 있기 때문에 긴 Wafer 수평선까지 Optical Flow에 영향을 주어 Type 1보다 검출이 어려운 것으로 보고 있습니다.

### 향후 개선

STEP 1은 지금 당장 건드리지 않습니다.

현재 확보된 **13개의 Type 2 clip만으로 STEP 2와 STEP 3을 먼저 완성**합니다.

그 후 긴 영상으로 돌아가 Type 2 Recall을 개선합니다.

---

# 5. STEP 2 — 정확한 Measurement Frame 찾기

13개의 Type 2 clip은 각각 약 3~5초이고, 중요한 점은 **13개 모두 우리가 원하는 측정 순간을 포함하고 있다는 것**입니다.

따라서 현재 문제는

> Type 2인가?

가 아니라

> 이 3~5초 안에서 정확히 어느 프레임을 측정해야 하는가?

입니다.

현재까지 사용한 방법은 Optical Flow의 vertical component였습니다.

```text
CST 이동
↓
|flow_y| 증가

CST STOP
↓
flow_y ≈ 0
```

즉,

```text
CST UP
   ↓
vertical flow 발생
   ↓
vertical flow 감소
   ↓
stable
   ↓
Measurement Frame
```

으로 잡았습니다.

하지만 현재 사람이 `measurement_frame.jpg`를 확인했을 때 **13개 중 약 6개만 원하는 순간**이었습니다.

### 왜 실패하는가

Blade 역시 동시에 움직이기 때문입니다.

현재 ROI에서 Optical Flow를 계산하면

```text
CST 움직임
+
Blade 움직임
+
Wafer 움직임
+
미세 진동
```

이 섞입니다.

그래서 단순히

```text
vertical pulse → stable
```

만 찾으면 Blade 움직임 이후의 정지나 다른 순간을 CST STOP으로 오인할 수 있습니다.

### 개선 방향

Type 2의 전체 시퀀스를 활용해야 합니다.

특히 측정 시점에는 매우 강한 특징이 있습니다.

```text
CST UP
   ↓
CST STOP
   ↓
[ Measurement ]
   ↓
Blade OUT
```

따라서 앞으로는 단순히 앞에서부터 `UP → STOP`을 찾기보다는,

**Blade OUT을 먼저 찾고 그 직전의 CST UP → STOP 구간을 역추적**하는 방법이 더 적합합니다.

3~5초짜리 이미 Type 2라고 알려진 clip이므로 실시간 causal detector처럼 동작할 필요도 없습니다.

---

# 6. STEP 3 — Wafer / Blade 검출

현재 가장 큰 기술적인 문제입니다.

지금까지의 방식은 Sobel-Y를 사용했습니다.

Wafer가 수평선이므로 Y 방향 밝기 변화가 크게 발생합니다.

```text
──────── wafer
↑
강한 Y-gradient
```

따라서

```python
Sobel(gray, dx=0, dy=1)
```

후 X 방향으로 값을 집계하여 Y-profile을 만들었습니다.

개념적으로는:

```text
Edge
 │
 │        ▲ wafer
 │        │
 │   ▲    │
 │   │    │      ▲
 │___│____│______│________ Y
```

peak를 찾는 방식입니다.

하지만 현재 **13개 모두 `LINE_DETECTION_FAILED`**였습니다.

이는 단순 threshold 문제가 아닐 가능성이 높습니다.

---

# 7. 기존 Line Detection의 근본적인 문제

실제 영상에서 하나의 Wafer가 반드시 하나의 Peak를 만드는 것은 아닙니다.

얇더라도 물체에는 위/아래 경계가 있기 때문에

```text
────────── Upper edge
   Wafer
────────── Lower edge
```

두 개의 Sobel peak가 발생할 수 있습니다.

Blade 역시

```text
══════════ Top edge

══════════ Bottom edge
```

처럼 여러 peak를 만들 수 있습니다.

따라서 지금처럼

```text
Peak 1 = Upper Wafer
Peak 2 = Blade
Peak 3 = Lower Wafer
```

로 바로 대응시키는 방식은 취약합니다.

---

# 8. 앞으로의 Line Detection 방향

Peak가 아니라 **수평 Object 단위로 검출**하는 것이 좋습니다.

영상 구조상 Wafer와 Blade에는 좋은 구분 특징이 있습니다.

### Wafer

```text
────────────────────────────
←            길다            →
```

- 매우 얇음
- 수평
- 좌우 길이가 김

### Blade

```text
          ═════════════
          ←   짧음   →
```

- Wafer보다 조금 두꺼움
- 수평
- Wafer보다 폭이 짧음
- Y 위치가 거의 고정

따라서 향후에는

```text
Sobel-Y
 ↓
Edge binary
 ↓
Horizontal morphology
 ↓
Connected Component / Contour
 ↓
수평 Object 추출
 ↓
width / thickness / Y 분석
```

방식이 더 적합합니다.

즉 **단순 1D peak detection에서 2D geometry detection으로 변경**하는 방향입니다.

---

# 9. 우측 빛번짐 문제

실제 CCTV 영상 우측에 강한 빛번짐이 있기 때문에 현재는 사용자가 지정한 ROI의 **좌측 50%만 모든 판정에 사용**하기로 했습니다.

기존 ROI가

```text
x = 7
y = 346
w = 2156
h = 277
```

이라면 실제 분석 영역은

```text
x = 7
y = 346
w = 1078
h = 277
```

입니다.

이 영역만

```text
Optical Flow
CST motion
Sobel
Wafer detection
Blade detection
Gap measurement
```

에 사용합니다.

이 선택은 또 하나의 장점이 있습니다. 분석 이미지를 width=640으로 resize할 경우 세로 해상도를 더 확보할 수 있어 **Wafer 사이의 작은 pixel 간격 측정에 유리**합니다.

---

# 10. 최종 시스템 구조

전체 POC가 성공하면 최종적으로는 다음 구조가 됩니다.

```text
               CCTV
                 │
                 ▼
        ┌─────────────────┐
        │ Motion Detector │
        └────────┬────────┘
                 │
          Type 2 candidate
                 ▼
       ┌────────────────────┐
       │ Sequence Analyzer  │
       │ Blade IN           │
       │ CST UP             │
       │ CST STOP           │
       │ Blade OUT          │
       └─────────┬──────────┘
                 │
         Measurement Frame
                 ▼
       ┌────────────────────┐
       │ Horizontal Object  │
       │ Detection          │
       └─────────┬──────────┘
                 │
       ┌─────────┼──────────┐
       ▼         ▼          ▼
 Upper Wafer   Blade    Lower Wafer
       │         │          │
       └─────────┼──────────┘
                 ▼
           Gap Measurement
                 │
         ┌───────┴────────┐
         ▼                ▼
      Top Gap         Bottom Gap
         │                │
         └───────┬────────┘
                 ▼
             Gap Ratio
                 │
                 ▼
           OK / Abnormal
```

---

# 11. 개발 순서를 다시 잡으면

지금부터는 **한 번에 전체를 고치지 않는 것**이 중요합니다.

**Phase A — Timing POC:** 13개 Type 2 clip에서 measurement frame을 먼저 13/13에 가깝게 맞춥니다. 이 단계에서는 line detection을 아예 평가하지 않습니다.

**Phase B — Geometry POC:** Phase A에서 사람이 확인한 정상 measurement frame만 대상으로 Upper Wafer / Blade / Lower Wafer 검출을 개발합니다. 여기서는 timing detector를 신경 쓰지 않습니다.

**Phase C — Gap Measurement:** 세 Object의 Y center를 구하고 `top_gap`, `bottom_gap`, `ratio`를 계산합니다. 정상 데이터 분포를 확보한 뒤 OK/NG 기준을 결정합니다.

**Phase D — End-to-End:** `Type2 clip → measurement frame → object detection → gap ratio`를 연결합니다.

**Phase E — Long Video:** 마지막에 원본 긴 CCTV 영상으로 돌아가 현재 Type 2 13건 검출 Recall을 개선합니다.

이 순서가 지금까지의 시행착오를 가장 많이 줄일 수 있습니다.

특히 다음 작업에서는 **v3.1을 계속 복잡하게 수정하기보다 Timing 전용 스크립트와 Line/Gap 전용 스크립트를 아예 분리하는 것**을 권합니다. 그래야 지금처럼 `LINE_DETECTION_FAILED` 때문에 timing 검증까지 막히는 일이 다시 생기지 않습니다.
