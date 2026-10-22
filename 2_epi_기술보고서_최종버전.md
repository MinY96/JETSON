# CCTV 기반 CST 동작 감지 및 Blade–Wafer 간격 측정: 머신비전 기법 이해

> **대상:** 머신비전 입문자 / 제조기술팀 내부 공유  
> **기준:** CST Type2 이벤트 검출, 측정 프레임 선정, Blade–Wafer Gap 측정 코드  
> **목표:** 각 OpenCV 기술의 역할과 실제 프로젝트 내 데이터 흐름을 이해하기  
> **유의:** 아래 임계값은 해당 CCTV·ROI에서 검증한 경험적 설정값이며 모든 설비에 보편적으로 적용되는 값은 아님.

## 먼저, 전체 흐름을 한눈에

```text
CCTV 영상 프레임(Ft)
    │
    ├─ [움직임 감지] CST ROI ── Phase Correlation ── cst_dy
    │
    └─ [움직임 감지] Blade2 ROI ─ Frame Difference ─ blade2_motion_ratio
                │
                ▼
       시계열 그룹화 및 Type2 이벤트 판정
                │
                ▼
       이벤트 구간(Clip) 확보
                │
                ▼
 [측정 시점 선정] Blade 변화 패턴(temporal_ratio의 Valley → Rise)
                │
                ▼
        Measurement Frame 한 장
                │
                ├─ Template Matching → Blade 기준선(Top/Bottom)
                └─ Sobel Y + 수평 지지도 → Wafer 상·하단 경계
                │
                ▼
    Top Gap / Bottom Gap / Diff / Ratio
                │
                ▼
     Overlay·Detail 이미지, 영상, CSV·JSON 저장
```

**왜 여러 기법을 쓰나?** 각 기법이 잘하는 일이 다르기 때문. Phase Correlation은 '어느 방향으로 얼마나 움직였는지', Frame Difference는 '얼마나 많은 픽셀이 변했는지', Template Matching은 'Blade가 어디 있는지', Sobel은 'Wafer의 수평 경계가 어디 있는지'를 담당한다.

---

## 1. OpenCV Phase Correlation — CST 수직 이동량 측정

### 쉬운 이해

종이에 그린 무늬가 위로 5픽셀 밀렸다고 상상하자. 픽셀 하나하나가 똑같은지 확인하기보다, **영상 전체 패턴이 어떤 방향으로 얼마만큼 평행 이동했는지** 찾는 방법이 Phase Correlation이다.

### 원리

- 두 이미지의 주파수 영역 정보를 비교해 상대적 평행 이동을 추정한다(푸리에 변환 기반 위상 상관).
- 출력: `(dx, dy)` 이동량, `response`(상관 응답). `dy`가 음수이면 영상 좌표계에서 위 방향 이동을 의미한다.
- **주의:** 이미지에 충분한 질감이 있어야 하며, 조명 변화·가림·회전·큰 형상 변화가 있으면 오차가 생길 수 있다. `response`가 높다고 항상 실제 이동이라는 보장은 없다.

### 실제 적용

1. CST ROI: `(2039, 110, 149, 654)`를 자른다.
2. 컬러 영상을 Grayscale로 변환한다.
3. Gaussian Blur로 미세한 픽셀 노이즈를 완화한다.
4. 연속 프레임 `t−1`과 `t`를 `cv2.phaseCorrelate()`로 비교한다.
5. `cst_dy` 시계열을 생성한다.

```python
prev = cv2.GaussianBlur(prev_cst_gray, (5, 5), 0).astype("float32")
curr = cv2.GaussianBlur(curr_cst_gray, (5, 5), 0).astype("float32")
(dx, dy), response = cv2.phaseCorrelate(prev, curr)
```

코드는 이해를 위한 최소 예시이다. 실제 프로젝트 함수에는 ROI 추출, Hanning Window 등이 포함된다.

### 값 해석 예시

| 측정값 | 해석 예시 |
|---|---|
| `dy ≈ 0` | CST의 상하 이동이 거의 없음 |
| `dy = −5.0` | 직전 프레임에 비해 위로 약 5픽셀 이동 |
| `dy = −11.0` | 더 큰 상향 이동 |

**Type2 분류 핵심:** 한 프레임의 `dy`가 아니라 **하나의 CST 움직임 이벤트에 포함된 `dy` 최솟값**을 사용한다. 큰 동작이 순간적으로 `−5`를 지나더라도 전체 피크가 `−11`이라면 Type2로 잘못 선택하지 않기 위해서다.

```text
1차 움직임 후보: cst_dy ≤ −2.0
CST 움직임 그룹 간격: 0.40초 이하
Type2 이벤트: −8.0 < min(cst_dy in event) ≤ −2.0
```

※ 여기의 '픽셀'은 ROI에서 산출되는 **프레임 간 상대 이동량**이며 실제 기구의 mm 이동 거리가 아니다.

---

## 2. Frame Difference — Blade 움직임 비율 추출

### 쉬운 이해

같은 장소를 찍은 사진 두 장을 겹쳐 **바뀐 부분에만 색을 칠한다**고 생각하면 된다. 변화한 픽셀이 많을수록 Blade 움직임의 신호가 커진다.

### 원리

두 프레임 각 픽셀 밝기의 절댓값 차이를 구한다.

\[
D_t(x,y)=|I_t(x,y)-I_{t-1}(x,y)|
\]

밝기 차이가 임계값 이상인 픽셀을 움직인 픽셀로 취급한다.

\[
M_t(x,y)=\begin{cases}1,&D_t(x,y)\ge12\\0,&D_t(x,y)<12\end{cases}
\]

\[
\text{Motion Ratio}=\frac{\text{변화 픽셀 개수}}{\text{ROI 픽셀 개수}}
\]

예: 100,000픽셀 중 2,000픽셀이 변화 → `motion_ratio = 0.020 = 2%`.

### 실제 적용 (이벤트 검출)

1. 최종 선정 Blade2 ROI `(309, 379, 1593, 236)`를 자른다.
2. 분석 속도를 위해 너비 480픽셀로 축소한다(종횡비 유지).
3. Grayscale + Gaussian Blur `(5,5)` 적용.
4. 이전 분석 프레임과 현재 프레임의 절댓값 차이(`cv2.absdiff`)를 계산.
5. 변화량 `≥12`인 픽셀의 비율을 `blade2_motion_ratio`로 기록.

```python
diff = cv2.absdiff(prev_blade_gray, curr_blade_gray)
motion_ratio = float((diff >= 12).mean())
```

**Blade IN/OUT 기준:** 주변 탐색 구간에 있는 비율 값의 75분위수와 `0.015` 중 큰 값을 사용한다.

\[
T_{blade}=\max(Q_{0.75}(\mathrm{motion\_ratio}),0.015)
\]

CST 시작 **이전 3초**에서 Blade IN, CST 종료 **이후 3초**에서 Blade OUT을 찾는다. 가까운 움직임은 최대 0.20초 간격으로 그룹화한다.

### 강점과 주의점

- 장점: 구현이 단순하고 연산량이 비교적 적다. Blade1·Blade3나 Optical Flow를 생략하면 속도 개선에 유리하다.
- 주의: **밝기 변화·반사·CCTV 흔들림**도 움직임으로 감지될 수 있다. '무언가 달라졌다'는 신호이지 'Blade만 움직였다'는 객체 분류 결과는 아니다.

---

## 3. Blade 변화 패턴 분석 — 측정 프레임은 왜 이 시점인가?

### 어려운 문제

Type2 구간을 찾았다고 그 구간의 아무 프레임이나 측정할 수 있는 것은 아니다. Blade와 Wafer의 상대 위치가 잘 보이면서 Blade가 아직 완전히 빠져나가지 않은 순간을 골라야 한다.

### 사용하는 두 가지 차분 신호

이벤트 감지의 `blade2_motion_ratio`와 달리 **측정 프레임 선정**에서는 클립의 Blade ROI `(309,359,1593,216)`를 별도로 분석한다.

- `change_ratio`: **CST 피크 직전**에 선정된 기준 프레임과 현재 프레임의 차이. '기준 상태에서 얼마나 달라졌나?'
- `temporal_ratio`: **바로 이전 프레임과 현재 프레임**의 차이. '지금 이 순간 얼마나 변화하고 있나?'
- 전처리: Blade ROI Grayscale·Blur → 픽셀 차이 임계(15) → 2×3 Morphological Opening(작은 점 노이즈 제거) → 테두리 일부 제외 → 변화 픽셀 비율.
- 보조 특징: `change_energy`, `temporal_energy`, `sharpness`(Laplacian 분산)도 계산하지만, 최신 프레임 선정의 핵심은 `temporal_ratio`의 Valley/Rise이다.

### Valley → Rise 원리

```text
temporal_ratio
    ↑
높음│ ●●
    │   ●●
    │     ●●                  ●●●●  ← Blade OUT 재상승(Rise)
낮음│        ●●___●___●     ●●
    │              ↑       ↑
    │           Valley   Rise 시작
    └──────────────────────────────────→ Frame
                           ↑
                 Measurement = Rise 직전
```

그림은 설명용 개념도이며 실제 측정값 그래프는 아니다.

1. CST 움직임 피크 프레임을 찾는다.
2. 피크 뒤의 탐색 범위(기본 **+3~+15 프레임**)에서 `temporal_ratio`가 작아지는 **Valley**를 찾는다.
3. 저점 이후 충분한 변화량이 **연속 2프레임** 나타나는 최초의 **Rise**를 찾는다.
4. Rise 직전(기본 1프레임 전)을 측정 프레임으로 선택한다.
5. 저점과 너무 먼 프레임을 선택하지 않도록 Valley 뒤 최대 3프레임 거리로 제한한다.

### 실제 기준값

| 파라미터 | 기본값 | 의미 |
|---|---:|---|
| Valley threshold | `0.06` | 저점으로 인정할 움직임 비율 상한 |
| Rise threshold | `0.075` | 재상승 판단의 기본 임계 |
| Rise delta | `0.055` | 저점에 더해 요구하는 증가량 |
| Rise 확인 길이 | 2프레임 | 순간 노이즈로 판단하지 않기 위한 연속성 |
| Rise 검색 범위 | Valley 이후 최대 8프레임 | 검색 범위 제한 |
| Measurement lead | 1프레임 | Rise보다 앞선 프레임 선택 |

Rise의 실제 판정 기준: `temporal_ratio ≥ max(0.075, Valley 값 + 0.055)`가 **2프레임 연속** 충족되는지 검사한다.

**왜 이 방식인가?** 변화량이 낮아지는 구간은 상대적으로 동작이 안정된 순간을 의미하고, 재상승은 Blade 이탈 움직임의 시작을 시사한다. 둘 사이를 측정 후보로 선택하는 것이다. 단, 모든 움직임 저점이 기구의 완전한 정지를 의미하는 것은 아니므로 실제 영상 검증이 필요하다.

**실패 상태 예:** `valley_not_found`, `rise_not_found`, `measurement_frame_missing`. 이때 무조건 임의 프레임을 측정하는 대신 실패로 기록하는 것이 안전하다.

---

## 4. Template Matching — 이미지 속 Blade 위치 찾기

### 쉬운 이해

찾고 싶은 부품의 작은 사진을 견본으로 갖고, 큰 사진 위를 한 칸씩 이동하면서 **어느 위치가 견본과 가장 비슷한지** 점수를 매기는 방식이다. 제품의 바코드나 로고 위치를 찾는 장면을 떠올리면 된다.

### 원리

- OpenCV 함수: `cv2.matchTemplate()`
- 본 프로젝트 비교 방식: `cv2.TM_CCOEFF_NORMED` — 평균 밝기를 뺀 패턴 간 정규화 상관 점수 사용.
- 가장 높은 점수와 위치를 `cv2.minMaxLoc()`로 선정한다.
- 점수가 클수록 템플릿 패턴과 유사하나, 높은 점수 자체가 '항상 정답'임을 의미하지는 않는다.

### 실제 적용

- 탐색 ROI: `(50, 300, 620, 250)` — 최근 사용자가 확정한 값.
- Reference: `blade_left_bright.png`, `blade_left_dark.png` 두 장.
- 비교 순서: **Bright+Gray → Dark+Gray → Bright+CLAHE → Dark+CLAHE**.
- 각각 기본 통과 점수 `≥0.65`. 단계별 순서로 통과한 첫 결과를 선택한다.
- 통과 실패 시 유효 후보 중 최대 점수를 검토하여 `review` 또는 `match_failed`로 구분할 수 있다.
- 필요할 때 예상 위치와 실제 위치의 차이도 검사한다(`expected_match_x/y`가 설정된 경우).

### CLAHE는 무엇인가?

Contrast Limited Adaptive Histogram Equalization. 이미지를 작은 영역으로 나누어 **국소적인 밝기 대비를 보정**한다. 조명이 어둡거나 밝기 차가 작아도 패턴을 구별하기 쉽게 하는 전처리다. 과도하게 적용하면 노이즈가 강조될 수 있다.

### 매칭 위치에서 Blade 기준선 얻기

Template Matching 결과 `(x,y,w,h)`와 템플릿별 비율·오프셋 설정을 조합하여 다음을 얻는다.

```text
Blade 상단 Y (blade_top_y)
Blade 하단 Y (blade_bottom_y)
기준 측정 X (measure_x)
```

**중요:** Template Matching은 Blade 기준 위치를 찾는 기술이다. 상·하단 Wafer의 경계까지 Template Matching으로 찾는 것은 아니다.

---

## 5. Sobel Edge Detection — Wafer 수평 경계 찾기

### 쉬운 이해

회색 바닥에서 검은 줄이 시작되면 밝기가 갑자기 바뀐다. **밝기가 급격하게 변하는 곳이 '경계(Edge)'**다. Sobel은 이런 변화를 숫자로 표현하는 필터다.

### 원리

- `Sobel X`: 좌우 방향 밝기 변화에 민감 → 주로 **세로선** 강조.
- `Sobel Y`: 위아래 방향 밝기 변화에 민감 → 주로 **가로선** 강조.
- Wafer가 화면에서 긴 수평 경계를 형성하므로 `Sobel Y`를 사용한다.

### 실제 적용

1. Measurement ROI `(50,315,620,235)`를 Grayscale·Gaussian Blur 처리한다.
2. `cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)`로 Y 방향 기울기 `Gy` 계산.
3. 각 행의 `abs(Gy)` 평균을 계산하고 평활화하여 **행별 에지 강도 프로파일** 생성.
4. 충분히 강한 국소 최대점(후보 수평선)을 찾는다.
5. 영상 가로 방향으로 경계가 이어지는지 **Horizontal Support**를 검사해 반사광이나 작은 노이즈를 걸러낸다.
6. Blade 상단 위, Blade 하단 아래에서 각각 **가장 가까운 유효 Wafer 경계**를 선택한다.

### 왜 Horizontal Support가 필요한가?

Sobel 점수가 큰 위치가 항상 Wafer는 아니다. 나사·조명 반사·테이프 등에서 강한 에지가 생길 수 있다. 따라서 측정 ROI를 여러 **가로 Strip**으로 분할하여, 특정 Y 위치에서 강한 경계가 여러 Strip에 걸쳐 나타나는지 검사한다. **강하면서 넓게 이어지는 가로선**을 우선 선정하는 것이다.

검증된 원본 코드의 주요 기본값:

| 구분 | 기본값 | 역할 |
|---|---:|---|
| `wafer_edge_threshold` | 10.0 | 후보 에지의 최소 강도 |
| `wafer_min_prominence` | 4.0 | 주변보다 뚜렷한 봉우리인지 |
| `wafer_support_gradient` | 15.0 | Strip에서 유효 에지로 셀 Gradient 기준 |
| `wafer_min_support` | 0.20 | 가로 방향 지지도 기준 |
| `wafer_min_span_ratio` | 0.35 | 관찰된 경계의 최소 가로 길이 비율 |
| `wafer_min_valid_strips` | 3 | 유효 Strip 최소 개수 |
| Gap 탐색 범위 | 5~100 px | Blade에서 Wafer까지 허용 거리 |

**제약:** Wafer가 없거나 가려져 있으면 경계 미검출 상태가 정상적일 수 있다. 예를 들어 첫 Slot에서 아래 Wafer가 없는 경우 하단 Gap을 임의로 계산하지 않는다.

---

## 6. 결과값 계산과 해석

```text
     Top Wafer 경계     Yw_top ──────────────────────
                          ↑ Top Gap
     Blade 상단         Yb_top ──────────────────────
                       [      BLADE      ]
     Blade 하단         Yb_bottom ───────────────────
                          ↓ Bottom Gap
     Bottom Wafer 경계  Yw_bottom ───────────────────
```

\[
\begin{aligned}
TopGap &= Y_{BladeTop}-Y_{TopWafer}\\
BottomGap &= Y_{BottomWafer}-Y_{BladeBottom}\\
Diff &= |TopGap-BottomGap|\\
Top\% &= 100\times \frac{TopGap}{TopGap+BottomGap}\\
Bottom\% &= 100-Top\%
\end{aligned}
\]

| 결과 | 의미 |
|---|---|
| Top Gap (px) | 상단 Wafer–Blade 상단 거리 |
| Bottom Gap (px) | Blade 하단–하단 Wafer 거리 |
| Diff (px) | 두 간격의 절댓값 차이 |
| Ratio | 두 간격의 상대 비율(예: `4:6`) |
| `status` | 위치 검출·측정 성공 상태(제품 OK/NG와 다름) |

**픽셀≠실제 mm:** 실제 물리 간격으로 환산하려면 카메라 보정, 설치 거리, 렌즈 왜곡, 픽셀 스케일 검증 등이 필요하다.

---

## 7. 기술별 역할 비교 — 발표용 한 장 요약

| 기법 | 한 문장 설명 | 입력 | 출력 | 프로젝트에서 답하는 질문 |
|---|---|---|---|---|
| Phase Correlation | 이미지 패턴의 **평행 이동량** 추정 | CST 전후 프레임 | `cst_dy` | CST가 위로 얼마나 움직였는가? |
| Frame Difference | 픽셀 **밝기 변화 면적** 계산 | Blade 전후 프레임 | `motion_ratio` | Blade 주변에 움직임이 있는가? |
| Valley/Rise Analysis | 움직임 비율의 **감소→재상승** 지점 탐색 | 클립별 `temporal_ratio` | 측정 프레임 번호 | 어느 순간의 이미지를 측정할까? |
| Template Matching | 견본과 가장 비슷한 이미지 위치 탐색 | 측정 이미지+Blade 견본 | Blade 상·하단 기준선 | Blade가 정확히 어디에 있는가? |
| Sobel Y + Support | **연속적인 수평 경계** 선택 | 측정 이미지 ROI | 상·하단 Wafer Y | Wafer 경계는 어디에 있는가? |

### 팀원에게 설명할 때 핵심 문장

> "이 시스템은 CCTV를 바로 OK/NG로 분류하는 AI가 아니라, 먼저 CST·Blade의 움직임을 숫자로 바꾸고, 측정하기 적절한 한 장을 고른 다음, Blade와 Wafer의 경계를 찾아 픽셀 간격을 계산하는 머신비전 알고리즘입니다. 측정한 간격을 바탕으로 한 OK/NG 기준은 후속 단계에서 정하게 됩니다."

### 팀 공유용 5분 발표 구성

1. **30초:** 분석 목적 — CST Type2 동작 때 Blade–Wafer 간격 자동 측정
2. **1분:** Phase Correlation(이동 거리) vs Frame Difference(변화 면적)
3. **1분:** Blade 움직임 그래프에서 Valley/Rise로 측정 순간 찾기
4. **1분:** Template Matching으로 Blade, Sobel Y로 Wafer 찾기
5. **1분:** Overlay/Detail 결과와 실제 영상 정합성 검증 포인트
6. **30초:** 향후 품질 Spec 기반 OK/NG 판정과 운영 속도 검증

### 공유 시 함께 보여주면 좋은 실제 자료

- `clip_overlay.mp4`: 검출 이벤트와 움직임 순서 설명
- `measurement_frame.jpg`: 측정 시점이 적절한지 확인
- `overlay.jpg`, `detail.jpg`: Blade/ Wafer 경계, Top/Bottom Gap 시각화
- `result.json`: 픽셀 간격·Diff·Ratio 설명
- `temporal_ratio` 그래프: Valley/Rise 선정 기준 설명 (별도 디버그 CSV로 생성 가능)

---

## 8. 한계와 검증 유의사항

- **카메라 위치·조명·ROI가 바뀌면 임계값 재검증이 필요**하다.
- 움직임 감지의 검출 성공과, 측정 프레임 및 Gap의 정합성은 각각 별도로 평가해야 한다.
- 실시간 운영에서 입력 FPS보다 분석 FPS가 낮으면 프레임 유실이 발생할 수 있어 처리량·지연시간을 계측해야 한다.
- `measurement_selection_failed`나 매칭 실패는 자동으로 정상/불량을 의미하지 않는다.
- 기존 POC의 이벤트 검출·육안 검증 결과가 실제 운용 전체 조건에서의 신뢰도를 보증하는 것은 아니다.

## 9. 소스 기준

기법 설명과 기본 파라미터는 프로젝트의 GitHub 원본 스크립트인 아래 파일의 관련 함수에서 확인했다.

- `phase_e1_01_extract_motion_signals_v2.py`: `prepare_cst`, `calculate_phase_shift`, `prepare_blade`, `calculate_frame_difference`
- `phase_e3_03_extract_clips.py`: `detect_cst_up_events`, `find_blade_in`, `find_blade_out`
- `phase_a1_select_measurement_frame_v2.py`: `masked_image_difference`, `calculate_blade_signal`
- `phase_a1_select_measurement_frame_v3.py`: `find_transition`
- `phase_b_template_matching_v5_1.py`: `select_blade_match`, `calculate_wafer_profile`, `detect_wafer_lines`, `find_nearest_wafer`

소스: https://github.com/MinY96/JETSON/tree/main/EPI/integrated
