# Jetson 라인 재방문 테스트 가이드
## NVR HDMI → Capture Card → Jetson 검은 화면 원인 분리

> 목적  
> 라인 NVR 화면을 Jetson으로 수집할 때 발생한 **간헐적 Black 화면**의 원인을  
> `기존 HDMI 케이블 / 새 HDMI 케이블 / NVR 출력 / Capture Card / Jetson 수집 경로`로 단계적으로 분리한다.
>
> 현재까지 확인된 사항
>
> - Jetson USB Capture Card는 USB 3.x `5000M` 연결 시 정상 인식됨
> - Capture Card는 `MJPG 1920x1080 @ 60 FPS` 지원
> - 사무실 PC → 다른 HDMI 케이블 → 동일 Capture Card → 동일 Jetson 테스트에서는 Black 화면이 보이지 않았음
> - 라인 테스트 영상의 Black 구간은 평균 밝기 약 `14`
> - 라인 영상 분석에서 `Read failures = 0`인데 Black frame이 연속 발생했으므로,
>   Jetson이 프레임을 못 읽었다기보다 Capture Card가 **검은 프레임 자체를 전달했을 가능성**이 높음
> - 라인 NVR 출력 설정은 `1920x1080 / 60Hz`로 확인됨
> - 무선 키보드/마우스는 Capture Card 연결 시 간섭으로 먹통이 될 수 있으므로 주의

---

# 0. 오늘 테스트 목표

```text
1) 기존 라인 HDMI 케이블 그대로 테스트
2) Black 발생 여부 확인
3) NVR은 그대로 두고 HDMI 케이블만 새 케이블로 교체
4) 동일 조건으로 다시 테스트
5) 새 케이블에서도 Black 발생 시
   → NVR ↔ Capture Card HDMI 호환성 / Sync 문제 집중 확인
6) 필요 시 1080p60 → 1080p30으로 변경하여 추가 비교
7) 필요 시 OpenCV 저장과 FFmpeg Raw MJPEG 저장을 비교
```

---

# 1. 현장 도착 직후 기본 확인

## 1-1. 프로젝트 이동

```bash
cd ~/projects/cctv-anomaly
source .venv/bin/activate
```

```bash
which python
python --version
```

예상:

```text
/home/jetson/projects/cctv-anomaly/.venv/bin/python
Python 3.12.x
```

## 1-2. 저장 공간 확인

```bash
df -h /
```

권장:

```text
Available >= 50 GB
```

현재 Capture 데이터 용량:

```bash
du -sh ~/projects/cctv-anomaly/data/capture
```

최근 세션:

```bash
ls -lht ~/projects/cctv-anomaly/data/capture | head
```

---

# 2. Capture Card USB 연결 확인

Capture Card는 반드시 이전에 정상 확인한 **USB 3.x 포트**에 연결한다.

```bash
lsusb -t
```

정상 예:

```text
Bus 002 ... 10000M
└─ Hub ... 10000M
   └─ Video ... 5000M
```

판정:

```text
Capture Card = 5000M 이상 → 정상
Capture Card = 480M       → 테스트 중단 후 USB 포트 변경
```

480M이면:

```text
1. Capture Card USB 분리
2. 다른 USB 포트 연결
3. lsusb -t 재확인
```

---

# 3. Capture Device 확인

```bash
v4l2-ctl --list-devices
```

예상:

```text
USB3 Video: USB3 Video
    /dev/video0
    /dev/video1
```

기본 테스트 장치:

```text
/dev/video0
```

장치 번호가 달라졌다면 이후 명령의 `/dev/video0`, `--device 0`을 실제 번호로 변경한다.

---

# 4. 지원 포맷 확인

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
```

현재 확인된 지원 조건:

```text
MJPG
1920x1080
60 FPS
50 FPS
30 FPS
20 FPS
10 FPS
```

---

# 5. 현재 Capture 설정 확인

```bash
v4l2-ctl -d /dev/video0 --get-fmt-video
v4l2-ctl -d /dev/video0 --get-parm
```

목표:

```text
Width/Height      : 1920/1080
Pixel Format      : 'MJPG'
Frames per second : 60.000
```

---

# 6. NVR 출력 설정 확인

NVR 설정 화면에서 확인:

```text
Resolution   : 1920 x 1080
Refresh Rate : 60 Hz
```

기록:

```text
NVR 모델명               : ______________________
NVR HDMI 출력 해상도      : ______________________
NVR HDMI 출력 주사율      : ______________________
```

> 주의  
> NVR 메뉴에 `60 FPS`라고 표시되더라도 HDMI 출력 의미는 실제로 `60Hz`인 경우가 있다.  
> 오늘 테스트에서는 Capture Card V4L2 수집을 60 FPS로 맞춰 비교한다.

---

# 7. Capture Card를 1920x1080 / MJPG / 60 FPS로 강제 설정

```bash
v4l2-ctl -d /dev/video0 \
    --set-fmt-video=width=1920,height=1080,pixelformat=MJPG \
    --set-parm=60
```

정상 메시지:

```text
Frame rate set to 60.000 fps
```

재확인:

```bash
v4l2-ctl -d /dev/video0 --get-fmt-video
v4l2-ctl -d /dev/video0 --get-parm
```

---

# 8. TEST A — 기존 라인 HDMI 케이블

## 연결

```text
NVR HDMI OUT
    ↓
기존 라인 HDMI 케이블
    ↓
Capture Card HDMI IN
    ↓
USB 3.x
    ↓
Jetson
```

## 8-1. Preview로 20~30초 확인

```bash
python src/capture_usb_video.py \
    --device 0 \
    --width 1920 \
    --height 1080 \
    --fps 60 \
    --input-fourcc MJPG \
    --preview
```

확인:

- [ ] 화면 정상
- [ ] Black 순간 발생 여부
- [ ] 화면 깨짐 여부
- [ ] 키보드/마우스 정상 여부

종료:

```text
q
ESC
Ctrl+C
```

## 8-2. 본 테스트 3~5분

```bash
python src/capture_usb_video.py \
    --device 0 \
    --width 1920 \
    --height 1080 \
    --fps 60 \
    --input-fourcc MJPG \
    --no-preview
```

종료:

```text
q
또는
Ctrl+C
```

기록:

```text
Requested       : __________________
Actual          : __________________
Captured frames : __________________
Saved JPGs      : __________________
Read failures   : __________________
Elapsed         : __________________
Capture FPS     : __________________
```

정상 기대값:

```text
Actual        ≈ 1920x1080 @ 60 FPS
Capture FPS   ≈ 60
Read failures = 0
```

---

# 9. TEST A 영상 Black 분석

최근 세션:

```bash
ls -lht ~/projects/cctv-anomaly/data/capture | head
```

최근 MP4:

```bash
find ~/projects/cctv-anomaly/data/capture -name "*.mp4" \
    -printf "%T@ %p\n" | sort -nr | head
```

Black 분석:

```bash
python src/analyze_black_screen.py \
    data/capture/<세션폴더>/<영상파일>.mp4 \
    --mean-threshold 20 \
    --pixel-threshold 25 \
    --dark-ratio-threshold 0.99 \
    --min-duration 0
```

현재 라인에서 확인된 Black level이 약 `14`이므로 위 설정을 우선 사용한다.

결과 확인:

```bash
cat data/capture/<세션폴더>/<영상파일명>_black_analysis/summary.txt
```

```bash
cat data/capture/<세션폴더>/<영상파일명>_black_analysis/black_events.csv
```

기록:

```text
TEST A
Black events         : __________________
Total black duration : __________________
Read failures        : __________________
Capture FPS          : __________________
```

---

# 10. TEST A에서 Black 발생 시 즉시 로그 확인

## 최근 USB / UVC 로그

```bash
sudo dmesg -T | grep -Ei \
"usb|uvc|xhci|reset|disconnect|over-current|error" | tail -n 100
```

특히 아래를 찾는다.

```text
USB disconnect
reset SuperSpeed USB device
uvcvideo ...
xhci ...
over-current
```

## 실시간 USB 에러 감시

```bash
sudo dmesg -w | grep -Ei \
"usb|uvc|xhci|reset|disconnect|error"
```

Black 화면 발생 시 같은 시간대에 USB reset/disconnect가 있는지 확인.

## Jetson 상태

```bash
tegrastats
```

중지:

```text
Ctrl+C
```

---

# 11. TEST B — HDMI 케이블만 새 케이블로 교체

다른 조건은 바꾸지 않는다.

```text
변경 전
NVR
 ↓
기존 HDMI
 ↓
Capture Card
 ↓
Jetson

변경 후
NVR
 ↓
새 HDMI       ← 이것만 변경
 ↓
Capture Card
 ↓
Jetson
```

동일 유지 조건:

```text
NVR          : 동일
해상도        : 1920x1080
NVR 출력      : 60Hz
Capture       : MJPG
Capture FPS   : 60
USB 포트      : 동일 5000M 포트
Capture Card  : 동일
Jetson        : 동일
```

## 11-1. 설정 재확인

```bash
v4l2-ctl -d /dev/video0 \
    --set-fmt-video=width=1920,height=1080,pixelformat=MJPG \
    --set-parm=60
```

```bash
v4l2-ctl -d /dev/video0 --get-fmt-video
v4l2-ctl -d /dev/video0 --get-parm
```

## 11-2. 동일하게 3~5분 수집

```bash
python src/capture_usb_video.py \
    --device 0 \
    --width 1920 \
    --height 1080 \
    --fps 60 \
    --input-fourcc MJPG \
    --no-preview
```

## 11-3. Black 분석

```bash
python src/analyze_black_screen.py \
    data/capture/<TEST_B_세션>/<TEST_B_영상>.mp4 \
    --mean-threshold 20 \
    --pixel-threshold 25 \
    --dark-ratio-threshold 0.99 \
    --min-duration 0
```

기록:

```text
TEST B
Black events         : __________________
Total black duration : __________________
Read failures        : __________________
Capture FPS          : __________________
```

---

# 12. TEST A / B 결과 해석

## Case 1

```text
기존 HDMI : Black 발생
새 HDMI   : Black 없음
```

판단:

```text
→ 기존 라인 HDMI 케이블 문제 가능성 매우 높음
```

## Case 2

```text
기존 HDMI : Black 발생
새 HDMI   : Black 발생
```

판단:

```text
→ HDMI 케이블 단독 문제 가능성 낮음
→ NVR ↔ Capture Card HDMI Sync / 호환성 의심
```

다음:

```text
TEST C : 1080p30
TEST D : FFmpeg Raw MJPEG
가능하면 NVR → Monitor 직접 연결
```

## Case 3

```text
기존 HDMI : Black 없음
새 HDMI   : Black 없음
```

판단:

```text
→ 이전 현장 발생이 일시적이었을 가능성
→ 장시간 10~30분 테스트 필요
```

## Case 4

```text
기존 HDMI : Black 없음
새 HDMI   : Black 발생
```

판단:

```text
→ 새 HDMI 케이블 품질/접촉 문제 가능성
→ 기존 케이블 재시험
```

---

# 13. TEST C — 1920x1080 / 30 FPS 비교

새 HDMI에서도 60 FPS에서 Black이 발생할 때 수행.

NVR에서 가능하면 출력도 1080p30 또는 1080p/30Hz로 변경.

Capture Card 설정:

```bash
v4l2-ctl -d /dev/video0 \
    --set-fmt-video=width=1920,height=1080,pixelformat=MJPG \
    --set-parm=30
```

확인:

```bash
v4l2-ctl -d /dev/video0 --get-fmt-video
v4l2-ctl -d /dev/video0 --get-parm
```

수집:

```bash
python src/capture_usb_video.py \
    --device 0 \
    --width 1920 \
    --height 1080 \
    --fps 30 \
    --input-fourcc MJPG \
    --no-preview
```

Black 분석:

```bash
python src/analyze_black_screen.py \
    data/capture/<세션>/<영상>.mp4 \
    --mean-threshold 20 \
    --pixel-threshold 25 \
    --dark-ratio-threshold 0.99 \
    --min-duration 0
```

결과:

```text
60 FPS Black : __________________
30 FPS Black : __________________
```

해석:

```text
60 FPS에서만 Black
30 FPS에서는 정상
→ Capture Card / NVR HDMI 60Hz sync 안정성 의심
```

---

# 14. TEST D — FFmpeg Raw MJPEG 저장

목적:

```text
OpenCV decode / VideoWriter 영향을 제거하고
Capture Card가 전달하는 MJPEG 자체를 저장
```

폴더:

```bash
mkdir -p ~/projects/cctv-anomaly/data/capture/ffmpeg_test
```

## 60 FPS 5분

```bash
ffmpeg \
    -f v4l2 \
    -input_format mjpeg \
    -video_size 1920x1080 \
    -framerate 60 \
    -i /dev/video0 \
    -t 300 \
    -c:v copy \
    ~/projects/cctv-anomaly/data/capture/ffmpeg_test/raw_1080p60.mkv
```

## 30 FPS 5분

```bash
ffmpeg \
    -f v4l2 \
    -input_format mjpeg \
    -video_size 1920x1080 \
    -framerate 30 \
    -i /dev/video0 \
    -t 300 \
    -c:v copy \
    ~/projects/cctv-anomaly/data/capture/ffmpeg_test/raw_1080p30.mkv
```

영상 정보:

```bash
ffprobe ~/projects/cctv-anomaly/data/capture/ffmpeg_test/raw_1080p60.mkv
```

---

# 15. OpenCV vs FFmpeg 결과 해석

```text
OpenCV Black 있음
FFmpeg Black 있음
→ OpenCV/VideoWriter 문제가 아님
→ Capture Card가 이미 Black frame을 출력하고 있을 가능성 높음
→ NVR ↔ Capture Card HDMI 신호 문제 집중
```

```text
OpenCV Black 있음
FFmpeg Black 없음
→ OpenCV decode / 저장 경로 의심
```

```text
OpenCV Black 없음
FFmpeg Black 없음
→ 해당 조건에서는 정상
```

---

# 16. 가능하면 수행 — NVR HDMI OUT 직접 확인

가능한 경우:

```text
NVR HDMI OUT
    ↓
HDMI Cable
    ↓
Monitor
```

5~10분 관찰.

### Monitor에서도 Black 발생

```text
→ NVR HDMI 출력 자체 또는 HDMI 케이블 문제 가능
```

### Monitor에서는 정상, Capture Card에서만 Black

```text
→ NVR ↔ Capture Card 호환성 / HDMI Sync 문제 가능성 증가
```

---

# 17. HDMI 접촉 문제 확인

- [ ] NVR HDMI OUT 단자 흔들림 없음
- [ ] Capture Card HDMI IN 단자 흔들림 없음
- [ ] HDMI 케이블 완전히 삽입
- [ ] HDMI 연장 젠더 사용하지 않음
- [ ] HDMI 분배기 사용 여부 확인
- [ ] HDMI Splitter / Converter 사용 시 제거 테스트
- [ ] 케이블이 장비 전원선 또는 고전류 케이블에 강하게 눌려있지 않음

---

# 18. Capture Card USB 재연결 후

USB를 뺐다 다시 꽂았으면:

```bash
lsusb -t
```

반드시:

```text
5000M
```

확인.

그 다음:

```bash
v4l2-ctl --list-devices
```

장치 번호 재확인.

그리고:

```bash
v4l2-ctl -d /dev/video0 \
    --set-fmt-video=width=1920,height=1080,pixelformat=MJPG \
    --set-parm=60
```

---

# 19. 키보드/마우스 먹통 발생 시

현재 무선 키보드/마우스 사용 중이며 Capture Card 연결 시 먹통 경험 있음.

즉시 대응:

```text
1. Capture Card USB 분리
2. 키보드/마우스 복구 확인
3. Capture Card를 다시 USB 3.x 포트에 연결
4. lsusb -t 확인
```

가능하면:

```text
USB 3.x
└─ Capture Card

USB 2.0 / 연장 케이블
└─ 키보드/마우스 리시버
```

A/B 테스트 중에는 USB 포트와 장비 배치를 가능한 동일하게 유지한다.

---

# 20. Capture 중 프로그램이 멈춘 것 같을 때

```bash
ps -ef | grep capture_usb_video.py
```

```bash
top
```

```bash
tegrastats
```

```bash
df -h /
```

정상 종료 요청:

```bash
pkill -INT -f capture_usb_video.py
```

그래도 종료되지 않으면:

```bash
pkill -TERM -f capture_usb_video.py
```

`kill -9`는 MP4 손상 가능성이 있으므로 최후 수단.

---

# 21. 에러 확인 명령어 모음

## USB topology

```bash
lsusb -t
```

## USB 장치

```bash
lsusb
```

## Video 장치

```bash
v4l2-ctl --list-devices
```

## 현재 포맷

```bash
v4l2-ctl -d /dev/video0 --get-fmt-video
```

## 현재 FPS

```bash
v4l2-ctl -d /dev/video0 --get-parm
```

## 지원 포맷/FPS

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
```

## 최근 USB/UVC 오류

```bash
sudo dmesg -T | grep -Ei \
"usb|uvc|xhci|reset|disconnect|over-current|error" | tail -n 100
```

## 실시간 USB/UVC 오류

```bash
sudo dmesg -w | grep -Ei \
"usb|uvc|xhci|reset|disconnect|error"
```

## 저장공간

```bash
df -h /
```

## 온도/부하

```bash
tegrastats
```

---

# 22. MP4 정상 여부 확인

최근 MP4:

```bash
find ~/projects/cctv-anomaly/data/capture \
    -name "*.mp4" \
    -printf "%T@ %p\n" | sort -nr | head
```

파일 정보:

```bash
ffprobe <영상경로>
```

확인:

```text
Resolution
FPS
Duration
Codec
```

---

# 23. 테스트 결과 기록표

## TEST A — 기존 HDMI / 1080p60

```text
시간                  :
HDMI 케이블           : 기존
NVR 출력              : 1920x1080 / 60Hz
Capture               : MJPG / 1920x1080 / 60fps
USB                   : 5000M / 480M
Capture FPS           :
Read failures         :
Black events          :
Black duration        :
dmesg error           :
특이사항              :
```

## TEST B — 새 HDMI / 1080p60

```text
시간                  :
HDMI 케이블           : 새 케이블
NVR 출력              : 1920x1080 / 60Hz
Capture               : MJPG / 1920x1080 / 60fps
USB                   : 5000M / 480M
Capture FPS           :
Read failures         :
Black events          :
Black duration        :
dmesg error           :
특이사항              :
```

## TEST C — 새 HDMI / 1080p30

```text
시간                  :
HDMI 케이블           :
NVR 출력              :
Capture               : MJPG / 1920x1080 / 30fps
USB                   :
Capture FPS           :
Read failures         :
Black events          :
Black duration        :
dmesg error           :
특이사항              :
```

## TEST D — FFmpeg Raw MJPEG

```text
시간                  :
HDMI 케이블           :
NVR 출력              :
Capture               :
FFmpeg                 : raw MJPEG copy
Black 발생 여부        :
특이사항              :
```

---

# 24. 최종 판단표

| 기존 HDMI | 새 HDMI | 30FPS | FFmpeg Raw | 판단 |
|---|---|---|---|---|
| Black | 정상 | - | - | 기존 HDMI 케이블 문제 가능성 높음 |
| Black | Black | 정상 | 정상 | 60Hz/60FPS sync 또는 OpenCV 조건 의심 |
| Black | Black | Black | Black | NVR↔Capture Card HDMI 호환성 가능성 매우 높음 |
| Black | Black | Black | 정상 | OpenCV decode/VideoWriter 경로 의심 |
| 정상 | 정상 | - | - | 일시적 문제 가능, 장시간 재확인 필요 |

---

# 25. 오늘 가장 중요한 판단 순서

```text
[1]
기존 HDMI + 1080p60
        ↓
Black?
 ├─ NO  → 10~30분 장시간 재확인
 └─ YES
      ↓
[2]
새 HDMI + 1080p60
        ↓
Black?
 ├─ NO  → 기존 HDMI 문제 가능성 높음
 └─ YES
      ↓
[3]
새 HDMI + 1080p30
        ↓
Black?
 ├─ NO  → 60Hz/60FPS Sync 문제 가능
 └─ YES
      ↓
[4]
FFmpeg Raw MJPEG
        ↓
Black?
 ├─ NO  → OpenCV 수집/저장 경로 의심
 └─ YES → NVR ↔ Capture Card HDMI 호환성 집중
```

---

# 26. 현장 철수 전 필수

- [ ] TEST A 결과 기록
- [ ] TEST B 결과 기록
- [ ] 필요 시 TEST C 수행
- [ ] 필요 시 TEST D 수행
- [ ] 생성 MP4 파일 존재 확인
- [ ] `ffprobe` 정상
- [ ] Black analysis 결과 저장
- [ ] `summary.txt` 확인
- [ ] `black_events.csv` 확인
- [ ] 테스트별 HDMI 케이블 구분
- [ ] 테스트별 FPS 기록
- [ ] 저장공간 확인
- [ ] Jetson 정상 종료

Jetson 종료:

```bash
sudo poweroff
```

완전히 종료된 후 전원 분리.

---

# 27. 빠른 복사용 핵심 명령어

## 1080p60 설정

```bash
v4l2-ctl -d /dev/video0 \
    --set-fmt-video=width=1920,height=1080,pixelformat=MJPG \
    --set-parm=60
```

## 상태 확인

```bash
lsusb -t
v4l2-ctl -d /dev/video0 --get-fmt-video
v4l2-ctl -d /dev/video0 --get-parm
```

## 수집

```bash
cd ~/projects/cctv-anomaly
source .venv/bin/activate

python src/capture_usb_video.py \
    --device 0 \
    --width 1920 \
    --height 1080 \
    --fps 60 \
    --input-fourcc MJPG \
    --no-preview
```

## Black 분석

```bash
python src/analyze_black_screen.py \
    data/capture/<세션>/<영상>.mp4 \
    --mean-threshold 20 \
    --pixel-threshold 25 \
    --dark-ratio-threshold 0.99 \
    --min-duration 0
```

## USB 에러

```bash
sudo dmesg -T | grep -Ei \
"usb|uvc|xhci|reset|disconnect|over-current|error" | tail -n 100
```

## 장치 부하

```bash
tegrastats
```

## 저장 공간

```bash
df -h /
```

## 종료

```text
q
ESC
Ctrl+C
```
