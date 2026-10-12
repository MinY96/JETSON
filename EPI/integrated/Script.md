해당 폴더에 있는 파일들은 phase_c 코드에서 참조하는 스크립트들이야.
1차로 너가 만든 phase_c_intergrated_pipeline.py 안에서 참조하는 스크립트들의 원본이야. (파일명과 코드가 일부 수정되어서 보여주는 거야)
그리고 해당 파일들은 실제 프로젝트에서는 scrips 라는 폴더 내에 존재해.
그래서 나는 다음과 같이 코드를 수정했어.
    # Existing script paths
    parser.add_argument(
        "--e1-script",
        default="scripts/phase_e1_01_extract_motion_signals_v2.py",
    )

    parser.add_argument(
        "--e3-script",
        default="scripts/phase_e3_03_extract_clips.py",
    )

    parser.add_argument(
        "--a2-script",
        default="scripts/phase_a1_select_measurement_frame_v2.py",
    )

    parser.add_argument(
        "--a3-script",
        default="scripts/phase_a1_select_measurement_frame_v3.py",
    )

    parser.add_argument(
        "--b-script",
        default="scripts/phase_b_template_matching_v5_1.py",
    )

이제 내가 하고 싶은 건 phase_c_intergrated_pipeline.py 처럼 스크립트를 실행시키는 게 아니라, 해당 스크립트들이 src에 모듈로 구현되어지고
main.py가 실행되면 영상 분석이 되도록 하는거야. 그리고 산출물은 output 폴더에 저장이 되게 하는거지. 인자를 추가해서 단계별 산출물 저장을 할 지, 최종 산출물만 저장이 되게 할 지 고르게 하고 싶어

현재 프로젝트 구조는 다음과 같아.
- .venv : python 패키지 가상환경
- data: videos 폴더 존재
- output: 산출물 폴더
- reference: matching 시 사용하는 이미지 저장 폴더, blade_left_bright.png, blade_left_dark.png 2장 존재
- scripts: 진행한 스크립트 파일들
- src: → 여기 폴더에 기능 단위 폴더를 만들어서 main.py에서 사용 시 불러오도록 구현해야 함. scripts 내 파일들에서 참조해야 함. phase 단위로 폴더를 나누어도 좋을 듯
- main.py: 최종 실행 파일
- requirements.txt, roi.txt(blade_roi, cst_roi 정보), 명령여모음.txt
