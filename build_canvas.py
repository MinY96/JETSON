def build_canvas(
    image: np.ndarray,
    row: dict[str, str],
    *,
    roi_name: str,
    current: int,
    total: int,
) -> np.ndarray:

    polygon = parse_polygon(
        row.get(
            "polygon_local",
            "",
        )
    )

    rendered = draw_polygon(
        image,
        polygon,
    )

    image_h, image_w = rendered.shape[:2]

    # 오른쪽 정보 패널
    panel_width = INFO_PANEL_WIDTH

    # 이미지 높이보다 정보 Panel에 필요한 높이가 더 클 수 있으므로
    # 최소 높이를 확보
    required_panel_height = 560

    canvas_height = max(
        image_h,
        required_panel_height,
    )

    canvas_width = (
        image_w
        + panel_width
    )

    canvas = np.zeros(
        (
            canvas_height,
            canvas_width,
            3,
        ),
        dtype=np.uint8,
    )

    # 약간 밝은 어두운 배경
    canvas[:] = (
        25,
        25,
        25,
    )

    # -----------------------------------------------------
    # ROI Image
    # -----------------------------------------------------

    image_y = (
        canvas_height
        - image_h
    ) // 2

    canvas[
        image_y:image_y + image_h,
        0:image_w,
    ] = rendered

    # 이미지 / panel 구분선
    cv2.line(
        canvas,
        (image_w, 0),
        (image_w, canvas_height),
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )

    # -----------------------------------------------------
    # Information Panel
    # -----------------------------------------------------

    panel_x = (
        image_w
        + PANEL_PADDING_X
    )

    y = PANEL_PADDING_Y

    manual_label = (
        row.get(
            "manual_label",
            "",
        )
        or "UNLABELED"
    )

    localization_label = (
        row.get(
            "localization_label",
            "",
        )
        or "-"
    )

    # Header
    put_text(
        canvas,
        (
            f"{roi_name.upper()} "
            f"[{current + 1}/{total}]"
        ),
        panel_x,
        y,
        scale=0.82,
        thickness=2,
        color=(255, 255, 255),
    )

    y += 45

    # ------------------------------------
    # Frame information
    # ------------------------------------

    put_text(
        canvas,
        "FRAME",
        panel_x,
        y,
        scale=0.55,
        color=(150, 200, 255),
    )

    y += LINE_HEIGHT

    info_lines = [
        (
            f"Video : "
            f"{row.get('video_name', '-')}"
        ),
        (
            f"Frame : "
            f"{row.get('frame_idx', '-')}"
        ),
        (
            f"Time  : "
            f"{format_float(row.get('timestamp_sec'), 2)} sec"
        ),
    ]

    for line in info_lines:
        put_text(
            canvas,
            line,
            panel_x,
            y,
        )

        y += LINE_HEIGHT

    y += 10

    # ------------------------------------
    # Matching
    # ------------------------------------

    put_text(
        canvas,
        "MATCHING",
        panel_x,
        y,
        scale=0.55,
        color=(150, 200, 255),
    )

    y += LINE_HEIGHT

    match_lines = [
        (
            f"Status        : "
            f"{row.get('match_status', '-')}"
        ),
        (
            f"Good matches  : "
            f"{row.get('good_matches') or '-'}"
        ),
        (
            f"Inliers       : "
            f"{row.get('inliers') or '-'}"
        ),
        (
            f"Inlier ratio  : "
            f"{format_float(row.get('inlier_ratio'))}"
        ),
        (
            f"Area ratio    : "
            f"{format_float(row.get('area_ratio'))}"
        ),
        (
            f"Rotation      : "
            f"{format_float(row.get('rotation_deg'), 2)} deg"
        ),
        (
            f"API latency   : "
            f"{format_float(row.get('api_duration_ms'), 1)} ms"
        ),
    ]

    for line in match_lines:
        put_text(
            canvas,
            line,
            panel_x,
            y,
        )

        y += LINE_HEIGHT

    y += 10

    # ------------------------------------
    # Label
    # ------------------------------------

    put_text(
        canvas,
        "LABEL",
        panel_x,
        y,
        scale=0.55,
        color=(150, 200, 255),
    )

    y += LINE_HEIGHT

    put_text(
        canvas,
        (
            f"Inspection : "
            f"{manual_label}"
        ),
        panel_x,
        y,
        scale=0.68,
        thickness=2,
        color=(
            120,
            255,
            120,
        )
        if manual_label == "visible"
        else (
            120,
            180,
            255,
        )
        if manual_label == "not_visible"
        else (
            220,
            220,
            220,
        ),
    )

    y += 32

    put_text(
        canvas,
        (
            f"Localization : "
            f"{localization_label}"
        ),
        panel_x,
        y,
    )

    y += 42

    # ------------------------------------
    # Key Help
    # ------------------------------------

    put_text(
        canvas,
        "KEYS",
        panel_x,
        y,
        scale=0.55,
        color=(150, 200, 255),
    )

    y += LINE_HEIGHT

    key_lines = [
        "1 : Visible",
        "2 : Not Visible",
        "3 : Ignore",
        "",
        "G : Localization Good",
        "B : Localization Bad",
        "U : Localization Unknown",
        "",
        "A / Left  : Previous",
        "D / Right : Next",
        "Space     : Next",
        "",
        "0 : Clear",
        "W : Save",
        "Q : Save & Quit",
    ]

    for line in key_lines:
        if line:
            put_text(
                canvas,
                line,
                panel_x,
                y,
                scale=0.56,
            )

        y += 24

    return canvas


def resize_for_screen(
    image: np.ndarray,
    max_width: int,
    max_height: int,
) -> np.ndarray:

    height, width = image.shape[:2]

    width_scale = (
        max_width / width
    )

    height_scale = (
        max_height / height
    )

    scale = min(
        width_scale,
        height_scale,
        1.0,
    )

    if scale >= 1.0:
        return image

    resized_width = max(
        1,
        int(
            round(
                width * scale
            )
        ),
    )

    resized_height = max(
        1,
        int(
            round(
                height * scale
            )
        ),
    )

    return cv2.resize(
        image,
        (
            resized_width,
            resized_height,
        ),
        interpolation=cv2.INTER_AREA,
    )


python scripts/03_label_gate_dataset.py \
    --roi left \
    --sample-per-status 150 \
    --max-width 1800 \
    --max-height 1000
