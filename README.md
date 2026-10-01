def make_overlay(
    image_rgb: np.ndarray,
    anomaly_map: np.ndarray,
    gt_mask: np.ndarray,
    *,
    defect_type: str,
    image_score: float,
    metrics: dict[str, Any],
) -> np.ndarray:

    # ========================================================
    # Display size
    # ========================================================

    original_height, original_width = (
        image_rgb.shape[:2]
    )

    # 각 panel이 최소 800px width가 되도록 확대
    # 동시에 최소 3배 확대
    display_scale = max(
        3.0,
        800.0 / original_width,
    )

    display_width = int(
        round(
            original_width
            * display_scale
        )
    )

    display_height = int(
        round(
            original_height
            * display_scale
        )
    )

    # ========================================================
    # Resize Original
    # ========================================================

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR,
    )

    image_display = cv2.resize(
        image_bgr,
        (
            display_width,
            display_height,
        ),
        interpolation=cv2.INTER_CUBIC,
    )

    # ========================================================
    # Resize anomaly map
    # ========================================================

    anomaly_display = cv2.resize(
        anomaly_map.astype(
            np.float32
        ),
        (
            display_width,
            display_height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    normalized = normalize_map(
        anomaly_display
    )

    heatmap = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_JET,
    )

    # ========================================================
    # Heatmap Overlay
    # ========================================================

    overlay = cv2.addWeighted(
        image_display,
        0.55,
        heatmap,
        0.45,
        0,
    )

    # ========================================================
    # Resize GT mask
    # ========================================================

    mask_u8 = (
        gt_mask.astype(
            np.uint8
        )
        * 255
    )

    mask_display = cv2.resize(
        mask_u8,
        (
            display_width,
            display_height,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    gt_display = (
        mask_display > 0
    )

    # ========================================================
    # GT contour
    # ========================================================

    contours, _ = cv2.findContours(
        mask_display,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (
            255,
            255,
            255,
        ),
        4,
        cv2.LINE_AA,
    )

    # ========================================================
    # Peak position
    # ========================================================

    peak_x = int(
        round(
            metrics[
                "peak_x"
            ]
            * display_scale
        )
    )

    peak_y = int(
        round(
            metrics[
                "peak_y"
            ]
            * display_scale
        )
    )

    cv2.drawMarker(
        overlay,
        (
            peak_x,
            peak_y,
        ),
        (
            0,
            255,
            255,
        ),
        markerType=(
            cv2.MARKER_CROSS
        ),
        markerSize=30,
        thickness=4,
        line_type=cv2.LINE_AA,
    )

    # Peak 주변 원도 추가
    cv2.circle(
        overlay,
        (
            peak_x,
            peak_y,
        ),
        16,
        (
            0,
            255,
            255,
        ),
        3,
        cv2.LINE_AA,
    )

    # ========================================================
    # GT Mask visualization
    # ========================================================

    mask_visual = cv2.cvtColor(
        mask_display,
        cv2.COLOR_GRAY2BGR,
    )

    # Mask contour를 한 번 더 표시
    cv2.drawContours(
        mask_visual,
        contours,
        -1,
        (
            0,
            255,
            255,
        ),
        4,
        cv2.LINE_AA,
    )

    # ========================================================
    # Individual image title bars
    # ========================================================

    title_height = 70

    def add_title(
        image: np.ndarray,
        title: str,
    ) -> np.ndarray:

        title_bar = np.zeros(
            (
                title_height,
                image.shape[1],
                3,
            ),
            dtype=np.uint8,
        )

        cv2.putText(
            title_bar,
            title,
            (
                20,
                47,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.05,
            (
                235,
                235,
                235,
            ),
            2,
            cv2.LINE_AA,
        )

        return np.vstack(
            [
                title_bar,
                image,
            ]
        )

    original_panel = add_title(
        image_display,
        "Synthetic NG",
    )

    anomaly_panel = add_title(
        overlay,
        "PatchCore Anomaly Map + GT",
    )

    mask_panel = add_title(
        mask_visual,
        "Synthetic GT Mask",
    )

    # ========================================================
    # Horizontal content
    # ========================================================

    content = np.hstack(
        [
            original_panel,
            anomaly_panel,
            mask_panel,
        ]
    )

    total_width = (
        content.shape[1]
    )

    # ========================================================
    # Large information panel
    # ========================================================

    panel_height = 260

    panel = np.zeros(
        (
            panel_height,
            total_width,
            3,
        ),
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # Main title
    # --------------------------------------------------------

    cv2.putText(
        panel,
        (
            f"{defect_type.upper()} "
            f"| PatchCore Localization Validation"
        ),
        (
            30,
            48,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.20,
        (
            255,
            255,
            255,
        ),
        3,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Row 1
    # --------------------------------------------------------

    line1 = (
        f"Image Score : "
        f"{image_score:.3f}"
        f"      "
        f"Peak Hit : "
        f"{metrics['peak_hit']}"
        f"      "
        f"Peak : "
        f"({metrics['peak_x']}, "
        f"{metrics['peak_y']})"
    )

    cv2.putText(
        panel,
        line1,
        (
            30,
            100,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.90,
        (
            220,
            220,
            220,
        ),
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Row 2
    # --------------------------------------------------------

    line2 = (
        f"Mean Inside : "
        f"{metrics['mean_inside']:.3f}"
        f"      "
        f"Mean Outside : "
        f"{metrics['mean_outside']:.3f}"
        f"      "
        f"Inside / Outside : "
        f"{metrics['mean_inside_outside_ratio']:.3f}"
    )

    cv2.putText(
        panel,
        line2,
        (
            30,
            145,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (
            220,
            220,
            220,
        ),
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Row 3
    # --------------------------------------------------------

    line3 = (
        f"Top 1%  P/R : "
        f"{metrics['top1_precision']:.3f}"
        f" / "
        f"{metrics['top1_recall']:.3f}"
        f"      "
        f"Top 5%  P/R : "
        f"{metrics['top5_precision']:.3f}"
        f" / "
        f"{metrics['top5_recall']:.3f}"
    )

    cv2.putText(
        panel,
        line3,
        (
            30,
            190,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (
            220,
            220,
            220,
        ),
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Row 4
    # --------------------------------------------------------

    line4 = (
        f"Top 10% P/R : "
        f"{metrics['top10_precision']:.3f}"
        f" / "
        f"{metrics['top10_recall']:.3f}"
        f"      "
        f"Mask Pixel Ratio : "
        f"{metrics['mask_pixel_ratio']:.4f}"
        f"      "
        f"Max In/Out Ratio : "
        f"{metrics['max_inside_outside_ratio']:.3f}"
    )

    cv2.putText(
        panel,
        line4,
        (
            30,
            235,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (
            220,
            220,
            220,
        ),
        2,
        cv2.LINE_AA,
    )

    # ========================================================
    # Final image
    # ========================================================

    result = np.vstack(
        [
            panel,
            content,
        ]
    )

    return result
