def apply_water_effect(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    darkness: float,
    blue_tint: float,
    rng: np.random.Generator,
) -> np.ndarray:

    image_float = image.astype(
        np.float32
    )

    mask = _clip_mask(
        mask
    )

    alpha = mask[
        ...,
        None,
    ]

    # --------------------------------------------------------
    # 1. Wet appearance
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        image_float,
        (0, 0),
        sigmaX=1.3,
        sigmaY=1.3,
    )

    wet = (
        image_float * 0.70
        + blurred * 0.30
    )

    wet *= (
        1.0
        - darkness
    )

    # --------------------------------------------------------
    # 2. Blue tint
    #
    # OpenCV = BGR
    #
    # 눈에 잘 띄는 파란/청록 계열을 의도적으로 사용
    # --------------------------------------------------------

    blue_color = np.array(
        [
            255.0,   # B
            145.0,   # G
            70.0,    # R
        ],
        dtype=np.float32,
    )

    wet = (
        wet
        * (
            1.0
            - blue_tint
        )
        + blue_color
        * blue_tint
    )

    # --------------------------------------------------------
    # 3. Composite
    # --------------------------------------------------------

    output = (
        image_float
        * (
            1.0
            - alpha
        )
        + wet
        * alpha
    )

    # --------------------------------------------------------
    # 4. Water edge highlight
    # --------------------------------------------------------

    mask_u8 = (
        mask
        * 255.0
    ).astype(
        np.uint8
    )

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    edge = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_GRADIENT,
        kernel,
    ).astype(
        np.float32
    ) / 255.0

    # 밝은 청색 highlight
    edge_color = np.array(
        [
            90.0,
            45.0,
            15.0,
        ],
        dtype=np.float32,
    )

    edge_strength = float(
        rng.uniform(
            0.5,
            1.0,
        )
    )

    output += (
        edge[
            ...,
            None,
        ]
        * edge_color
        * edge_strength
    )

    # --------------------------------------------------------
    # 5. Soft highlight
    # --------------------------------------------------------

    highlight = cv2.GaussianBlur(
        mask,
        (0, 0),
        sigmaX=2.0,
        sigmaY=2.0,
    )

    highlight_strength = float(
        rng.uniform(
            4.0,
            12.0,
        )
    )

    # Blue 쪽 highlight를 조금 더 강하게
    highlight_color = np.array(
        [
            1.5,
            1.0,
            0.6,
        ],
        dtype=np.float32,
    )

    output += (
        highlight[
            ...,
            None,
        ]
        * highlight_strength
        * highlight_color
    )

    return np.clip(
        output,
        0,
        255,
    ).astype(
        np.uint8
    )
