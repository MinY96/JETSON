def apply_gate(
    df: pd.DataFrame,
    threshold: GateThreshold,
) -> np.ndarray:

    # 반드시 독립적인 writable NumPy array로 생성
    prediction = (
        df["match_status"]
        .eq("matched")
        .to_numpy(
            dtype=bool,
            copy=True,
        )
    )

    if threshold.min_good_matches is not None:
        values = (
            df["good_matches"]
            .to_numpy(
                dtype=float,
                copy=False,
            )
        )

        condition = (
            np.isfinite(values)
            & (
                values
                >= threshold.min_good_matches
            )
        )

        prediction = (
            prediction
            & condition
        )

    if threshold.min_inliers is not None:
        values = (
            df["inliers"]
            .to_numpy(
                dtype=float,
                copy=False,
            )
        )

        condition = (
            np.isfinite(values)
            & (
                values
                >= threshold.min_inliers
            )
        )

        prediction = (
            prediction
            & condition
        )

    if threshold.min_inlier_ratio is not None:
        values = (
            df["inlier_ratio"]
            .to_numpy(
                dtype=float,
                copy=False,
            )
        )

        condition = (
            np.isfinite(values)
            & (
                values
                >= threshold.min_inlier_ratio
            )
        )

        prediction = (
            prediction
            & condition
        )

    if threshold.min_area_ratio is not None:
        values = (
            df["area_ratio"]
            .to_numpy(
                dtype=float,
                copy=False,
            )
        )

        condition = (
            np.isfinite(values)
            & (
                values
                >= threshold.min_area_ratio
            )
        )

        prediction = (
            prediction
            & condition
        )

    if threshold.max_area_ratio is not None:
        values = (
            df["area_ratio"]
            .to_numpy(
                dtype=float,
                copy=False,
            )
        )

        condition = (
            np.isfinite(values)
            & (
                values
                <= threshold.max_area_ratio
            )
        )

        prediction = (
            prediction
            & condition
        )

    if threshold.max_abs_rotation is not None:
        values = np.abs(
            df["rotation_deg"]
            .to_numpy(
                dtype=float,
                copy=False,
            )
        )

        condition = (
            np.isfinite(values)
            & (
                values
                <= threshold.max_abs_rotation
            )
        )

        prediction = (
            prediction
            & condition
        )

    return prediction
