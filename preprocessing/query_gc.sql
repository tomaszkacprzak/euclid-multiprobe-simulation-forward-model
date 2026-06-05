SELECT
    a.object_id,
    right_ascension,
    declination,
    tile_index,
    det_quality_flag,
    flux_vis_unif,
    phz_mode_1,
    phz_median,
    phz_flags,
    lensmc.she_lensmc_weight,
    flux_detection_total,
    fluxerr_detection_total
FROM euclid_tr1_a_v1 AS a
LEFT JOIN euclid_tr1_b_v1_1 AS b
    ON a.object_id = b.object_id
LEFT JOIN euclid_tr1_c_lensmc AS lensmc
    ON b.object_id = lensmc.object_id
WHERE
    flux_vis_unif <= 575.44
    AND flux_vis_unif >= 1.44544
    AND phz_mode_1 >= 0.200
    AND phz_mode_1 < 1.0
    AND phz_flags = 0
    AND phz_classification IN (2, 3)
    AND spurious_flag = 0
    AND point_like_prob <= 0.4
    AND flux_h_unif >= 5.7544
    AND eff_cov_flag > 0
    -- AND eff_cov_map > 0.8
    AND vis_det = 1
    AND det_quality_flag IN (0, 1, 2, 3, 512, 513, 514, 515)
    AND declination < 0
