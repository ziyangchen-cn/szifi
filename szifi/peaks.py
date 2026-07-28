"""
szifi.peaks -- scale-aware peak detection / consolidation for matched-filter SNR maps.

This is the "SNR maps -> catalogue" step, written to fix the over-merging of close cluster pairs.
szifi's theta_500-scaled merge (cat.merge_detections, modes 'masking'/'masking_max') deletes a
genuine cluster that lies within max(fac_theta_500*theta_500, radius_arcmin) of a stronger
neighbour -- even when the two are clearly SEPARATE peaks in the SNR map.

Diagnosed on the SO Cluster-Finder-Challenge sims (lowmass / mss2 noise): a real pair 3.4' apart
(M500c 4.5e14 @ z=0.53 and 4.9e14 @ z=0.31), BOTH detected by szifi at q~10.7, was collapsed into
one by masking_max because the higher-q detection was fit at theta_500=5.4' (mask radius 5.4' > 3.4').

KEY IDEA: clusters BLEND at large theta_500 but RESOLVE at small theta_500. So localise / test peaks
on a RESOLVED-SCALE map
        peak_map = max over (theta_500 <= theta_cap)
and keep two nearby maxima as SEPARATE peaks only when the SADDLE between them on peak_map is deep
enough:  min(peak heights) - saddle >= saddle_depth.  A shallow dip -- a noise bump, or one cluster
seen at several scales -- is treated as a single peak, so noise does not split one cluster.

RECOMMENDED ENTRY POINT (drop-in replacement for cat.merge_detections, applied PER TILE):
        consolidate_saddle_catalogue(catalogue, q_tensor, theta_500_vec, wcs, res_arcmin, ...)
It KEEPS szifi's noise-controlled 3D-local-max candidates and only changes how they are
consolidated across scale (this is what controls the false-detection rate). Validated on the 9 CFC
"missing" tiles: recovers the genuinely-separable over-merge (the 3.4' pair) at ZERO purity cost
(false 9->10 over 10 tiles, P=0.993 unchanged), robust to theta_cap (2-12') and saddle_depth (2-3).
The remaining "missing" objects are genuine <=1.9' (near-beam) blends, not separable in the SNR maps.

NOTE: detecting peaks directly on max-over-theta (`detect_catalogue_from_snr`) inflates noise
(max of many maps far exceeds 5 sigma) -> ~80 false/tile. Prefer the candidate-based consolidation.
"""
import numpy as np
from scipy.ndimage import maximum_filter

__all__ = ["consolidate_saddle_catalogue", "consolidate_candidates", "detect_from_candidates",
           "detect_catalogue_from_snr", "detect_peaks", "relocalize",
           "maps_from_tensor", "maps_from_loader"]


def _segment_min(M, p, q):
    """Min of M along the straight pixel segment between p and q (interior points only) -- the saddle."""
    (i0, j0), (i1, j1) = p, q
    d = int(max(abs(i1 - i0), abs(j1 - j0)))
    if d <= 1:
        return min(M[i0, j0], M[i1, j1])
    ii = np.round(np.linspace(i0, i1, d + 1)).astype(int)
    jj = np.round(np.linspace(j0, j1, d + 1)).astype(int)
    return float(M[ii, jj][1:-1].min())


def consolidate_candidates(ci, cj, cq, peak_map, res_arcmin, saddle_depth=2.0, link_radius_arcmin=12.0):
    """Consolidate pre-detected candidate peaks (pixel coords ci,cj, ordering value cq) with the
    saddle test on peak_map. Two candidates are kept separate iff the saddle between them is deep
    (>= saddle_depth); else the lower-cq one is dropped (same peak). Returns indices of KEPT candidates.
    Replaces szifi's theta-scaled merge while preserving its noise-controlled candidate set."""
    ci = np.asarray(ci, int); cj = np.asarray(cj, int); cq = np.asarray(cq, float)
    order = np.argsort(cq)[::-1]
    R_pix = link_radius_arcmin / res_arcmin
    acc, keep = [], []
    for k in order:
        i, j = int(ci[k]), int(cj[k]); new = True
        for ai, aj in acc:
            if np.hypot(i - ai, j - aj) > R_pix:
                continue
            depth = min(peak_map[i, j], peak_map[ai, aj]) - _segment_min(peak_map, (i, j), (ai, aj))
            if depth < saddle_depth:
                new = False; break
        if new:
            acc.append((i, j)); keep.append(int(k))
    return np.array(keep, int)


def relocalize(ci, cj, peak_map, reloc_radius_pix):
    """Snap each candidate to the brightest peak_map pixel within reloc_radius_pix -- re-localise a
    broad-filter detection (offset from the true centre) onto the resolved-scale peak.
    (Caution: re-localising compact, well-centred detections can merge them onto neighbours; on the
    CFC tiles this was net-negative, so it is OFF by default in the consolidation entry points.)"""
    ny, nx = peak_map.shape; R = int(round(reloc_radius_pix)); ni = []; nj = []
    for i, j in zip(np.asarray(ci, int), np.asarray(cj, int)):
        i0, i1 = max(0, i - R), min(ny, i + R + 1); j0, j1 = max(0, j - R), min(nx, j + R + 1)
        sub = peak_map[i0:i1, j0:j1]; a = np.unravel_index(int(np.argmax(sub)), sub.shape)
        ni.append(i0 + a[0]); nj.append(j0 + a[1])
    return np.array(ni, int), np.array(nj, int)


def maps_from_tensor(q_tensor, theta_500_vec, theta_cap_arcmin):
    """Build (peak_map = max over theta<=cap, score_map = max over all theta, theta_arg_map)."""
    tv = np.asarray(theta_500_vec, float)
    cap = tv <= theta_cap_arcmin
    if not cap.any():
        cap[:3] = True
    return q_tensor[:, :, cap].max(axis=2), q_tensor.max(axis=2), tv[q_tensor.argmax(axis=2)]


def maps_from_loader(load_slice, theta_500_vec, theta_cap_arcmin, shape):
    """Stream slices (load_slice(j) -> 2D array) to build the 3 maps without holding the tensor."""
    tv = np.asarray(theta_500_vec, float); ny, nx = shape
    peak = np.full((ny, nx), -np.inf, np.float32)
    score = np.full((ny, nx), -np.inf, np.float32)
    targ = np.zeros((ny, nx), np.float32)
    for j in range(len(tv)):
        s = load_slice(j).astype(np.float32)
        upd = s > score; score[upd] = s[upd]; targ[upd] = tv[j]
        if tv[j] <= theta_cap_arcmin:
            np.maximum(peak, s, out=peak)
    return peak, score, targ


def detect_peaks(peak_map, score_map, theta_arg_map, q_th, res_arcmin, wcs=None,
                 saddle_depth=2.0, link_radius_arcmin=12.0):
    """Full from-SNR peak detection: 2D maxima of peak_map above q_th, saddle-consolidated.
    WARNING: detecting on max-over-theta inflates the noise -> high false rate. Prefer
    consolidate_saddle_catalogue / detect_from_candidates (seeded by szifi's 3D-local-max candidates)."""
    is_max = (maximum_filter(peak_map, size=3) == peak_map) & (peak_map > q_th)
    cand = np.argwhere(is_max)
    if len(cand) == 0:
        return {k: np.array([]) for k in ("i", "j", "q_opt", "q_peak", "theta_500", "ra", "dec")}
    cand = cand[np.argsort(peak_map[cand[:, 0], cand[:, 1]])[::-1]]
    keep = consolidate_candidates(cand[:, 0], cand[:, 1], peak_map[cand[:, 0], cand[:, 1]],
                                  peak_map, res_arcmin, saddle_depth, link_radius_arcmin)
    ni, nj = cand[keep, 0], cand[keep, 1]
    out = {"i": ni, "j": nj, "q_opt": score_map[ni, nj], "q_peak": peak_map[ni, nj],
           "theta_500": theta_arg_map[ni, nj]}
    if wcs is not None:
        from pixell import enmap
        dec, ra = enmap.pix2sky(score_map.shape, wcs, np.array([ni, nj]))
        out["ra"] = np.rad2deg(ra) % 360.; out["dec"] = np.rad2deg(dec)
    return out


def detect_catalogue_from_snr(q_tensor, theta_500_vec, q_th, res_arcmin, wcs=None,
                              theta_cap_arcmin=4.0, saddle_depth=2.0, link_radius_arcmin=12.0):
    """Headline (from-SNR) API: SNR tensor -> peak catalogue (dict of arrays). See detect_peaks warning."""
    peak_map, score_map, theta_arg_map = maps_from_tensor(q_tensor, theta_500_vec, theta_cap_arcmin)
    return detect_peaks(peak_map, score_map, theta_arg_map, q_th, res_arcmin, wcs=wcs,
                        saddle_depth=saddle_depth, link_radius_arcmin=link_radius_arcmin)


def detect_from_candidates(ci, cj, cand_q, peak_map, score_map, theta_arg_map, res_arcmin, wcs=None,
                           reloc_radius_arcmin=0.0, reloc_theta_min=0.0, cand_theta=None,
                           saddle_depth=2.0, link_radius_arcmin=12.0):
    """Seed from szifi's noise-controlled candidates -> (optional) re-localise BROAD ones onto the
    resolved-scale peak_map -> saddle-consolidate. q_opt/theta_500 are szifi's reliable values
    (cand_q, cand_theta). Re-localisation OFF by default (was net-negative on the CFC tiles)."""
    ci = np.asarray(ci, int).copy(); cj = np.asarray(cj, int).copy()
    cand_q = np.asarray(cand_q, float)
    cth = None if cand_theta is None else np.asarray(cand_theta, float)
    if reloc_radius_arcmin and reloc_radius_arcmin > 0:
        if cth is not None and reloc_theta_min > 0:
            sel = cth > reloc_theta_min
            if sel.any():
                ri, rj = relocalize(ci[sel], cj[sel], peak_map, reloc_radius_arcmin / res_arcmin)
                ci[sel] = ri; cj[sel] = rj
        else:
            ci, cj = relocalize(ci, cj, peak_map, reloc_radius_arcmin / res_arcmin)
    keep = consolidate_candidates(ci, cj, cand_q, peak_map, res_arcmin, saddle_depth, link_radius_arcmin)
    ni, nj = ci[keep], cj[keep]
    out = {"i": ni, "j": nj, "q_opt": cand_q[keep], "q_peak": peak_map[ni, nj],
           "theta_500": (cth[keep] if cth is not None else theta_arg_map[ni, nj]), "keep": keep}
    if wcs is not None:
        from pixell import enmap
        dec, ra = enmap.pix2sky(score_map.shape, wcs, np.array([ni, nj]))
        out["ra"] = np.rad2deg(ra) % 360.; out["dec"] = np.rad2deg(dec)
    return out


def consolidate_saddle_catalogue(catalogue, q_tensor, theta_500_vec, wcs, res_arcmin,
                                 q_key="q_opt", theta_cap_arcmin=4.0, saddle_depth=2.0,
                                 link_radius_arcmin=12.0):
    """Drop-in replacement for cat.merge_detections, applied PER TILE.

    catalogue      : szifi.cat.cluster_catalogue of candidate detections for one tile, with sky
                     positions in 'ra','dec' (radians; CAR) and significance in `q_key`.
    q_tensor       : (ny, nx, n_theta) SNR maps for this tile (matches szifi's saved slices).
    theta_500_vec  : (n_theta,) filter scales in arcmin.
    wcs            : pixell WCS of the tile (q_tensor pixel grid).
    res_arcmin     : pixel size in arcmin.

    Keeps one entry per distinct SNR peak (all catalogue keys preserved), using the resolved-scale
    saddle test instead of the theta_500-scaled mask. Returns a new cluster_catalogue.
    """
    import copy as _copy
    from pixell import enmap
    tv = np.asarray(theta_500_vec, float)
    cap = tv <= theta_cap_arcmin
    if not cap.any():
        cap[:3] = True
    peak_map = q_tensor[:, :, cap].max(axis=2)
    ny, nx = peak_map.shape
    ra = np.asarray(catalogue.catalogue["ra"], float)
    dec = np.asarray(catalogue.catalogue["dec"], float)
    yy, xx = enmap.sky2pix((ny, nx), wcs, np.array([dec, ra]))
    yy = np.clip(np.round(yy).astype(int), 0, ny - 1)
    xx = np.clip(np.round(xx).astype(int), 0, nx - 1)
    q = np.asarray(catalogue.catalogue[q_key], float)
    keep = consolidate_candidates(yy, xx, q, peak_map, res_arcmin,
                                  saddle_depth=saddle_depth, link_radius_arcmin=link_radius_arcmin)
    out = _copy.deepcopy(catalogue)
    out.select_indices(np.sort(keep))
    return out
