def choose_checkup_key(df):
    """Pick a grouping key that creates repeated EIS sweeps (not per-row unique IDs)."""
    candidates = ['checkup_idx', 'checkup', 'chk_idx', 'seq_nr', 'sequence_nr', 'seq', 'timestamp_origin', 'sd_block_id']
    best_key, best_score = (None, -1.0)

    def _score_key(c):
        vc = df[c].value_counts(dropna=False)
        if len(vc) < 2:
            return None
        med = float(vc.median())
        mx = float(vc.max())
        if med < 3 and mx < 5:
            return None
        if med > 5000:
            return None
        return med + 0.1 * mx
    for c in candidates:
        if c not in df.columns:
            continue
        sc = _score_key(c)
        if sc is not None and sc > best_score:
            best_key, best_score = (c, sc)
    if best_key is None:
        blocked = {'freq_Hz', 'z_re_comp_mOhm', 'z_im_comp_mOhm', 'z_amp_comp_mOhm', 'z_ph_comp_deg'}
        for c in df.columns:
            if c in blocked:
                continue
            sc = _score_key(c)
            if sc is not None and sc > best_score:
                best_key, best_score = (c, sc)
    return best_key

def extract_re_rct_from_eis(eis_fp, crossing_mode='auto'):
    """Returns DataFrame with columns: checkup_idx, Re, Rct."""
    de = pd.read_csv(eis_fp, sep=';')
    for col in ['freq_Hz']:
        if col not in de.columns:
            raise ValueError(f"EIS file missing expected column '{col}': {eis_fp}")
    has_amp_phase = {'z_amp_comp_mOhm', 'z_ph_comp_deg'}.issubset(set(de.columns))
    has_re_im = {'z_re_comp_mOhm', 'z_im_comp_mOhm'}.issubset(set(de.columns))
    if not (has_amp_phase or has_re_im):
        raise ValueError(f'EIS file missing impedance fields in {eis_fp}.')
    strict_mask = pd.Series(True, index=de.index)
    if 'is_rt' in de.columns:
        rt_num = pd.to_numeric(de['is_rt'], errors='coerce')
        rt_str = de['is_rt'].astype(str).str.lower().str.strip()
        strict_mask &= (rt_num == 1) | rt_str.isin(['1', 'true', 'yes'])
    if 'soc_nom' in de.columns:
        soc_num = pd.to_numeric(de['soc_nom'], errors='coerce')
        soc_str = de['soc_nom'].astype(str)
        strict_mask &= np.isclose(soc_num.fillna(-9999), 50.0) | soc_str.str.contains('50', na=False)
    if 'valid' in de.columns:
        v_num = pd.to_numeric(de['valid'], errors='coerce')
        v_str = de['valid'].astype(str).str.lower().str.strip()
        strict_mask &= (v_num == 1) | v_str.isin(['1', 'true', 'yes'])
    de_strict = de.loc[strict_mask].copy()
    if len(de_strict):
        de = de_strict
    else:
        soft_mask = pd.Series(True, index=de.index)
        if 'is_rt' in de.columns:
            soft_mask &= de['is_rt'] == 1
        if 'valid' in de.columns:
            soft_mask &= de['valid'] == 1
        de_soft = de.loc[soft_mask].copy()
        if len(de_soft):
            de = de_soft
            print(f'Warning: no RT+SOC50 rows in {os.path.basename(eis_fp)}; using RT(+valid) fallback.')
        else:
            print(f'Warning: no usable EIS rows after filtering in {os.path.basename(eis_fp)}.')
            return pd.DataFrame(columns=['checkup_idx', 'Re', 'Rct'])
    chk_key = choose_checkup_key(de)
    if chk_key is None:
        if 'timestamp_s' in de.columns:
            de = de.sort_values('timestamp_s').reset_index(drop=True)
        else:
            de = de.reset_index(drop=True)
        f = pd.to_numeric(de['freq_Hz'], errors='coerce').values
        if len(f) == 0:
            return pd.DataFrame(columns=['checkup_idx', 'Re', 'Rct'])
        new_grp = np.zeros(len(f), dtype=int)
        if len(f) > 1:
            new_grp[1:] = (np.diff(f) > 0).astype(int)
        de['checkup_idx'] = np.cumsum(new_grp).astype(int)
        chk_key = 'checkup_idx'
    rows = []
    groups = []
    for chk_val, grp in de.groupby(chk_key):
        if len(grp) < 3:
            continue
        if 'timestamp_s' in grp.columns:
            ord_val = float(pd.to_numeric(grp['timestamp_s'], errors='coerce').median())
        else:
            ord_val = float(np.min(grp.index.values))
        groups.append((ord_val, chk_val, grp))
    groups = sorted(groups, key=lambda t: (np.nan_to_num(t[0], nan=1e+18), str(t[1])))
    for chk_new_idx, (_, chk_val, grp) in enumerate(groups):
        grp = grp.sort_values('freq_Hz', ascending=False)
        if {'z_re_comp_mOhm', 'z_im_comp_mOhm'}.issubset(set(grp.columns)):
            z_r = pd.to_numeric(grp['z_re_comp_mOhm'], errors='coerce').values.astype(float)
            z_i = pd.to_numeric(grp['z_im_comp_mOhm'], errors='coerce').values.astype(float)
        else:
            z_amp = pd.to_numeric(grp['z_amp_comp_mOhm'], errors='coerce').values.astype(float)
            z_ph = pd.to_numeric(grp['z_ph_comp_deg'], errors='coerce').values.astype(float)
            z_r = z_amp * np.cos(np.radians(z_ph))
            z_i = z_amp * np.sin(np.radians(z_ph))
        good = np.isfinite(z_r) & np.isfinite(z_i)
        z_r = z_r[good]
        z_i = z_i[good]
        if len(z_r) < 3:
            continue
        n_hf = min(3, len(z_r))
        Re = float(np.mean(z_r[:n_hf]))
        signs = np.sign(z_i)
        neg_to_pos = np.where((signs[:-1] < 0) & (signs[1:] >= 0))[0]
        pos_to_neg = np.where((signs[:-1] > 0) & (signs[1:] <= 0))[0]
        if crossing_mode == 'neg_to_pos':
            cand = neg_to_pos
        elif crossing_mode == 'pos_to_neg':
            cand = pos_to_neg
        else:
            cand = neg_to_pos if len(neg_to_pos) else pos_to_neg
        Rct = np.nan
        if len(cand):
            i = int(cand[0])
            x0, x1 = (float(z_r[i]), float(z_r[i + 1]))
            y0, y1 = (float(z_i[i]), float(z_i[i + 1]))
            den = y1 - y0
            if abs(den) > 1e-12:
                x_cross = x0 - y0 * (x1 - x0) / den
                rct_val = x_cross - Re
                if np.isfinite(rct_val) and rct_val >= 0:
                    Rct = float(rct_val)
        if not np.isfinite(Rct):
            zero_idx = np.where(np.isclose(z_i, 0.0, atol=1e-06))[0]
            if len(zero_idx):
                x_cross = float(np.max(z_r[zero_idx]))
                rct_val = x_cross - Re
                if np.isfinite(rct_val) and rct_val >= 0:
                    Rct = float(rct_val)
        if not np.isfinite(Rct):
            x_tail = float(np.nanpercentile(z_r, 95))
            rct_val = x_tail - Re
            if np.isfinite(rct_val):
                Rct = float(max(rct_val, 1e-06))
        if not np.isfinite(Rct) or Rct <= 1e-06:
            rct_span = float(np.nanmax(z_r) - Re)
            if np.isfinite(rct_span):
                Rct = float(max(rct_span, 1e-06))
        rows.append({'checkup_idx': int(chk_new_idx), 'Re': Re, 'Rct': Rct, 'Re_contributor_min_t': float(np.asarray(grp.timestamp_s)[good][:n_hf].min()), 'Re_contributor_max_t': float(np.asarray(grp.timestamp_s)[good][:n_hf].max()), 'group_max_t':float(grp.timestamp_s.max()), 'group_distinct_frequencies':int(grp.freq_Hz.nunique())})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values('checkup_idx').drop_duplicates('checkup_idx', keep='first')
    return out