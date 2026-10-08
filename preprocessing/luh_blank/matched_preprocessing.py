RAW_BASE, files = discover_dataset_base(RAW_BASE_CANDIDATES)
print_section('Phase 2 Data Loading')
print_kv('Dataset base', RAW_BASE)
print_kv('EOC files found', len(files))

sample_cols = pd.read_csv(files[0], sep=';').columns.tolist()
print_kv('Sample columns', sample_cols[:15])

q_col = choose_column(sample_cols, ['cap_aged_est_Ah'])

# Strict EOC target detection: avoid protocol/meta columns being mistaken as Re/Rct.
def is_valid_eoc_re(col_name):
    c = str(col_name).lower()
    good = ('re' in c) or ('r0' in c) or ('rs' in c) or ('electrolyte' in c)
    bad = any(t in c for t in [
        'cyc_', 'charged', 'condition', 'age_', 'temp', 'soc', 'rate',
        'timestamp', 'origin', 'block', 'profile', 'v_', 'i_'
    ])
    return good and not bad

def is_valid_eoc_rct(col_name):
    c = str(col_name).lower()
    good = ('rct' in c) or ('r_ct' in c) or ('charge_transfer' in c) or ('r1' in c)
    bad = any(t in c for t in [
        'cyc_', 'charged', 'condition', 'age_', 'temp', 'soc', 'rate',
        'timestamp', 'origin', 'block', 'profile', 'v_', 'i_'
    ])
    return good and not bad

re_try = choose_column(
    sample_cols,
    ['eis_re_ohm', 're_ohm', 'r_e_ohm', 'electrolyte_resistance_ohm', 'rs_ohm', 'r0_ohm'],
    required_tokens=['r'],
    forbidden_tokens=['rct', 'ct', 'init', 'timestamp', 'cyc', 'charged', 'condition', 'age']
)
rct_try = choose_column(
    sample_cols,
    ['eis_rct_ohm', 'rct_ohm', 'r_ct_ohm', 'charge_transfer_resistance_ohm', 'r1_ohm'],
    required_tokens=['rct'],
    forbidden_tokens=['re', 'init', 'timestamp', 'cyc', 'charged', 'condition', 'age']
)

re_col = re_try if (re_try is not None and is_valid_eoc_re(re_try)) else None
rct_col = rct_try if (rct_try is not None and is_valid_eoc_rct(rct_try)) else None

eis_dirs, eis_files_list = find_eis_sources(RAW_BASE)
USE_EIS_NYQUIST = len(eis_files_list) > 0

def choose_checkup_key(df):
    """Pick a grouping key that creates repeated EIS sweeps (not per-row unique IDs)."""
    candidates = ['checkup_idx', 'checkup', 'chk_idx', 'seq_nr', 'sequence_nr', 'seq', 'timestamp_origin', 'sd_block_id']
    best_key, best_score = None, -1.0

    def _score_key(c):
        vc = df[c].value_counts(dropna=False)
        if len(vc) < 2:
            return None
        med = float(vc.median())
        mx = float(vc.max())
        if med < 3 and mx < 5:
            return None
        # Prefer keys whose groups look like frequency sweeps (few to few-hundred points).
        if med > 5000:
            return None
        return med + 0.1 * mx

    for c in candidates:
        if c not in df.columns:
            continue
        sc = _score_key(c)
        if sc is not None and sc > best_score:
            best_key, best_score = c, sc

    # Generic fallback: scan remaining columns for usable repeated-group structure.
    if best_key is None:
        blocked = {'freq_Hz', 'z_re_comp_mOhm', 'z_im_comp_mOhm', 'z_amp_comp_mOhm', 'z_ph_comp_deg'}
        for c in df.columns:
            if c in blocked:
                continue
            sc = _score_key(c)
            if sc is not None and sc > best_score:
                best_key, best_score = c, sc

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
        raise ValueError(f"EIS file missing impedance fields in {eis_fp}.")

    # Strict filter first: RT + SOC50 (+ valid when available).
    strict_mask = pd.Series(True, index=de.index)
    if 'is_rt' in de.columns:
        rt_num = pd.to_numeric(de['is_rt'], errors='coerce')
        rt_str = de['is_rt'].astype(str).str.lower().str.strip()
        strict_mask &= ((rt_num == 1) | rt_str.isin(['1', 'true', 'yes']))
    if 'soc_nom' in de.columns:
        soc_num = pd.to_numeric(de['soc_nom'], errors='coerce')
        soc_str = de['soc_nom'].astype(str)
        strict_mask &= (np.isclose(soc_num.fillna(-9999), 50.0) | soc_str.str.contains('50', na=False))
    if 'valid' in de.columns:
        v_num = pd.to_numeric(de['valid'], errors='coerce')
        v_str = de['valid'].astype(str).str.lower().str.strip()
        strict_mask &= ((v_num == 1) | v_str.isin(['1', 'true', 'yes']))
    de_strict = de.loc[strict_mask].copy()

    if len(de_strict):
        de = de_strict
    else:
        # Graceful fallback: RT (+valid) without SOC constraint.
        soft_mask = pd.Series(True, index=de.index)
        if 'is_rt' in de.columns:
            soft_mask &= (de['is_rt'] == 1)
        if 'valid' in de.columns:
            soft_mask &= (de['valid'] == 1)
        de_soft = de.loc[soft_mask].copy()
        if len(de_soft):
            de = de_soft
            print(f"Warning: no RT+SOC50 rows in {os.path.basename(eis_fp)}; using RT(+valid) fallback.")
        else:
            print(f"Warning: no usable EIS rows after filtering in {os.path.basename(eis_fp)}.")
            return pd.DataFrame(columns=['checkup_idx', 'Re', 'Rct'])

    chk_key = choose_checkup_key(de)
    if chk_key is None:
        # Last-resort grouping: build sweep index from frequency reset boundaries.
        if 'timestamp_s' in de.columns:
            de = de.sort_values('timestamp_s').reset_index(drop=True)
        else:
            de = de.reset_index(drop=True)
        f = pd.to_numeric(de['freq_Hz'], errors='coerce').values
        if len(f) == 0:
            return pd.DataFrame(columns=['checkup_idx', 'Re', 'Rct'])
        new_grp = np.zeros(len(f), dtype=int)
        if len(f) > 1:
            # Typical sweep is descending frequency; an increase indicates new sweep.
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

    groups = sorted(groups, key=lambda t: (np.nan_to_num(t[0], nan=1e18), str(t[1])))

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
            # Auto mode: use neg->pos when available (standard Nyquist with -Im convention),
            # otherwise fallback to pos->neg.
            cand = neg_to_pos if len(neg_to_pos) else pos_to_neg

        Rct = np.nan
        if len(cand):
            i = int(cand[0])
            x0, x1 = float(z_r[i]), float(z_r[i + 1])
            y0, y1 = float(z_i[i]), float(z_i[i + 1])
            den = (y1 - y0)
            if abs(den) > 1e-12:
                x_cross = x0 - y0 * (x1 - x0) / den
                rct_val = x_cross - Re
                if np.isfinite(rct_val) and rct_val >= 0:
                    Rct = float(rct_val)

        # Fallback A: use explicit near-zero imaginary points if present.
        if not np.isfinite(Rct):
            zero_idx = np.where(np.isclose(z_i, 0.0, atol=1e-6))[0]
            if len(zero_idx):
                x_cross = float(np.max(z_r[zero_idx]))
                rct_val = x_cross - Re
                if np.isfinite(rct_val) and rct_val >= 0:
                    Rct = float(rct_val)

        # Fallback B: if no zero crossing exists, approximate the right intercept
        # from the high-real tail of the Nyquist curve.
        if not np.isfinite(Rct):
            x_tail = float(np.nanpercentile(z_r, 95))
            rct_val = x_tail - Re
            if np.isfinite(rct_val):
                Rct = float(max(rct_val, 1e-6))

        # Fallback C: use Re span when right-intercept estimate is degenerate.
        if (not np.isfinite(Rct)) or (Rct <= 1e-6):
            rct_span = float(np.nanmax(z_r) - Re)
            if np.isfinite(rct_span):
                Rct = float(max(rct_span, 1e-6))

        rows.append({'checkup_idx': int(chk_new_idx), 'Re': Re, 'Rct': Rct})

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values('checkup_idx').drop_duplicates('checkup_idx', keep='first')
    return out

print_section('Detected Target Columns', char='-')
print_kv('Q (EOC)', q_col)
print_kv('Re (EOC candidate)', re_col)
print_kv('Rct (EOC candidate)', rct_col)
print_kv('EIS dirs found', len(eis_dirs))
print_kv('EIS files found', len(eis_files_list))
print_kv('USE_EIS_NYQUIST', USE_EIS_NYQUIST)

if q_col is None:
    raise ValueError('Cannot detect Q column.')
if not USE_EIS_NYQUIST:
    raise ValueError('EIS Nyquist files not found. Cannot reconstruct Re/Rct reliably.')

col_map = {
    q_col: 'Q',
    'age_temp': 'temperature',
    'age_chg_rate': 'c_rate_chg',
    'age_dischg_rate': 'c_rate_dischg',
    'age_soc': 'soc_window',
    'num_cycles_op': 'cycle',
}
if re_col is not None:
    col_map[re_col] = 'Re'
if rct_col is not None:
    col_map[rct_col] = 'Rct'

base_keep = [
    'cell_id', 'timestamp_s', 'checkup_idx', 'age_type', 'soh_cap',
    'cyc_condition', 'cyc_charged', 't_start_degC'
]

def unique_preserve_order(cols):
    out, seen = [], set()
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out

def match_eis_file(cell_id, all_eis):
    exact = [f for f in all_eis if f'_{cell_id}.csv' in f or f'{cell_id}.csv' in f]
    if exact:
        return exact[0]
    partial = [f for f in all_eis if cell_id in os.path.basename(f)]
    if partial:
        return partial[0]

    # Fallback 1: match by Sxx_Cyy suffix token.
    cell_tail = '_'.join(str(cell_id).split('_')[-2:])
    tail_hits = [
        f for f in all_eis
        if '_'.join(os.path.basename(f).replace('.csv', '').split('_')[-2:]) == cell_tail
    ]
    if len(tail_hits) == 1:
        return tail_hits[0]
    if len(tail_hits) > 1:
        return sorted(tail_hits)[0]

    # Fallback 2: match by Cyy token only (last resort).
    cell_c = str(cell_id).split('_')[-1]
    c_hits = [
        f for f in all_eis
        if os.path.basename(f).replace('.csv', '').split('_')[-1] == cell_c
    ]
    if len(c_hits) == 1:
        return c_hits[0]

    return None

EIS_SIGN_MODE = 'auto'

# Visual Nyquist QA on up to 3 files to confirm sign convention quickly.
if USE_EIS_NYQUIST and len(eis_files_list):
    sample_eis = eis_files_list[:3]
    fig, axes = plt.subplots(1, len(sample_eis), figsize=(5 * len(sample_eis), 4))
    if len(sample_eis) == 1:
        axes = [axes]
    for ax, fp in zip(axes, sample_eis):
        de = pd.read_csv(fp, sep=';')
        strict_mask = pd.Series(True, index=de.index)
        if 'is_rt' in de.columns:
            strict_mask &= (de['is_rt'] == 1)
        if 'soc_nom' in de.columns:
            strict_mask &= (de['soc_nom'] == 50)
        de = de.loc[strict_mask].copy() if strict_mask.any() else de.copy()
        if len(de) == 0:
            ax.set_title(os.path.basename(fp)[:24] + ' (no RT/SOC50)')
            ax.axis('off')
            continue
        key = choose_checkup_key(de)
        if key is None:
            continue
        chk0 = de[de[key] == de[key].min()].sort_values('freq_Hz', ascending=False)
        if len(chk0) < 3:
            continue
        if {'z_re_comp_mOhm', 'z_im_comp_mOhm'}.issubset(set(chk0.columns)):
            z_r = chk0['z_re_comp_mOhm'].astype(float).values
            z_i = chk0['z_im_comp_mOhm'].astype(float).values
        else:
            z_r = chk0['z_amp_comp_mOhm'].astype(float).values * np.cos(np.radians(chk0['z_ph_comp_deg'].astype(float).values))
            z_i = chk0['z_amp_comp_mOhm'].astype(float).values * np.sin(np.radians(chk0['z_ph_comp_deg'].astype(float).values))
        ax.plot(z_r, -z_i, 'o-', ms=3)
        ax.set_xlabel('Re(Z) mOhm')
        ax.set_ylabel('-Im(Z) mOhm')
        ax.set_title(os.path.basename(fp)[:24])
        ax.axhline(0, color='gray', lw=0.6)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

all_rows = []
eis_cache = {}
for fp in files:
    d = pd.read_csv(fp, sep=';')
    cell_id = os.path.basename(fp).replace('cell_eocv2_', '').replace('.csv', '')
    d['cell_id'] = cell_id

    if 'checkup_idx' not in d.columns:
        d = d.sort_values('timestamp_s').reset_index(drop=True)
        d['checkup_idx'] = d.groupby('cell_id').cumcount()
    else:
        chk_num = pd.to_numeric(d['checkup_idx'], errors='coerce')
        if chk_num.notna().mean() >= 0.8:
            d['checkup_idx'] = chk_num.astype(float)
        else:
            # Non-numeric checkup labels: rebuild a stable sequential index.
            if 'timestamp_s' in d.columns:
                d = d.sort_values('timestamp_s').reset_index(drop=True)
            else:
                d = d.reset_index(drop=True)
            d['checkup_idx'] = np.arange(len(d), dtype=float)

    keep = unique_preserve_order(
        [c for c in base_keep if c in d.columns] +
        [c for c in col_map.keys() if c in d.columns]
    )
    d = d[keep].copy().rename(columns=col_map)
    d = d.loc[:, ~d.columns.duplicated()].copy()

    # Always overwrite Re/Rct from Nyquist-derived EIS extraction.
    d['Re'] = np.nan
    d['Rct'] = np.nan

    eis_fp = match_eis_file(cell_id, eis_files_list)
    if eis_fp:
        if eis_fp not in eis_cache:
            eis_cache[eis_fp] = extract_re_rct_from_eis(eis_fp, crossing_mode=EIS_SIGN_MODE)
        de = eis_cache[eis_fp]
        if len(de):
            de = de.rename(columns={'Re': 'Re_eis', 'Rct': 'Rct_eis'})
            d_merge = d.copy()
            d_merge['checkup_idx'] = pd.to_numeric(d_merge['checkup_idx'], errors='coerce')
            de['checkup_idx'] = pd.to_numeric(de['checkup_idx'], errors='coerce')
            d_merge = d_merge[d_merge['checkup_idx'].notna()].copy()
            de = de[de['checkup_idx'].notna()].copy()
            if len(d_merge) and len(de):
                d_merge['checkup_idx'] = d_merge['checkup_idx'].astype(np.float64)
                de['checkup_idx'] = de['checkup_idx'].astype(np.float64)
                d = pd.merge_asof(
                    d_merge.sort_values('checkup_idx'),
                    de.sort_values('checkup_idx'),
                    on='checkup_idx',
                    direction='nearest'
                )
            else:
                d = d_merge.copy()
            for c in ['Re_eis', 'Rct_eis']:
                if c in d.columns:
                    d[c] = d[c].ffill().bfill()
            d['Re'] = d['Re_eis']
            d['Rct'] = d['Rct_eis']
            d = d.drop(columns=[c for c in ['Re_eis', 'Rct_eis'] if c in d.columns])

    all_rows.append(d)

df = pd.concat(all_rows, ignore_index=True)

cells_before_eis_drop = df['cell_id'].nunique()
nan_re_cells = sorted(df.loc[df['Re'].isna() | df['Rct'].isna(), 'cell_id'].astype(str).unique().tolist())
if len(nan_re_cells):
    print(f'WARNING: {len(nan_re_cells)} cells have missing EIS-derived Re/Rct and will be dropped.')
    print(nan_re_cells)

for y in ['Q', 'Re', 'Rct']:
    df = df[df[y].notna()].copy()
print_kv('Cells after EIS filter', f"{df['cell_id'].nunique()} / {cells_before_eis_drop}")

if 'cyc_condition' in df.columns:
    df = df[df['cyc_condition'].isin(PROTOCOL_CYC_CONDITION)].copy()
if 'cyc_charged' in df.columns:
    df = df[df['cyc_charged'].isin(PROTOCOL_CYC_CHARGED)].copy()

if len(df) < MIN_ROWS_AFTER_FILTER and ALLOW_PROTOCOL_FALLBACK:
    print('Protocol filter too strict - applying fallback.')
    df = pd.concat(all_rows, ignore_index=True)
    for y in ['Q', 'Re', 'Rct']:
        df = df[df[y].notna()].copy()
    if 'cyc_condition' in df.columns:
        df = df[df['cyc_condition'].isin([1, 2])].copy()
    if 'cyc_charged' in df.columns:
        df = df[df['cyc_charged'].isin([0, 1])].copy()

df = df[(df['Q'] > 0) & (df['Re'] > 0) & (df['Rct'] > 0)].copy()
if 'soh_cap' in df.columns:
    df = df[(df['soh_cap'] >= 0) & (df['soh_cap'] <= 105)].copy()

df = df.sort_values(['cell_id', 'checkup_idx']).reset_index(drop=True)

# Clip gross outliers before monotonic enforcement.
def clip_and_enforce(s, z_thresh=3.0):
    s = s.astype(float)
    mu = float(s.median())
    sd = float(s.std())
    if not np.isfinite(sd) or sd < 1e-12:
        return s.cummax()
    s_clip = s.clip(lower=mu - z_thresh * sd, upper=mu + z_thresh * sd)
    return s_clip.cummax()

# Monotonic ground-truth enforcement.
df['Q'] = df.groupby('cell_id')['Q'].transform(lambda s: s.cummin())
df['Re'] = df.groupby('cell_id')['Re'].transform(clip_and_enforce)
df['Rct'] = df.groupby('cell_id')['Rct'].transform(clip_and_enforce)

# Derived features.
df['Q0'] = df.groupby('cell_id')['Q'].transform('first')
df['Re0'] = df.groupby('cell_id')['Re'].transform('first')
df['Rct0'] = df.groupby('cell_id')['Rct'].transform('first')

if 'cycle' in df.columns and df['cycle'].notna().mean() > 0.9:
    df['cycle'] = df['cycle'].astype(float)
    df['k_exp'] = df['cycle'] - df.groupby('cell_id')['cycle'].transform('first')
    k_unit = 'real_cycles'
else:
    df['k_exp'] = df['checkup_idx'].astype(float)
    k_unit = 'checkup_idx'
df['k_exp'] = df['k_exp'].clip(lower=0)
print_kv('k_exp unit', k_unit)
if k_unit == 'checkup_idx':
    print('WARNING: k_exp is checkup index, not real cycles. Physics rates are per-checkup.')

if 'temperature' not in df.columns:
    for alt in ['t_start_degC', 'age_temp', 't_avg_degC']:
        if alt in df.columns:
            df = df.rename(columns={alt: 'temperature'})
            break

if 'age_type' in df.columns and df['age_type'].dtype == object:
    df['age_type'] = pd.Categorical(df['age_type']).codes.astype(np.float32)

print_section('Phase 2 Table Summary')
print_kv('Rows', f'{len(df):,}')
print_kv('Cells', df['cell_id'].nunique())
print('\nTarget statistics:')
print(df[['Q', 'Re', 'Rct']].describe().T[['mean', 'std', 'min', 'max']])

cell_re_std = df.groupby('cell_id')['Re'].std()
cell_rct_std = df.groupby('cell_id')['Rct'].std()
print_section('EIS Sanity Diagnostics', char='-')
print_kv('Median Re std per cell', f"{float(cell_re_std.median()):.4f} mOhm")
print_kv('Median Rct std per cell', f"{float(cell_rct_std.median()):.4f} mOhm")
print_kv('Rct mean', f"{float(df['Rct'].mean()):.4f} mOhm")
print_kv('Re min / Rct min', f"{float(df['Re'].min()):.4f} / {float(df['Rct'].min()):.4f}")
assert float(df['Re'].std()) > 1e-6, 'Re is still constant - check EIS Nyquist extraction.'
assert abs(float(df['Re'].min()) - float(df['Rct'].min())) > 1e-9, 'Re and Rct mins are identical - check EIS extraction.'