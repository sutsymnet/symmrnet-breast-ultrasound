#!/usr/bin/env python3
"""
audit_pooling.py - static audit of a wavelet-pooling notebook or script.

Reads a .ipynb or .py file WITHOUT executing it and reports, for the thesis
record:

  1. the pooling layer's filter coefficients, and whether they match a real
     PyWavelets filter;
  2. the shape into which those coefficients are folded, and whether that
     shape is the separable 2-D analysis kernel h (x) h;
  3. the normalisation applied, and the resulting DC gain;
  4. the rank of the kernel (rank 1 = separable) and its frequency response
     at DC and at the two Nyquist axes;
  5. the operation order inside call(), including any nonlinearity and how
     the spatial reduction is actually performed;
  6. the input-stage wavelet basis and its sub-band rescaling factors;
  7. the model architecture and its trainable-parameter count, computed from
     the layer list.

Usage
    python audit_pooling.py notebook.ipynb
    python audit_pooling.py *.ipynb --csv audit.csv

Requires numpy. PyWavelets is optional but strongly recommended: without it
the coefficient-identity check is skipped.
"""

import argparse
import ast
import csv
import glob
import json
import os
import re
import sys

import numpy as np

try:
    import pywt
    HAVE_PYWT = True
except ImportError:
    HAVE_PYWT = False


# --------------------------------------------------------------------------- io
def strip_magics(src):
    """Comment out Jupyter magics and shell escapes so ast.parse succeeds."""
    out = []
    for line in src.split('\n'):
        st = line.lstrip()
        if st.startswith('!') or st.startswith('%') or st.startswith('?'):
            out.append('#' + line)
        else:
            out.append(line)
    return '\n'.join(out)


def load_source(path):
    """Return the concatenated code of a .ipynb, or the text of a .py."""
    if path.endswith('.ipynb'):
        nb = json.load(open(path, encoding='utf-8'))
        cells = [c for c in nb.get('cells', []) if c.get('cell_type') == 'code']
        src = '\n'.join(''.join(c.get('source', [])) for c in cells)
    else:
        src = open(path, encoding='utf-8').read()
    return strip_magics(src)


# ------------------------------------------------------------------ ast helpers
def literal(node):
    """ast.literal_eval that returns None instead of raising."""
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def find_pool_classes(tree):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and re.search(r'pool', node.name, re.I):
            out.append(node)
    return out


def call_name(node):
    """Dotted name of a Call's func, e.g. 'tf.nn.depthwise_conv2d'."""
    f = node.func if isinstance(node, ast.Call) else node
    parts = []
    while isinstance(f, ast.Attribute):
        parts.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        parts.append(f.id)
    return '.'.join(reversed(parts))


def kwarg(node, name):
    for k in node.keywords:
        if k.arg == name:
            return literal(k.value)
    return None



def resolve_scalar(node, env):
    """Resolve a literal, or NAME[i] where NAME is a known coefficient array."""
    v = literal(node)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = resolve_scalar(node.operand, env)
        return None if inner is None else -inner
    if isinstance(node, ast.Subscript):
        base = getattr(node.value, 'id', None) or getattr(node.value, 'attr', None)
        idx = literal(node.slice)
        if base in env and isinstance(idx, int) and 0 <= idx < len(env[base]):
            return float(env[base][idx])
    return None


def resolve_matrix(node, env):
    """Resolve a nested list literal whose entries may be NAME[i] subscripts."""
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    rows = []
    for r in node.elts:
        if not isinstance(r, (ast.List, ast.Tuple)):
            return None
        vals = [resolve_scalar(e, env) for e in r.elts]
        if any(v is None for v in vals):
            return None
        rows.append(vals)
    if not rows or len({len(r) for r in rows}) != 1:
        return None
    return np.array(rows, dtype=float)


# ------------------------------------------------------------ layer extraction
def audit_layer(cls):
    """Pull coefficients, kernel layout, normalisation and call() order."""
    info = {'class': cls.name, 'coeffs': {}, 'kernels': {}, 'norm': None,
            'ops': [], 'reshape': {}, 'unused': []}

    for node in ast.walk(cls):
        # 1-D coefficient lists:  symlet4_h = [ ... ]
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            name = getattr(tgt, 'id', None) or getattr(tgt, 'attr', None)
            val = literal(node.value)
            if name and isinstance(val, list) and val and all(
                    isinstance(v, (int, float)) for v in val):
                info['coeffs'][name] = np.array(val, dtype=float)
            # nested list inside tf.constant -> the folded kernel
            if name and isinstance(node.value, ast.Call) and \
                    call_name(node.value).endswith('constant') and node.value.args:
                k = resolve_matrix(node.value.args[0], info['coeffs'])
                if k is not None:
                    info['kernels'][name] = k
            # normalisation:  self.h_kernel = self.h_kernel / <expr>
            if name and isinstance(node.value, ast.BinOp) and \
                    isinstance(node.value.op, ast.Div):
                d = ast.dump(node.value.right)
                if 'abs' in d and 'reduce_sum' in d:
                    info['norm'] = 'l1  (sum of |coefficients|)'
                elif 'square' in d or 'norm' in d or 'sqrt' in d:
                    info['norm'] = 'l2'
                else:
                    info['norm'] = 'other (see source)'

    # call() body: operation order
    for node in ast.walk(cls):
        if isinstance(node, ast.FunctionDef) and node.name == 'call':
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Call):
                    continue
                nm = call_name(sub)
                if nm.endswith('reshape') and len(sub.args) > 1:
                    shp = literal(sub.args[1])
                    src = getattr(sub.args[0], 'attr', None) or \
                        getattr(sub.args[0], 'id', '?')
                    if shp:
                        info['reshape'][src] = shp
                if re.search(r'(conv2d|relu|elu|sigmoid|tanh|avg_pool|max_pool|'
                             r'space_to_depth|reduce_mean)', nm):
                    entry = {'op': nm}
                    st = kwarg(sub, 'strides')
                    if st is None:
                        for a in sub.args:
                            v = literal(a)
                            if isinstance(v, list) and len(v) == 4:
                                st = v
                    if st:
                        entry['strides'] = st
                    pad = kwarg(sub, 'padding')
                    if pad:
                        entry['padding'] = pad
                    ks = kwarg(sub, 'ksize')
                    if ks:
                        entry['ksize'] = ks
                    info['ops'].append(entry)

    used = ' '.join(str(o) for o in info['ops'])
    call_src = ''
    for node in ast.walk(cls):
        if isinstance(node, ast.FunctionDef) and node.name == 'call':
            call_src = ast.dump(node)
    for kname in info['kernels']:
        if kname not in call_src and kname.replace('self.', '') not in call_src:
            info['unused'].append(kname)
    return info


# ------------------------------------------------------------- kernel analysis
def match_pywt(h, tol=1e-8):
    """Return names of PyWavelets filters equal to h (dec_lo or dec_hi)."""
    if not HAVE_PYWT:
        return ['(pywt not installed - check skipped)']
    hits = []
    for name in pywt.wavelist(kind='discrete'):
        try:
            w = pywt.Wavelet(name)
        except Exception:
            continue
        if True:
            for label, f in (('dec_lo', w.dec_lo), ('dec_hi', w.dec_hi)):
                f = np.array(f)
                if f.shape == h.shape and np.allclose(f, h, atol=tol):
                    hits.append(f'{name}.{label}')
    return hits


def freq_response(K, wx, wy):
    rows, cols = K.shape
    v = 0j
    for m in range(rows):
        for n in range(cols):
            v += K[m, n] * np.exp(-1j * (wy * m + wx * n))
    return abs(v)


def analyse_kernel(K, norm_label):
    rows, cols = K.shape
    Kn = K / np.abs(K).sum() if norm_label and norm_label.startswith('l1') else K
    rank = int(np.linalg.matrix_rank(Kn))
    return {
        'shape': f'{rows} x {cols}',
        'rank': rank,
        'separable': rank == 1,
        'dc_gain': float(Kn.sum()),
        'H_nyq_x': float(freq_response(Kn, np.pi, 0.0)),
        'H_nyq_y': float(freq_response(Kn, 0.0, np.pi)),
        'H_corner': float(freq_response(Kn, np.pi, np.pi)),
        'n_negative': int((Kn < 0).sum()),
    }


# ------------------------------------------------------- preprocessing + model
def audit_preprocessing(src):
    out = {'basis': sorted(set(re.findall(r"pywt\.i?dwt2\([^)]*?['\"](\w+)['\"]", src, re.S))),
           'factors': {}}
    for m in re.finditer(r'(\w*c([AHVD]))\s*=\s*c[AHVD]\s*\*\s*([\d.]+)', src):
        out['factors'][m.group(2)] = float(m.group(3))
    out['clip'] = bool(re.search(r'np\.clip\(', src))
    out['divide255'] = bool(re.search(r'/\s*255', src))
    out['clahe'] = bool(re.search(r'(?i)clahe', src))
    return out


LAYER_RE = re.compile(
    r'(Conv2D|Dense|AveragePooling2D|MaxPooling2D|GlobalAveragePooling2D|'
    r'BatchNormalization|Dropout|Flatten|\w*PoolingLayer)\s*\(([^)]*)\)')


def module_consts(src):
    env = {}
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return env
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                isinstance(node.targets[0], ast.Name):
            v = literal(node.value)
            if isinstance(v, int):
                env[node.targets[0].id] = v
    return env


def num_args(args, env):
    """Numbers in an argument string, resolving known constant names."""
    out = []
    for tok in re.findall(r"[A-Za-z_]\w*|\d+", args):
        if tok.isdigit():
            out.append(int(tok))
        elif tok in env:
            out.append(env[tok])
    return out


def audit_model(src):
    i = src.find('def build_')
    if i < 0:
        return None
    seg = src[i:i + 3000]
    env = module_consts(src)
    hw = env.get('IMAGE_HEIGHT') or env.get('IMAGE_WIDTH')
    ch = env.get('IMAGE_CHANNELS')
    m = re.search(r'input_shape\s*=\s*\(([^)]*)\)', seg)
    if m:
        dims = num_args(m.group(1), env)
        if len(dims) == 3:
            hw, ch = dims[0], dims[2]
    layers, params, flat = [], 0, None
    for m in LAYER_RE.finditer(seg):
        kind, args = m.group(1), m.group(2)
        nums = num_args(args, env)
        p = 0
        if kind == 'Conv2D' and nums and ch:
            f = nums[0]
            k = nums[1] if len(nums) > 1 else 3
            p = k * k * ch * f + f
            ch = f
        elif kind == 'Dense' and nums:
            u = nums[0]
            if flat is None and hw and ch:
                flat = hw * hw * ch
            p = flat * u + u if flat else 0
            flat = u
        elif 'PoolingLayer' in kind or kind in ('AveragePooling2D', 'MaxPooling2D'):
            if hw:
                hw //= 2
        elif kind == 'Flatten' and hw and ch:
            flat = hw * hw * ch
        elif kind == 'BatchNormalization' and ch:
            p = 4 * ch
        layers.append((kind, args.strip()[:40], p))
        params += p
    return {'layers': layers, 'params': params,
            'has_batchnorm': any(l[0] == 'BatchNormalization' for l in layers),
            'extra_avgpool': sum(1 for l in layers if l[0] == 'AveragePooling2D')}


# ----------------------------------------------------------------------- report
def audit_file(path):
    src = load_source(path)
    tree = ast.parse(src)
    rows = []
    print('=' * 78)
    print('FILE:', os.path.basename(path))
    print('=' * 78)

    classes = find_pool_classes(tree)
    if not classes:
        print('  no class matching /pool/ found')
    for cls in classes:
        info = audit_layer(cls)
        print(f'\n[1] POOLING LAYER  {info["class"]}')
        for name, h in info['coeffs'].items():
            hits = match_pywt(h)
            print(f'    coefficients {name:<14} n={len(h):<3} '
                  f'matches: {", ".join(hits) if hits else "NO PyWavelets FILTER MATCHES"}')

        print(f'\n[2] KERNEL CONSTRUCTION   normalisation: {info["norm"]}')
        for name, K in info['kernels'].items():
            a = analyse_kernel(K, info['norm'])
            src_len = K.size
            print(f'    {name:<14} shape {a["shape"]:<8} '
                  f'(separable 2-D analysis kernel would be {src_len} x {src_len})')
            print(f'        rank {a["rank"]}   separable: '
                  f'{"YES" if a["separable"] else "NO  <-- not a 2-D wavelet filter"}')
            print(f'        DC gain {a["dc_gain"]:.4f}'
                  f'{"   <-- not unit / sqrt2 normalised" if abs(a["dc_gain"]-1) > 1e-6 and abs(a["dc_gain"]-np.sqrt(2)) > 1e-6 else ""}')
            print(f'        |H| at Nyquist-x {a["H_nyq_x"]:.4f}   '
                  f'Nyquist-y {a["H_nyq_y"]:.4f}   corner {a["H_corner"]:.4f}')
            if max(a['H_nyq_x'], a['H_nyq_y']) > 0.25:
                print('        NOT low-pass on both axes: high-frequency energy survives')
            rows.append(dict(file=os.path.basename(path), layer=info['class'],
                             kernel=name, **a))
        if info['unused']:
            print(f'    constructed but NEVER USED in call(): {", ".join(info["unused"])}')

        print('\n[3] call() OPERATION ORDER')
        for j, op in enumerate(info['ops'], 1):
            extra = ' '.join(f'{k}={v}' for k, v in op.items() if k != 'op')
            print(f'    {j}. {op["op"]:<26} {extra}')
        names = [o['op'] for o in info['ops']]
        if any('relu' in n or 'elu' in n or 'sigmoid' in n for n in names):
            print('    NONLINEARITY present -> the operator is not a linear transform')
        if any('avg_pool' in n or 'max_pool' in n for n in names):
            print('    reduction performed by a pooling op, not by transform decimation')
        if not any('conv' in n for n in names[1:]) and len(
                [n for n in names if 'conv' in n]) < 2:
            print('    only ONE filtering path executed -> no sub-band decomposition')

    pre = audit_preprocessing(src)
    print('\n[4] INPUT PREPROCESSING')
    print(f'    wavelet basis   : {", ".join(pre["basis"]) or "none found"}')
    print(f'    rescale factors : {pre["factors"] or "none found"}')
    print(f'    /255            : {pre["divide255"]}     clip: {pre["clip"]}'
          f'     CLAHE: {pre["clahe"]}')

    mod = audit_model(src)
    print('\n[5] MODEL')
    if not mod:
        print('    no build_* function found')
    else:
        for kind, args, p in mod['layers']:
            print(f'    {kind:<24} {args:<42} params {p:>9,}')
        print(f'    {"TOTAL trainable":<24} {"":<42} params {mod["params"]:>9,}')
        print(f'    BatchNormalization present : {mod["has_batchnorm"]}')
        print(f'    extra AveragePooling2D     : {mod["extra_avgpool"]}')
    print()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='+', help='.ipynb or .py files (globs allowed)')
    ap.add_argument('--csv', help='write the kernel-metric table to this file')
    a = ap.parse_args()

    files = []
    for p in a.paths:
        files.extend(sorted(glob.glob(p)) or [p])
    if not HAVE_PYWT:
        print('WARNING: PyWavelets not installed; coefficient identity not checked.\n'
              '         pip install PyWavelets\n')

    rows = []
    for f in files:
        try:
            rows.extend(audit_file(f))
        except Exception as e:
            print(f'!! {f}: {type(e).__name__}: {e}\n')

    if a.csv and rows:
        with open(a.csv, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f'kernel-metric table written to {a.csv}  ({len(rows)} rows)')


if __name__ == '__main__':
    main()
