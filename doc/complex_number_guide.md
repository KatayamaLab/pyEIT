# pyEIT 複素数対応ガイド

## 概要

pyEIT は複素数誘電率・複素インピーダンスに対応しており、周波数依存性を持つEIT（Electrical Impedance Tomography）計測が可能です。

このガイドでは、複素数を使用したEIT計測の基本的な使用方法を説明します。

---

## 1. 基本概念

### 複素誘電率とは

実際の生体組織や材料のインピーダンスは周波数に依存しており、複素数で表現されます：

```
σ(ω) = σ_dc + i·ω·ε = 実部 + i·虚部

σ_dc: DC導電率（周波数に無関係）
ω: 角周波数 (2π·f)
ε: 誘電率（周波数依存）
```

### pyEIT での表現

pyEIT では複素数誘電率を直接使用できます：

```python
import numpy as np

# 例：1kHz での生体組織のインピーダンス
freq = 1000  # Hz
omega = 2 * np.pi * freq

sigma_dc = 0.5    # DC導電率 (S/m)
epsilon = 1e-7    # 誘電率 (F/m)

# 複素誘電率
sigma_complex = sigma_dc + 1j * omega * epsilon
print(f"複素誘電率: {sigma_complex}")
# Output: 複素誘電率: (0.5+0.00062832j)
```

---

## 2. 基本的な使用方法

### 2.1 メッシュ作成と複素誘電率設定

```python
from pyeit.mesh import create, PyEITMesh
import numpy as np

# メッシュ作成
mesh = create(16, h0=0.1)

# 複素誘電率の定義
n_elems = mesh.n_elems
sigma_real = np.ones(n_elems) * 0.5      # 実部：DC導電率
sigma_imag = np.ones(n_elems) * 0.001    # 虚部：周波数成分

# 複素誘電率（numpy は自動的に複素型を認識）
sigma_complex = sigma_real + 1j * sigma_imag

# メッシュに設定
mesh.perm = sigma_complex

print(f"メッシュ dtype: {mesh.dtype}")
# Output: メッシュ dtype: complex128
```

### 2.2 測定プロトコル設定

```python
from pyeit.eit.protocol import build_meas_pattern_std, PyEITProtocol

# 励起電極パターン（16個の電極で隣同士を励起）
ex_mat = build_meas_pattern_std(16, dist_exc=1, step_meas=1, parser_meas="meas_current")

# 測定パターン作成
meas_mat, keep_ba = build_meas_pattern_std(ex_mat, 16, 1, "meas_current")

# プロトコルオブジェクト
protocol = PyEITProtocol(ex_mat, meas_mat, keep_ba)
```

### 2.3 前場計算（複素電圧測定値の計算）

```python
from pyeit.eit.fem import EITForward

# 前場ソルバー作成
fwd = EITForward(mesh, protocol)

# 複素電圧測定値を計算
v_complex = fwd.solve_eit(perm=sigma_complex)

print(f"測定値 dtype: {v_complex.dtype}")
print(f"測定値形状: {v_complex.shape}")
print(f"測定値 (最初の5個): {v_complex[:5]}")

# 出力例：
# 測定値 dtype: complex128
# 測定値形状: (208,)
# 測定値 (最初の5個): [0.125+0.001j 0.120+0.0008j ...]
```

---

## 3. 周波数依存シミュレーション

### 3.1 マルチ周波数計測

```python
import numpy as np
from pyeit.mesh import create
from pyeit.eit.fem import EITForward
from pyeit.eit.protocol import build_meas_pattern_std, PyEITProtocol

# メッシュ作成
mesh = create(16, h0=0.1)
n_elems = mesh.n_elems

# 基本パラメータ（生体組織の典型値）
sigma_dc = 0.5  # DC導電率
epsilon_0 = 8.854e-12  # 真空の誘電率

# 複数の周波数で計測
frequencies = [1e3, 1e4, 1e5, 1e6]  # 1kHz, 10kHz, 100kHz, 1MHz
results = {}

for freq in frequencies:
    print(f"\n周波数: {freq/1e3:.0f} kHz")
    
    # 周波数依存の複素誘電率
    omega = 2 * np.pi * freq
    relative_permittivity = 100  # 相対誘電率（周波数依存）
    epsilon = epsilon_0 * relative_permittivity
    
    sigma_complex = sigma_dc + 1j * omega * epsilon
    mesh.perm = np.ones(n_elems) * sigma_complex
    
    # プロトコル設定
    ex_mat = build_meas_pattern_std(16, dist_exc=1, step_meas=1, parser_meas="meas_current")
    meas_mat, keep_ba = build_meas_pattern_std(ex_mat, 16, 1, "meas_current")
    protocol = PyEITProtocol(ex_mat, meas_mat, keep_ba)
    
    # 前場計算
    fwd = EITForward(mesh, protocol)
    v = fwd.solve_eit()
    
    # 結果保存
    results[freq] = {
        'magnitude': np.abs(v),
        'phase': np.angle(v),
        'real': np.real(v),
        'imag': np.imag(v)
    }
    
    print(f"  |V| = {np.mean(np.abs(v)):.6f}")
    print(f"  ∠V = {np.mean(np.degrees(np.angle(v))):.2f}°")
    print(f"  Re(V) = {np.mean(np.real(v)):.6f}")
    print(f"  Im(V) = {np.mean(np.imag(v)):.6f}")
```

---

## 4. 逆問題解法（複素ヤコビアン法）

### 4.1 参照状態と摂動状態の計測

```python
import numpy as np
from pyeit.mesh import create
from pyeit.eit.fem import EITForward
from pyeit.eit.jac import JAC
from pyeit.eit.protocol import build_meas_pattern_std, PyEITProtocol

# メッシュ作成
mesh = create(16, h0=0.1)

# 参照状態（均一な複素誘電率）
sigma_ref = 0.5 + 1j * 0.001
mesh.perm = np.ones(mesh.n_elems) * sigma_ref

# プロトコル設定
ex_mat = build_meas_pattern_std(16, dist_exc=1, step_meas=1, parser_meas="meas_current")
meas_mat, keep_ba = build_meas_pattern_std(ex_mat, 16, 1, "meas_current")
protocol = PyEITProtocol(ex_mat, meas_mat, keep_ba)

# 参照電圧を計測
fwd = EITForward(mesh, protocol)
v_ref = fwd.solve_eit()

# 摂動状態：メッシュの一部に異なる誘電率を持つ領域を追加
sigma_perturb = 0.5 + 1j * 0.001
sigma_perturb_anomaly = 1.0 + 1j * 0.005  # 異常領域

perm_perturb = np.ones(mesh.n_elems) * sigma_perturb
perm_perturb[:5] = sigma_perturb_anomaly  # 最初の5要素を異常に

# 摂動状態の電圧を計測
v_perturb = fwd.solve_eit(perm=perm_perturb)

print(f"参照電圧 dtype: {v_ref.dtype}")
print(f"電圧変化（複素差）の大きさ: {np.linalg.norm(v_perturb - v_ref):.6f}")
```

### 4.2 複素ヤコビアン法による再構成

```python
# JAC ソルバーセットアップ
solver = JAC(mesh, protocol)
solver.setup(perm=sigma_ref)

print(f"ヤコビアン行列 dtype: {solver.J.dtype}")
print(f"ヤコビアン行列形状: {solver.J.shape}")
print(f"参照電圧 v0 dtype: {solver.v0.dtype}")

# 導電率変化を推定（solve_gs メソッド）
ds = solver.solve_gs(v_perturb, v_ref)

print(f"\n推定導電率変化:")
print(f"  dtype: {ds.dtype}")
print(f"  平均: {np.mean(ds):.6f}")
print(f"  最大: {np.max(ds):.6f}")
print(f"  最小: {np.min(ds):.6f}")

# 実部と虚部を分離表示
print(f"\n実部:")
print(f"  平均: {np.mean(np.real(ds)):.6f}")
print(f"虚部:")
print(f"  平均: {np.mean(np.imag(ds)):.6f}")
```

### 4.3 ガウス-ニュートン反復ソルバー

```python
# 静的EIT問題: 実測値から導電率分布を逆解析
v_measured = v_perturb  # 実測値（複素）
initial_guess = sigma_ref * np.ones(mesh.n_elems)

# GN反復ソルバー
perm_reconstructed = solver.gn(
    v_measured,
    x0=initial_guess,
    maxiter=5,
    gtol=1e-4,
    verbose=True
)

print(f"\n再構成導電率:")
print(f"  dtype: {perm_reconstructed.dtype}")
print(f"  実部平均: {np.mean(np.real(perm_reconstructed)):.6f}")
print(f"  虚部平均: {np.mean(np.imag(perm_reconstructed)):.6f}")
```

---

## 5. データ型の取り扱い

### 5.1 実数と複素数の自動処理

```python
import numpy as np
from pyeit.mesh import create
from pyeit.eit.fem import EITForward
from pyeit.eit.protocol import build_meas_pattern_std, PyEITProtocol

mesh = create(16, h0=0.1)
ex_mat = build_meas_pattern_std(16, dist_exc=1, step_meas=1, parser_meas="meas_current")
meas_mat, keep_ba = build_meas_pattern_std(ex_mat, 16, 1, "meas_current")
protocol = PyEITProtocol(ex_mat, meas_mat, keep_ba)

fwd = EITForward(mesh, protocol)

# ケース1: 実数入力
perm_real = np.ones(mesh.n_elems) * 0.5
v_real = fwd.solve_eit(perm=perm_real)
print(f"実数入力 → 出力 dtype: {v_real.dtype}")
# Output: float64（虚部は0に近い）

# ケース2: 複素数入力
perm_complex = np.ones(mesh.n_elems) * (0.5 + 1j * 0.001)
v_complex = fwd.solve_eit(perm=perm_complex)
print(f"複素数入力 → 出力 dtype: {v_complex.dtype}")
# Output: complex128
```

### 5.2 型チェック

```python
# 型チェック関数
def check_complex_data(data, name="data"):
    """複素数かどうかをチェック"""
    if np.iscomplexobj(data):
        real_part = np.real(data)
        imag_part = np.imag(data)
        imag_ratio = np.linalg.norm(imag_part) / np.linalg.norm(real_part)
        print(f"{name}:")
        print(f"  dtype: {data.dtype}")
        print(f"  複素数: Yes")
        print(f"  虚部/実部比: {imag_ratio:.2%}")
    else:
        print(f"{name}:")
        print(f"  dtype: {data.dtype}")
        print(f"  複素数: No")

check_complex_data(v_real, "実数測定値")
check_complex_data(v_complex, "複素数測定値")
```

---

## 6. 実践的な例：周波数スイープによる画像再構成

```python
import numpy as np
from pyeit.mesh import create
from pyeit.eit.fem import EITForward
from pyeit.eit.jac import JAC
from pyeit.eit.protocol import build_meas_pattern_std, PyEITProtocol

def reconstruct_at_frequency(freq, mesh, protocol, solver, sigma_ref):
    """
    特定周波数での画像再構成
    """
    # 周波数依存の複素誘電率
    omega = 2 * np.pi * freq
    sigma_dc = 0.5
    epsilon = 1e-7
    sigma_complex = sigma_dc + 1j * omega * epsilon
    
    # 摂動状態を仮定（実測値の代わり）
    mesh.perm = np.ones(mesh.n_elems) * sigma_complex
    fwd = EITForward(mesh, protocol)
    v_meas = fwd.solve_eit()
    
    # 参照状態
    v_ref = fwd.solve_eit(perm=sigma_ref * np.ones(mesh.n_elems))
    
    # 再構成
    ds = solver.solve_gs(v_meas, v_ref)
    
    return ds

# セットアップ
mesh = create(16, h0=0.1)
ex_mat = build_meas_pattern_std(16, dist_exc=1, step_meas=1, parser_meas="meas_current")
meas_mat, keep_ba = build_meas_pattern_std(ex_mat, 16, 1, "meas_current")
protocol = PyEITProtocol(ex_mat, meas_mat, keep_ba)

sigma_ref = 0.5 + 1j * 0.001
mesh.perm = np.ones(mesh.n_elems) * sigma_ref

fwd = EITForward(mesh, protocol)
solver = JAC(mesh, protocol)
solver.setup(perm=sigma_ref * np.ones(mesh.n_elems))

# 周波数スイープ
frequencies = [1e3, 1e4, 1e5]
reconstructions = {}

for freq in frequencies:
    ds = reconstruct_at_frequency(freq, mesh, protocol, solver, sigma_ref)
    reconstructions[freq] = ds
    print(f"{freq/1e3:.0f}kHz: |ds|_avg = {np.mean(np.abs(ds)):.6f}")

print("\n周波数依存の再構成完了")
```

---

## 7. よくある質問（FAQ）

### Q1: 複素数と実数を混在できますか？

**A**: はい。NumPy の自動型昇格により、実数と複素数の演算は自動的に複素数になります。

```python
real_part = 0.5
complex_part = 0.001j
result = real_part + complex_part  # → (0.5+0.001j)
```

### Q2: 虚部がある場合、計算が遅くなりますか？

**A**: 若干遅くなりますが、実用的には無視できる程度です。複素数は実数の2倍のメモリを使いますが、計算量は同等です。

### Q3: 既存のコード（実数専用）はそのまま動きますか？

**A**: はい。完全な後方互換性があります。すべての既存テストが合格しています。

### Q4: 複素ヤコビアン法と実数ヤコビアン法の違いは？

**A**: 複素ヤコビアン法では `J^H * J`（エルミート共役）を使用し、周波数依存性を保持します。実数法では `J^T * J` を使用します。

---

## 8. 参考資料

### 複素インピーダンス計測の理論

- 生体組織のインピーダンス: Cole-Cole モデル
- EIT のヤコビアン法: Adler et al. (2007)

### 関連する物理定数

```python
# よく使う定数
epsilon_0 = 8.854e-12  # 真空の誘電率 (F/m)
mu_0 = 4 * np.pi * 1e-7  # 真空の透磁率 (H/m)
c = 3e8  # 光速 (m/s)
```

---

## 9. トラブルシューティング

### 問題1: 警告「Casting complex values to real」が出る

**原因**: `spsolve()` が複素数を返さない場合がある
**解決**: 修正済み。fem.py で型変換を明示的に実行

### 問題2: 虚部がすべて0になる

**原因**: メッシュの dtype が実数になっている
**解決**: 
```python
# 複素数メッシュを確認
print(mesh.dtype)  # complex128 であることを確認

# 複素誘電率を明示的に設定
mesh.perm = np.array([...], dtype=complex)
```

### 問題3: ヤコビアン計算が失敗する

**原因**: 参照電圧が実数になっている
**解決**:
```python
# 参照状態も複素数で計算
sigma_ref = 0.5 + 1j * 0.001
mesh.perm = np.ones(mesh.n_elems) * sigma_ref
fwd = EITForward(mesh, protocol)
v_ref = fwd.solve_eit()
# v_ref が complex128 であることを確認
```

---

## まとめ

- ✅ 複素数誘電率を直接使用可能
- ✅ 周波数依存のEIT計測が可能
- ✅ 複素ヤコビアン法で逆問題を解く
- ✅ 実数入力との後方互換性を維持
- ✅ 自動型処理により安全に動作

詳細は各モジュールのドキュメント文字列を参照してください。
