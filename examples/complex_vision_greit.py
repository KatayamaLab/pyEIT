# coding: utf-8
"""
Complex number visualization for EIT reconstruction results using GREIT
GREITアルゴリズムを使用した複素数可視化
"""

import numpy as np
import matplotlib.pyplot as plt
from pyeit.mesh import create
from pyeit.eit.fem import EITForward
from pyeit.eit.greit import GREIT
from pyeit.eit.protocol import create as create_protocol


def plot_complex_split_greit(ds_complex, solver):
    """
    複素数再構成結果を実部と虚部に分離して表示（GREITグリッド版）

    Parameters
    ----------
    ds_complex : np.ndarray
        複素数再構成結果（フラット配列）
    solver : GREIT
        GREITソルバー
    """
    # マスク付きデータ取得
    ds_real = np.real(ds_complex).copy()
    ds_imag = np.imag(ds_complex).copy()

    xg_real, yg_real, ds_real_grid = solver.mask_value(ds_real, mask_value=np.nan)
    xg_imag, yg_imag, ds_imag_grid = solver.mask_value(ds_imag, mask_value=np.nan)

    # 図作成
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 実部プロット
    pc1 = axes[0].contourf(xg_real, yg_real, ds_real_grid, levels=20, cmap="RdBu_r")
    axes[0].set_aspect("equal")
    axes[0].set_title("Real Part (Conductivity Change)", fontsize=12, fontweight="bold")
    plt.colorbar(pc1, ax=axes[0], label="Re(Δσ)")

    # 虚部プロット
    pc2 = axes[1].contourf(xg_imag, yg_imag, ds_imag_grid, levels=20, cmap="RdBu_r")
    axes[1].set_aspect("equal")
    axes[1].set_title(
        "Imaginary Part (Frequency Component)", fontsize=12, fontweight="bold"
    )
    plt.colorbar(pc2, ax=axes[1], label="Im(Δσ)")

    fig.suptitle("GREIT: Real and Imaginary Parts", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_complex_with_stats_greit(ds_complex, solver):
    """
    複素数再構成結果を実部・虚部・マグニチュード・位相とともに表示（GREITグリッド版）

    Parameters
    ----------
    ds_complex : np.ndarray
        複素数再構成結果（フラット配列）
    solver : GREIT
        GREITソルバー
    """
    # データ抽出
    ds_real = np.real(ds_complex).copy()
    ds_imag = np.imag(ds_complex).copy()
    ds_mag = np.abs(ds_complex).copy()
    ds_phase = np.angle(ds_complex).copy()

    # マスク付きデータ取得
    xg_real, yg_real, ds_real_grid = solver.mask_value(ds_real, mask_value=np.nan)
    xg_imag, yg_imag, ds_imag_grid = solver.mask_value(
        ds_imag.copy(), mask_value=np.nan
    )
    xg_mag, yg_mag, ds_mag_grid = solver.mask_value(ds_mag.copy(), mask_value=np.nan)
    xg_phase, yg_phase, ds_phase_grid = solver.mask_value(
        ds_phase.copy(), mask_value=np.nan
    )

    # 図作成
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # プロットデータ
    data_list = [
        (
            xg_real,
            yg_real,
            ds_real_grid,
            axes[0, 0],
            "Real Part (Conductivity)",
            "RdBu_r",
        ),
        (
            xg_imag,
            yg_imag,
            ds_imag_grid,
            axes[0, 1],
            "Imaginary Part (Reactance)",
            "RdBu_r",
        ),
        (xg_mag, yg_mag, ds_mag_grid, axes[1, 0], "Magnitude |Δσ|", "viridis"),
        (xg_phase, yg_phase, ds_phase_grid, axes[1, 1], "Phase ∠Δσ (rad)", "hsv"),
    ]

    for xg_data, yg_data, ds_data, ax, title, cmap in data_list:
        pc = ax.contourf(xg_data, yg_data, ds_data, levels=20, cmap=cmap)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11, fontweight="bold")
        cbar = plt.colorbar(pc, ax=ax)

        # 統計情報を追加（マスク外の値のみ）
        valid_data = ds_data[~np.isnan(ds_data)]
        if len(valid_data) > 0:
            stats_text = (
                f"Min: {np.min(valid_data):.4f}\n"
                f"Max: {np.max(valid_data):.4f}\n"
                f"Mean: {np.mean(valid_data):.4f}\n"
                f"Std: {np.std(valid_data):.4f}"
            )
        else:
            stats_text = "No valid data"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    fig.suptitle(
        "GREIT: Complex Reconstruction with Statistics",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


def main():
    """メイン実行関数"""
    print("=" * 70)
    print("Complex Number Visualization - GREIT Algorithm")
    print("=" * 70)

    # 1. メッシュ作成
    print("\n[1] Creating mesh...")
    mesh = create(16, h0=0.1)
    print(f"    Mesh: {mesh.n_elems} elements, {mesh.n_nodes} nodes")

    # 2. プロトコル設定
    print("[2] Setting up protocol...")
    protocol = create_protocol(16, dist_exc=8, step_meas=1)
    print(
        f"    Protocol: {protocol.n_exc} excitations, {protocol.n_meas} measurements total"
    )

    # 3. 参照状態計測（複素誘電率）
    print("[3] Computing reference state (complex permittivity)...")
    sigma_ref = 1.0 + 0j
    mesh.perm = np.ones(mesh.n_elems) * sigma_ref
    print(f"    σ_ref = {sigma_ref}")

    fwd = EITForward(mesh, protocol)
    v_ref = fwd.solve_eit()
    print(
        f"    Reference voltage: dtype={v_ref.dtype}, mean={np.mean(np.abs(v_ref)):.6f}"
    )

    # 4. 摂動状態計測
    print("[4] Computing perturbed state...")
    perm_perturb = np.ones(mesh.n_elems) * sigma_ref
    perm_perturb[:8] = 1.0 + 1j * 0.005  # 異常領域
    v_perturb = fwd.solve_eit(perm=perm_perturb)
    print(f"    Perturbed voltage: dtype={v_perturb.dtype}")

    # 5. GREITを使用した逆問題解法
    print("[5] Solving inverse problem (GREIT)...")
    solver_greit = GREIT(mesh, protocol)
    solver_greit.setup(
        method="dist",
        p=0.20,
        lamb=1e-2,
        n=32,
        s=20.0,
        ratio=0.1,
        perm=sigma_ref * np.ones(mesh.n_elems),
    )

    ds_greit = solver_greit.solve(v_perturb, v_ref)
    print(f"    GREIT Reconstruction: dtype={ds_greit.dtype}")
    print(
        f"    Real part:      mean={np.mean(np.real(ds_greit)):.6f}, max={np.max(np.real(ds_greit)):.6f}"
    )
    print(
        f"    Imag part:      mean={np.mean(np.imag(ds_greit)):.6f}, max={np.max(np.imag(ds_greit)):.6f}"
    )
    print(
        f"    Magnitude:      mean={np.mean(np.abs(ds_greit)):.6f}, max={np.max(np.abs(ds_greit)):.6f}"
    )

    # 6. 可視化
    print("[6] Creating visualizations...")

    # Figure 1: シンプル版（実部と虚部）
    fig1 = plot_complex_split_greit(ds_greit, solver_greit)
    fig1.savefig("greit_simple.png", dpi=150, bbox_inches="tight")
    print("    ✓ Saved: greit_simple.png")

    # Figure 2: 統計情報付き版（グリッド版）
    fig2 = plot_complex_with_stats_greit(ds_greit, solver_greit)
    fig2.savefig("greit_with_stats.png", dpi=150, bbox_inches="tight")
    print("    ✓ Saved: greit_with_stats.png")
    print("✓ All GREIT visualizations completed!")
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
