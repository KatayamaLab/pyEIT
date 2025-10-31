# coding: utf-8
"""
Complex number visualization for EIT reconstruction results
オプション1: 実部と虚部を分離表示（シンプル版）
"""
import numpy as np
import matplotlib.pyplot as plt
from pyeit.mesh import create
from pyeit.eit.fem import EITForward
from pyeit.eit.jac import JAC
from pyeit.eit.protocol import create as create_protocol


def plot_complex_split(ds_complex, mesh):
    """
    複素数再構成結果を実部と虚部に分離して表示（シンプル版）
    
    Parameters
    ----------
    ds_complex : np.ndarray
        複素数再構成結果
    mesh : PyEITMesh
        メッシュオブジェクト
    """
    # データ抽出
    ds_real = np.real(ds_complex)
    ds_imag = np.imag(ds_complex)
    
    # メッシュ情報抽出
    nodes = mesh.node
    elements = mesh.element
    x = nodes[:, 0]
    y = nodes[:, 1]
    
    # 図作成
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 実部プロット
    pc1 = axes[0].tripcolor(x, y, elements, ds_real, cmap="RdBu_r")
    axes[0].set_aspect("equal")
    axes[0].set_title("Real Part (Conductivity Change)", fontsize=12, fontweight="bold")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    plt.colorbar(pc1, ax=axes[0], label="Re(Δσ)")
    
    # 虚部プロット
    pc2 = axes[1].tripcolor(x, y, elements, ds_imag, cmap="RdBu_r")
    axes[1].set_aspect("equal")
    axes[1].set_title("Imaginary Part (Frequency Component)", fontsize=12, fontweight="bold")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.colorbar(pc2, ax=axes[1], label="Im(Δσ)")
    
    fig.suptitle("Option 1: Real and Imaginary Parts", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_complex_with_stats(ds_complex, mesh):
    """
    複素数再構成結果を実部・虚部・マグニチュード・位相とともに表示
    
    Parameters
    ----------
    ds_complex : np.ndarray
        複素数再構成結果
    mesh : PyEITMesh
        メッシュオブジェクト
    """
    # データ抽出
    ds_real = np.real(ds_complex)
    ds_imag = np.imag(ds_complex)
    ds_mag = np.abs(ds_complex)
    ds_phase = np.angle(ds_complex)
    
    # メッシュ情報抽出
    nodes = mesh.node
    elements = mesh.element
    x = nodes[:, 0]
    y = nodes[:, 1]
    
    # 図作成
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # プロットデータ
    data_list = [
        (ds_real, axes[0, 0], "Real Part (Conductivity)", "RdBu_r"),
        (ds_imag, axes[0, 1], "Imaginary Part (Reactance)", "RdBu_r"),
        (ds_mag, axes[1, 0], "Magnitude |Δσ|", "viridis"),
        (ds_phase, axes[1, 1], "Phase ∠Δσ (rad)", "hsv"),
    ]
    
    for data, ax, title, cmap in data_list:
        pc = ax.tripcolor(x, y, elements, data, cmap=cmap)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = plt.colorbar(pc, ax=ax)
        
        # 統計情報を追加
        stats_text = (
            f"Min: {np.min(data):.4f}\n"
            f"Max: {np.max(data):.4f}\n"
            f"Mean: {np.mean(data):.4f}\n"
            f"Std: {np.std(data):.4f}"
        )
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        )
    
    fig.suptitle("Option 1: Complex Reconstruction with Statistics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def main():
    """メイン実行関数"""
    print("="*70)
    print("Complex Number Visualization - Option 1: Split Real/Imaginary Parts")
    print("="*70)
    
    # 1. メッシュ作成
    print("\n[1] Creating mesh...")
    mesh = create(16, h0=0.1)
    print(f"    Mesh: {mesh.n_elems} elements, {mesh.n_nodes} nodes")
    
    # 2. プロトコル設定
    print("[2] Setting up protocol...")
    protocol = create_protocol(16, dist_exc=8, step_meas=1)
    print(f"    Protocol: {protocol.n_exc} excitations, {protocol.n_meas} measurements total")
    
    # 3. 参照状態計測（複素誘電率）
    print("[3] Computing reference state (complex permittivity)...")
    sigma_ref = 0.5 + 1j * 0.001
    mesh.perm = np.ones(mesh.n_elems) * sigma_ref
    print(f"    σ_ref = {sigma_ref}")
    
    fwd = EITForward(mesh, protocol)
    v_ref = fwd.solve_eit()
    print(f"    Reference voltage: dtype={v_ref.dtype}, mean={np.mean(np.abs(v_ref)):.6f}")
    
    # 4. 摂動状態計測
    print("[4] Computing perturbed state...")
    perm_perturb = np.ones(mesh.n_elems) * sigma_ref
    perm_perturb[:8] = 1.0 + 1j * 0.005  # 異常領域
    v_perturb = fwd.solve_eit(perm=perm_perturb)
    print(f"    Perturbed voltage: dtype={v_perturb.dtype}")
    
    # 5. 逆問題解法
    print("[5] Solving inverse problem (JAC)...")
    solver = JAC(mesh, protocol)
    solver.setup(perm=sigma_ref * np.ones(mesh.n_elems))
    
    ds = solver.solve_gs(v_perturb, v_ref)
    print(f"    Reconstruction: dtype={ds.dtype}")
    print(f"    Real part:      mean={np.mean(np.real(ds)):.6f}, max={np.max(np.real(ds)):.6f}")
    print(f"    Imag part:      mean={np.mean(np.imag(ds)):.6f}, max={np.max(np.imag(ds)):.6f}")
    print(f"    Magnitude:      mean={np.mean(np.abs(ds)):.6f}, max={np.max(np.abs(ds)):.6f}")
    
    # 6. 可視化
    print("[6] Creating visualizations...")
    
    # Figure 1: シンプル版（実部と虚部）
    fig1 = plot_complex_split(ds, mesh)
    fig1.savefig("option1_simple.png", dpi=150, bbox_inches="tight")
    print("    ✓ Saved: option1_simple.png")
    
    # Figure 2: 統計情報付き版
    fig2 = plot_complex_with_stats(ds, mesh)
    fig2.savefig("option1_with_stats.png", dpi=150, bbox_inches="tight")
    print("    ✓ Saved: option1_with_stats.png")
    
    # 7. 複数周波数での比較
    print("[7] Comparing multiple frequencies...")
    frequencies = [1e3, 1e4, 1e5]
    epsilon_0 = 8.854e-12
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for idx, freq in enumerate(frequencies):
        # 周波数依存の複素誘電率
        omega = 2 * np.pi * freq
        relative_perm = 100
        epsilon = epsilon_0 * relative_perm
        sigma_freq = 0.5 + 1j * omega * epsilon
        
        # メッシュ更新
        mesh.perm = np.ones(mesh.n_elems) * sigma_freq
        
        # 参照・摂動状態を計測
        fwd = EITForward(mesh, protocol)
        v_ref_freq = fwd.solve_eit()
        
        perm_perturb_freq = np.ones(mesh.n_elems) * sigma_freq
        perm_perturb_freq[:8] = sigma_freq * 1.5
        v_perturb_freq = fwd.solve_eit(perm=perm_perturb_freq)
        
        # 逆問題解法
        solver = JAC(mesh, protocol)
        solver.setup(perm=sigma_freq * np.ones(mesh.n_elems))
        ds_freq = solver.solve_gs(v_perturb_freq, v_ref_freq)
        
        # プロット
        nodes = mesh.node
        elements = mesh.element
        x = nodes[:, 0]
        y = nodes[:, 1]
        
        pc = axes[idx].tripcolor(x, y, elements, np.real(ds_freq), cmap="RdBu_r")
        axes[idx].set_aspect("equal")
        axes[idx].set_title(f"{freq/1e3:.0f} kHz", fontsize=11, fontweight="bold")
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
        plt.colorbar(pc, ax=axes[idx], label="Re(Δσ)")
        
        print(f"    {freq/1e3:.0f}kHz: Re(ds)_mean={np.mean(np.real(ds_freq)):.6f}")
    
    fig.suptitle("Multi-frequency Comparison (Real Parts)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig("option1_multifrequency.png", dpi=150, bbox_inches="tight")
    print("    ✓ Saved: option1_multifrequency.png")
    
    print("\n" + "="*70)
    print("✓ All visualizations completed!")
    print("="*70)
    
    plt.show()


if __name__ == "__main__":
    main()
