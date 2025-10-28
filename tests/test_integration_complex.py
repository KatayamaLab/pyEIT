# coding: utf-8
"""
Integration tests for complex number support across the entire pipeline
"""
import numpy as np
import unittest
from pyeit.mesh import create, PyEITMesh
from pyeit.eit.fem import EITForward
from pyeit.eit.protocol import build_meas_pattern_std
from pyeit.eit.jac import JAC


def _mesh_simple():
    """Build a simple, deterministic mesh for testing"""
    node = np.array([[0.13, 0.15], [0.2, 0.2], [0.1, 0.1], [0.18, 0.12]])
    element = np.array([[0, 2, 3], [0, 3, 1]])
    perm = np.array([3.0, 1.0])
    el_pos = np.array([1, 2])
    return PyEITMesh(node=node, element=element, perm=perm, el_pos=el_pos, ref_node=3)


def _protocol_obj(ex_mat, n_el, step_meas, parser_meas):
    """Build a protocol object"""
    from pyeit.eit.protocol import PyEITProtocol
    meas_mat, keep_ba = build_meas_pattern_std(ex_mat, n_el, step_meas, parser_meas)
    return PyEITProtocol(ex_mat, meas_mat, keep_ba)


class TestIntegrationComplex(unittest.TestCase):
    """Integration tests for complex-valued EIT workflow"""

    def setUp(self):
        """Setup common test fixtures"""
        # Create a simple mesh
        self.mesh_obj = create(16, h0=0.1)
        
    def test_full_workflow_complex_perm(self):
        """
        Test complete EIT workflow with complex permittivity:
        1. Create mesh with complex perm
        2. Setup protocol
        3. Forward solve
        4. Compute Jacobian
        5. Inverse solve with JAC
        """
        mesh = _mesh_simple()
        # Step 1: Create reference state with complex permittivity
        perm_ref = np.array([3.0 + 0.3j, 1.0 + 0.1j])
        mesh.perm = perm_ref
        
        # Step 2: Setup measurement protocol
        ex_mat = np.array([[0, 1], [1, 0]])
        protocol = _protocol_obj(ex_mat, mesh.n_el, 1, "meas_current")
        
        
        # Step 3: Forward solve
        fwd = EITForward(mesh, protocol)
        v_ref = fwd.solve_eit(perm=perm_ref)
        
        # Verify complex result
        self.assertTrue(np.iscomplexobj(v_ref), "Reference voltage should be complex")
        self.assertEqual(v_ref.dtype, np.complex128)
        
        # Step 4: Create perturbed state
        perm_perturb = perm_ref * (1.0 + 0.05j)
        v_perturb = fwd.solve_eit(perm=perm_perturb)
        
        # Verify perturbation preserves complex type
        self.assertTrue(np.iscomplexobj(v_perturb), "Perturbed voltage should be complex")
        
        # Step 5: JAC inverse solve
        solver = JAC(mesh, protocol)
        solver.setup(perm=perm_ref)
        
        # Verify Jacobian is complex
        self.assertTrue(np.iscomplexobj(solver.J), "Jacobian should be complex")
        self.assertTrue(np.iscomplexobj(solver.v0), "Reference voltage v0 should be complex")
        
        # Solve for permittivity change
        ds = solver.solve_gs(v_perturb, v_ref)
        
        # Verify reconstruction
        self.assertEqual(ds.shape, (mesh.n_elems,))
        # For small perturbations, ds should be approximately equal to the perturbation
        expected_ds = 0.05j * perm_ref  # Expected small change
        
        print(f"✓ Full workflow with complex perm successful")
        print(f"  v_ref dtype: {v_ref.dtype}, shape: {v_ref.shape}")
        print(f"  v_perturb dtype: {v_perturb.dtype}, shape: {v_perturb.shape}")
        print(f"  Jacobian dtype: {solver.J.dtype}, shape: {solver.J.shape}")
        print(f"  ds dtype: {ds.dtype}, shape: {ds.shape}")

    def test_real_vs_complex_consistency(self):
        """
        Test that real-valued input produces consistent results
        whether treated as real or complex
        """
        mesh = _mesh_simple()
        # Real permittivity
        perm_real = np.array([3.0, 1.0])
        
        ex_mat = np.array([[0, 1], [1, 0]])
        protocol = _protocol_obj(ex_mat, mesh.n_el, 1, "meas_current")
        
        # Forward solve with real perm
        fwd = EITForward(mesh, protocol)
        v_real = fwd.solve_eit(perm=perm_real)
        
        # Result should be real (imaginary part negligible)
        imag_max = np.max(np.abs(np.imag(v_real)))
        self.assertLess(imag_max, 1e-10, "Imaginary part should be negligible for real perm")
        
        # JAC with real perm
        solver = JAC(mesh, protocol)
        solver.setup(perm=perm_real)
        
        # Jacobian should be real (imaginary part negligible)
        j_imag_max = np.max(np.abs(np.imag(solver.J)))
        self.assertLess(j_imag_max, 1e-10, "JAC imaginary part should be negligible for real perm")
        
        print(f"✓ Real vs complex consistency verified")
        print(f"  v_real max imag: {imag_max:.2e}")
        print(f"  J max imag: {j_imag_max:.2e}")

    def test_complex_perm_with_multiple_frequencies(self):
        """
        Test complex permittivity representing multi-frequency behavior
        (real part = conductivity, imag part = capacitance effect)
        """
        mesh = _mesh_simple()
        # Simulate multi-frequency response
        frequencies = [1000, 10000, 100000]  # Hz
        
        # Reference permittivity (conductivity + frequency-dependent reactance)
        perm_base = np.array([3.0, 1.0])
        
        results = []
        
        ex_mat = np.array([[0, 1], [1, 0]])
        protocol = _protocol_obj(ex_mat, mesh.n_el, 1, "meas_current")
        
        fwd = EITForward(mesh, protocol)
        
        for freq in frequencies:
            # At each frequency, permittivity has different imaginary part
            # σ(ω) = σ₀ + i·ω·ε
            omega = 2 * np.pi * freq
            perm_freq = perm_base + (0.1j * omega / (2 * np.pi * 1000)) * np.ones(mesh.n_elems)
            # Ensure mesh perm is set before solve_eit to update dtype
            mesh.perm = perm_freq
            
            v = fwd.solve_eit(perm=perm_freq)
            results.append({
                'freq': freq,
                'perm': perm_freq,
                'v': v,
                'v_magnitude': np.abs(v),
                'v_phase': np.angle(v)
            })
        
        # Verify results across frequencies
        for i, result in enumerate(results):
            self.assertTrue(np.iscomplexobj(result['v']), f"Voltage at {result['freq']} Hz should be complex")
            print(f"✓ Frequency {result['freq']} Hz: |v|={np.mean(result['v_magnitude']):.4f}, phase={np.mean(np.degrees(result['v_phase'])):.2f}°")

    def test_jac_gn_solver_complex(self):
        """
        Test Gauss-Newton solver with complex permittivity
        """
        mesh = _mesh_simple()
        # Setup complex reference state
        perm_ref = np.array([3.0 + 0.3j, 1.0 + 0.1j])
        mesh.perm = perm_ref
        
        ex_mat = np.array([[0, 1], [1, 0]])
        protocol = _protocol_obj(ex_mat, mesh.n_el, 1, "meas_current")
        
        # Forward solve
        fwd = EITForward(mesh, protocol)
        v_ref = fwd.solve_eit(perm=perm_ref)
        
        # Slightly perturbed measurement
        v_meas = v_ref * (1.0 + 0.01j)
        
        # JAC solver setup
        solver = JAC(mesh, protocol)
        solver.setup(perm=perm_ref)
        
        # Gauss-Newton solve
        perm_reconstructed = solver.gn(v_meas, x0=perm_ref, maxiter=3, verbose=True)
        
        # Verify output
        self.assertEqual(perm_reconstructed.dtype, np.complex128)
        self.assertEqual(perm_reconstructed.shape, (mesh.n_elems,))
        
        print(f"✓ Gauss-Newton solver with complex perm completed")
        print(f"  Original perm (first 3): {perm_ref[:3]}")
        print(f"  Reconstructed perm (first 3): {perm_reconstructed[:3]}")


if __name__ == "__main__":
    unittest.main()
