# coding: utf-8
# Test for complex number support in pyEIT
"""
Unit tests for complex number support in pyEIT

This test suite validates that complex-valued permittivity and measurements
are properly handled throughout the EIT pipeline.

Key test cases:
1. FEM with complex permittivity
2. Complex-valued potential distribution
3. Complex Jacobian matrices
4. Inverse problem solving with complex data
"""

import unittest

import numpy as np
import pyeit.eit.fem
from pyeit.eit.protocol import PyEITProtocol, build_meas_pattern_std
from pyeit.mesh import PyEITMesh
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
    meas_mat, keep_ba = build_meas_pattern_std(ex_mat, n_el, step_meas, parser_meas)
    return PyEITProtocol(ex_mat, meas_mat, keep_ba)


class TestComplexFEM(unittest.TestCase):
    """Test FEM layer with complex permittivity"""

    def test_complex_perm_dtype_preservation(self):
        """
        Test that complex permittivity is properly recognized and dtypes are preserved.

        Expected behavior:
        - mesh.dtype should be complex128 when complex perm is provided
        - kg matrix should be complex128
        """
        node = np.array([[0.13, 0.15], [0.2, 0.2], [0.1, 0.1], [0.18, 0.12]])
        element = np.array([[0, 2, 3], [0, 3, 1]])

        # Create mesh with complex permittivity
        perm_complex = np.array([3.0 + 0.5j, 1.0 + 0.2j])
        el_pos = np.array([1, 2])
        mesh = PyEITMesh(
            node=node, element=element, perm=perm_complex, el_pos=el_pos, ref_node=3
        )

        # Check mesh dtype
        self.assertEqual(
            mesh.dtype,
            np.complex128,
            "mesh.dtype should be complex128 for complex perm",
        )

        # Check kg matrix dtype
        fwd = pyeit.eit.fem.Forward(mesh)
        self.assertEqual(fwd.kg.dtype, np.complex128, "kg matrix should be complex128")

    def test_complex_perm_kg_assembly(self):
        """
        Test that kg matrix assembly handles complex permittivity correctly.

        Expected behavior:
        - kg matrix should be assembled without errors for complex perm
        - kg matrix structure should be valid (sparse matrix format)
        """
        node = np.array([[0, 1], [0, 0], [1, 0]])
        element = np.array([[0, 1, 2]])
        perm_complex = np.array([1.0 + 0.1j])
        el_pos = np.array([0, 1])

        mesh = PyEITMesh(
            node=node, element=element, perm=perm_complex, el_pos=el_pos, ref_node=2
        )
        fwd = pyeit.eit.fem.Forward(mesh)

        # kg should be a valid sparse matrix
        self.assertTrue(hasattr(fwd.kg, "toarray"), "kg should be a sparse matrix")

        # kg_dense should have complex values
        kg_dense = fwd.kg.toarray()
        self.assertTrue(
            np.iscomplexobj(kg_dense), "kg matrix should contain complex values"
        )

    def test_solve_with_complex_perm(self):
        """
        Test Forward.solve() with complex permittivity.

        Expected behavior:
        - solve() should return complex-valued potential distribution
        - no RuntimeError or dtype mismatch
        """
        node = np.array([[0, 1], [0, 0], [1, 0]])
        element = np.array([[0, 1, 2]])
        perm_complex = np.array([1.0 + 0.1j])
        el_pos = np.array([0, 1])

        mesh = PyEITMesh(
            node=node, element=element, perm=perm_complex, el_pos=el_pos, ref_node=2
        )
        fwd = pyeit.eit.fem.Forward(mesh)

        # Solve should not raise errors
        f = fwd.solve(np.array([0, 1]))

        # Result should be complex-valued
        self.assertTrue(
            np.iscomplexobj(f), "Potential should be complex-valued for complex perm"
        )

    def test_solve_vectorized_with_complex_perm(self):
        """
        Test Forward.solve_vectorized() with complex permittivity.

        Expected behavior:
        - solve_vectorized() should handle multiple excitation patterns
        - output should be complex-valued
        """
        node = np.array([[0, 1], [0, 0], [1, 0], [0.5, 0.5]])
        element = np.array([[0, 1, 2], [0, 1, 3]])
        perm_complex = np.array([1.0 + 0.1j, 1.5 + 0.2j])
        el_pos = np.array([0, 1, 2])

        mesh = PyEITMesh(
            node=node, element=element, perm=perm_complex, el_pos=el_pos, ref_node=3
        )
        fwd = pyeit.eit.fem.Forward(mesh)

        # Multiple excitation patterns
        ex_mat = np.array([[0, 1], [1, 2]])
        f_all = fwd.solve_vectorized(ex_mat)

        # Should return (n_exc, n_nodes) array
        self.assertEqual(
            f_all.shape, (2, 4), "Output shape should match (n_exc, n_nodes)"
        )
        self.assertTrue(np.iscomplexobj(f_all), "Output should be complex-valued")


class TestComplexEITForward(unittest.TestCase):
    """Test EITForward layer with complex permittivity"""

    def test_solve_eit_with_complex_perm(self):
        """
        Test EITForward.solve_eit() with complex permittivity.

        Expected behavior:
        - solve_eit() should return boundary voltages as complex array
        - measurements should preserve complex information
        """
        mesh = _mesh_simple()

        # Replace perm with complex values
        mesh.perm = np.array([3.0 + 0.3j, 1.0 + 0.1j])

        ex_mat = np.array([[0, 1], [1, 0]])
        protocol = _protocol_obj(ex_mat, mesh.n_el, 1, "meas_current")
        fwd = pyeit.eit.fem.EITForward(mesh, protocol)

        # solve_eit should return complex voltages
        v = fwd.solve_eit()

        self.assertTrue(np.iscomplexobj(v), "EIT measurements should be complex-valued")

    def test_compute_jac_with_complex_perm(self):
        """
        Test EITForward.compute_jac() with complex permittivity.

        Expected behavior:
        - compute_jac() should return complex Jacobian matrix
        - reference voltage v0 should be complex-valued
        """
        mesh = _mesh_simple()
        mesh.perm = np.array([3.0 + 0.3j, 1.0 + 0.1j])

        ex_mat = np.array([[0, 1]])
        protocol = _protocol_obj(ex_mat, mesh.n_el, 1, "meas_current")
        fwd = pyeit.eit.fem.EITForward(mesh, protocol)

        # compute_jac should handle complex perm
        jac, v0 = fwd.compute_jac()

        self.assertTrue(np.iscomplexobj(jac), "Jacobian should be complex-valued")
        self.assertTrue(
            np.iscomplexobj(v0), "Reference voltage should be complex-valued"
        )


class TestComplexJAC(unittest.TestCase):
    """Test JAC inverse problem solver with complex data"""

    def test_jac_setup_with_complex_perm(self):
        """
        Test JAC.setup() with complex permittivity.

        Expected behavior:
        - setup() should initialize without errors
        - H matrix should be usable for subsequent inversions
        """
        mesh = _mesh_simple()
        mesh.perm = np.array([3.0 + 0.3j, 1.0 + 0.1j])

        ex_mat = np.array([[0, 1]])
        protocol = _protocol_obj(ex_mat, mesh.n_el, 1, "meas_current")

        solver = JAC(mesh, protocol)
        solver.setup(perm=mesh.perm)

        self.assertTrue(solver.is_ready, "Solver should be ready after setup()")
        self.assertTrue(np.iscomplexobj(solver.J), "Jacobian should be complex-valued")

    def test_jac_solve_gs_with_complex_voltage(self):
        """
        Test JAC.solve_gs() with complex-valued measurements.

        Expected behavior:
        - solve_gs() should handle complex v1 and v0
        - output should be a valid conductivity change estimate
        """
        mesh = _mesh_simple()
        mesh.perm = np.array([3.0 + 0.3j, 1.0 + 0.1j])

        ex_mat = np.array([[0, 1]])
        protocol = _protocol_obj(ex_mat, mesh.n_el, 1, "meas_current")

        solver = JAC(mesh, protocol)
        solver.setup(perm=mesh.perm)

        # Reference voltage (complex)
        v0 = solver.v0

        # Perturbed voltage (complex with small change)
        v1 = v0 * (1 + 0.1j * 0.01)

        # solve_gs should handle complex data
        ds = solver.solve_gs(v1, v0)

        self.assertEqual(
            ds.shape, (mesh.n_elems,), "Output shape should match n_elements"
        )


class TestComplexDataIntegrity(unittest.TestCase):
    """Test that complex data integrity is maintained through the pipeline"""

    def test_real_perm_maintains_real_solution(self):
        """
        Verify that using real permittivity produces real solutions.

        Expected behavior:
        - For real perm, potential should be real-valued (up to numerical error)
        - Imaginary parts should be negligible
        """
        node = np.array([[0, 1], [0, 0], [1, 0]])
        element = np.array([[0, 1, 2]])
        perm_real = np.array([1.0])
        el_pos = np.array([0, 1])

        mesh = PyEITMesh(
            node=node, element=element, perm=perm_real, el_pos=el_pos, ref_node=2
        )
        fwd = pyeit.eit.fem.Forward(mesh)

        f = fwd.solve(np.array([0, 1]))

        # For real perm, imaginary part should be negligible
        imag_max = np.max(np.abs(np.imag(f)))
        self.assertLess(
            imag_max, 1e-10, "Imaginary part should be negligible for real perm"
        )

    def test_complex_data_round_trip(self):
        """
        Test complex data preservation through forward-backward cycle.

        Expected behavior:
        - Complex perm → solve() → complex potential
        - Measurements should be complex-valued
        """
        mesh = _mesh_simple()

        # Create complex permittivity with significant imaginary part
        perm_complex = np.array([3.0 + 1.0j, 1.0 + 0.5j])
        mesh.perm = perm_complex

        ex_mat = np.array([[0, 1]])
        protocol = _protocol_obj(ex_mat, mesh.n_el, 1, "meas_current")
        fwd = pyeit.eit.fem.EITForward(mesh, protocol)

        # Measurements should be complex
        v = fwd.solve_eit()

        # Check that both real and imaginary parts are non-trivial
        real_max = np.max(np.abs(np.real(v)))
        imag_max = np.max(np.abs(np.imag(v)))

        self.assertGreater(real_max, 1e-10, "Real part should be significant")
        self.assertGreater(imag_max, 1e-10, "Imaginary part should be significant")


class TestComplexDocumentation(unittest.TestCase):
    """Test and document complex number workflows"""

    def test_example_complex_eit_workflow(self):
        """
        Example workflow for complex-valued EIT.

        This test serves as documentation for users on how to use complex numbers.
        """
        # Step 1: Create mesh with complex permittivity (e.g., from impedance data)
        node = np.array([[0, 1], [0, 0], [1, 0], [0.5, 0.5]])
        element = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3]])

        # Complex permittivity: conductivity + (frequency-dependent reactance)
        freq = 1000  # Hz
        perm = np.array(
            [
                1.0 + 0.5j,  # Element 1: Re=1.0, Im=0.5
                0.8 + 0.3j,  # Element 2: Re=0.8, Im=0.3
                1.2 + 0.6j,  # Element 3: Re=1.2, Im=0.6
            ]
        )
        el_pos = np.array([0, 1, 2])

        mesh = PyEITMesh(
            node=node, element=element, perm=perm, el_pos=el_pos, ref_node=3
        )

        # Step 2: Set up measurement protocol
        ex_mat = np.array([[0, 1]])
        protocol = _protocol_obj(ex_mat, mesh.n_el, 1, "meas_current")

        # Step 3: Forward solve with complex data
        fwd = pyeit.eit.fem.EITForward(mesh, protocol)
        v_ref = fwd.solve_eit()

        # Step 4: Perturb and measure
        mesh.perm = perm * (1 + 0.05j)  # Small perturbation
        fwd = pyeit.eit.fem.EITForward(mesh, protocol)
        v_perturb = fwd.solve_eit()

        # Step 5: Inverse problem
        mesh.perm = perm  # Reset to original
        solver = JAC(mesh, protocol)
        solver.setup(perm=perm)

        # Reconstruct permittivity change
        ds = solver.solve_gs(v_perturb, v_ref)

        # Verify reconstruction produces reasonable results
        self.assertEqual(
            ds.shape, (mesh.n_elems,), "Reconstruction should have correct shape"
        )


if __name__ == "__main__":
    unittest.main()
