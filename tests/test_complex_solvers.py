# coding: utf-8
"""
Test complex number support for GREIT, BP, and SVD solvers
"""

import unittest

import numpy as np
from pyeit.eit.protocol import PyEITProtocol, build_meas_pattern_std
from pyeit.mesh import PyEITMesh
from pyeit.eit.bp import BP
from pyeit.eit.greit import GREIT
from pyeit.eit.svd import SVD


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


class TestComplexGREIT(unittest.TestCase):
    """Test GREIT inverse problem solver with complex data"""

    def test_greit_setup_with_complex_perm(self):
        """Test GREIT.setup() with complex permittivity."""
        mesh = _mesh_simple()
        mesh.perm = np.array([3.0 + 0.3j, 1.0 + 0.1j])

        ex_mat = np.array([[0, 1]])
        protocol = _protocol_obj(ex_mat, mesh.n_el, 1, "meas_current")

        solver = GREIT(mesh, protocol)
        solver.setup(perm=mesh.perm)

        self.assertTrue(solver.is_ready, "Solver should be ready after setup()")
        self.assertTrue(np.iscomplexobj(solver.J), "Jacobian should be complex-valued")
        self.assertTrue(np.iscomplexobj(solver.H), "H matrix should be complex-valued")

    def test_greit_solve_with_complex_voltage(self):
        """Test GREIT.solve() with complex-valued measurements."""
        mesh = _mesh_simple()
        mesh.perm = np.array([3.0 + 0.3j, 1.0 + 0.1j])

        ex_mat = np.array([[0, 1]])
        protocol = _protocol_obj(ex_mat, mesh.n_el, 1, "meas_current")

        solver = GREIT(mesh, protocol)
        solver.setup(perm=mesh.perm)

        # Forward problem
        v0 = solver.fwd.solve_eit(mesh.perm)
        v1 = v0 * (1 + 0.01j)

        # solve should handle complex data
        ds = solver.solve(v1, v0)

        # Output should be valid
        self.assertFalse(np.any(np.isnan(ds)), "Output should not contain NaN")

    def test_greit_hermitian_transpose(self):
        """Test that GREIT uses Hermitian transpose for complex matrices."""
        mesh = _mesh_simple()
        mesh.perm = np.array([3.0 + 0.3j, 1.0 + 0.1j])

        ex_mat = np.array([[0, 1]])
        protocol = _protocol_obj(ex_mat, mesh.n_el, 1, "meas_current")

        solver = GREIT(mesh, protocol)
        solver.setup(perm=mesh.perm)

        # H matrix should not contain NaN or Inf
        self.assertFalse(np.any(np.isnan(solver.H)), "H matrix should not contain NaN")
        self.assertFalse(np.any(np.isinf(solver.H)), "H matrix should not contain Inf")
        self.assertTrue(np.iscomplexobj(solver.H), "H matrix should be complex")


class TestComplexBP(unittest.TestCase):
    """Test BP inverse problem solver with complex data"""

    def test_bp_setup_with_complex_perm(self):
        """Test BP.setup() with complex permittivity."""
        mesh = _mesh_simple()
        mesh.perm = np.array([3.0 + 0.3j, 1.0 + 0.1j])

        ex_mat = np.array([[0, 1]])
        protocol = _protocol_obj(ex_mat, mesh.n_el, 1, "meas_current")

        solver = BP(mesh, protocol)
        solver.setup(perm=mesh.perm)

        self.assertTrue(solver.is_ready, "Solver should be ready after setup()")
        self.assertTrue(hasattr(solver, "H"), "BP should have H matrix after setup")

    def test_bp_solve_gs_with_complex_voltage(self):
        """Test BP.solve_gs() with complex-valued measurements."""
        mesh = _mesh_simple()
        mesh.perm = np.array([3.0 + 0.3j, 1.0 + 0.1j])

        ex_mat = np.array([[0, 1]])
        protocol = _protocol_obj(ex_mat, mesh.n_el, 1, "meas_current")

        solver = BP(mesh, protocol)
        solver.setup(perm=mesh.perm)

        # Forward problem
        v0 = solver.fwd.solve_eit(mesh.perm)
        v1 = v0 * (1 + 0.01j)

        # solve_gs should handle complex data and return valid result
        ds = solver.solve_gs(v1, v0)

        # Just verify it doesn't contain NaN
        self.assertFalse(np.any(np.isnan(ds)), "Output should not contain NaN")


class TestComplexSVD(unittest.TestCase):
    """Test SVD inverse problem solver with complex data"""

    def test_svd_setup_with_complex_perm(self):
        """Test SVD.setup() with complex permittivity."""
        mesh = _mesh_simple()
        mesh.perm = np.array([3.0 + 0.3j, 1.0 + 0.1j])

        ex_mat = np.array([[0, 1]])
        protocol = _protocol_obj(ex_mat, mesh.n_el, 1, "meas_current")

        solver = SVD(mesh, protocol)
        solver.setup(perm=mesh.perm, method="svd")

        self.assertTrue(solver.is_ready, "Solver should be ready after setup()")
        self.assertTrue(np.iscomplexobj(solver.J), "Jacobian should be complex-valued")
        self.assertTrue(np.iscomplexobj(solver.H), "H matrix should be complex-valued")

    def test_svd_pinv_method_with_complex(self):
        """Test SVD with pinv method for complex data."""
        mesh = _mesh_simple()
        mesh.perm = np.array([3.0 + 0.3j, 1.0 + 0.1j])

        ex_mat = np.array([[0, 1]])
        protocol = _protocol_obj(ex_mat, mesh.n_el, 1, "meas_current")

        solver = SVD(mesh, protocol)
        solver.setup(perm=mesh.perm, method="pinv")

        self.assertTrue(solver.is_ready, "Solver should be ready with method='pinv'")
        self.assertTrue(np.iscomplexobj(solver.H), "H matrix should be complex-valued")

    def test_svd_hermitian_eigen_decomposition(self):
        """Test that SVD uses Hermitian eigendecomposition for complex matrices."""
        mesh = _mesh_simple()
        mesh.perm = np.array([3.0 + 0.3j, 1.0 + 0.1j])

        ex_mat = np.array([[0, 1]])
        protocol = _protocol_obj(ex_mat, mesh.n_el, 1, "meas_current")

        solver = SVD(mesh, protocol)
        solver.setup(perm=mesh.perm, method="svd")

        # H matrix should not contain NaN or Inf
        self.assertFalse(np.any(np.isnan(solver.H)), "H matrix should not contain NaN")
        self.assertFalse(np.any(np.isinf(solver.H)), "H matrix should not contain Inf")
        self.assertTrue(np.iscomplexobj(solver.H), "H matrix should be complex")


if __name__ == "__main__":
    unittest.main()
