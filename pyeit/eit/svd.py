# coding: utf-8
# pylint: disable=invalid-name, no-member, arguments-differ
""" dynamic EIT solver using SVD """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function, annotations

from typing import Union, Optional
import numpy as np
from .jac import JAC


class SVD(JAC):
    """implementing a sensitivity-based EIT imaging class"""

    def setup(  # type: ignore[override]
        self,
        n: int = 25,
        rcond: float = 1e-2,
        method: str = "svd",
        perm: Optional[Union[int, float, complex, np.ndarray]] = None,
        jac_normalized: bool = False,
    ) -> None:
        """
        Setup of SVD solver, singular value decomposition based reconstruction.

        Parameters
        ----------
        n : int, optional
            largest n eigenvalues to be kept, by default 25
        rcond : float, optional
            r-condition number of pinv, by default 1e-2
        method : str, optional
            reconstruction method, by default "svd"
            'svd': SVD truncation,
            'pinv': pseudo inverse
        perm : Union[int, float, np.ndarray], optional
            If perm is not None, a prior of perm distribution is used to build jac
        jac_normalized : bool, optional
            normalize the jacobian using f0 computed from input perm, by
            default False
        """
        # correct n_ord
        self.J, self.v0 = self.fwd.compute_jac(perm=perm, normalize=jac_normalized)
        nm, ne = self.J.shape
        n_ord = np.min([nm, ne, n])

        # passing imaging parameters
        self.params = {"n": n_ord, "rcond": rcond, "method": method}

        # pre-compute H0 for dynamical imaging
        if method == "pinv":
            self.H = np.linalg.pinv(self.J, rcond=rcond)

        elif method == "svd":
            # Use Hermitian transpose for complex matrices
            j_conj_t = self.J.conj().T if np.iscomplexobj(self.J) else self.J.T
            JtJ = np.dot(j_conj_t, self.J)

            # using svd
            # U, s, Ut = np.linalg.svd(JtJ)
            # U = U[:, :n_ord]
            # s = s[:n_ord]

            # using eigh (more faster for large, symmetric matrix)
            s, U = np.linalg.eigh(JtJ)
            idx = np.argsort(s)[::-1]
            s = s[idx[:n_ord]]
            U = U[:, idx[:n_ord]]

            # pseudo inverse with regularization for small/zero eigenvalues
            # Avoid division by zero for very small eigenvalues
            s_inv = np.zeros_like(s, dtype=np.complex128 if np.iscomplexobj(s) else float)
            for i, s_i in enumerate(s):
                if np.abs(s_i) > 1e-10:
                    s_inv[i] = 1.0 / s_i
            
            u_conj_t = U.conj().T if np.iscomplexobj(U) else U.T
            JtJ_inv = np.dot(U, np.dot(np.diag(s_inv), u_conj_t))
            self.H = np.dot(JtJ_inv, j_conj_t)
        self.is_ready = True

    def gn(self):
        """deactivate gn"""
        raise NotImplementedError()

    def solve_gs(self):
        """deactivate solve_gs"""
        raise NotImplementedError()

    def jt_solve(self):
        """deactivate jt_solve"""
        raise NotImplementedError()
