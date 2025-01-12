"""
Precision matrix operations for LDGM.

This module implements sparse precision matrix operations using scipy's LinearOperator
interface for efficient matrix-vector operations.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Optional, Tuple, Union

import numpy as np
import polars as pl
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, cg
from sksparse.cholmod import cholesky


@dataclass
class PrecisionOperator(LinearOperator):
    """
    LDGM precision matrix class implementing the LinearOperator interface.

    This class provides an efficient implementation of precision matrix operations
    using the scipy.sparse.linalg.LinearOperator interface. It supports matrix-vector
    multiplication and other essential operations for working with LDGM precision matrices.

    Attributes:
        _matrix: The precision matrix in sparse format
        variant_info: Polars DataFrame containing variant information
        _which_indices: Array of indices for current selection
        _solver: Previously computed Cholesky factorization
        _cholesky_is_up_to_date: Flag indicating whether the Cholesky factorization is up to date
    """
    _matrix: csc_matrix
    variant_info: pl.DataFrame
    _which_indices: Optional[np.ndarray] = None
    _solver: Optional[cholesky] = None
    _cholesky_is_up_to_date: bool = False

    @property
    def shape(self):
        """Get the shape of the matrix, accounting for partial indexing."""
        if self._which_indices is not None:
            n = len(self._which_indices)
            return (n, n)
        return self._matrix.shape

    @property
    def dtype(self) -> np.dtype:
        """Return the dtype of the precision matrix."""
        return self._matrix.dtype

    @property
    def nbytes(self) -> int:
        """Return the total memory usage in bytes.
        """
        # Get size of the sparse matrix
        matrix_size = (self._matrix.data.nbytes +
                      self._matrix.indices.nbytes +
                      self._matrix.indptr.nbytes)

        # Get size of the variant info DataFrame
        variant_info_size = self.variant_info.estimated_size()

        # Get size of the Cholesky factor if it exists
        factor_size = 0
        if self._solver is not None:
            L = self._solver.L()
            factor_size = (L.data.nbytes +
                         L.indices.nbytes +
                         L.indptr.nbytes)

        # Get size of other attributes
        other_size = 0 if self._which_indices is None else self._which_indices.nbytes

        return matrix_size + variant_info_size + factor_size + other_size

    @property
    def matrix(self):
        """Get the precision matrix."""
        return self._matrix

    @property
    def variant_indices(self):
        """Get the indices of the variants in the precision matrix."""
        return self.variant_info.select('index').to_numpy().flatten()

    @cached_property
    def diagonal_indices(self) -> np.ndarray:
        """Get indices of diagonal elements corresponding to _which_indices in _matrix.data.

        Returns:
            Array of indices into self._matrix.data where diagonal elements are stored
        """
        # Find positions of diagonal elements in the sparse matrix
        diag_indices = []
        for i in range(self._matrix.shape[0]):
            start = self._matrix.indptr[i]
            end = self._matrix.indptr[i + 1]
            diag_pos = start + np.where(self._matrix.indices[start:end] == i)[0][0]
            diag_indices.append(diag_pos)
        return np.array(diag_indices)[self._get_mask]


    def times_scalar(self, multiplier: float) -> None:
        """Multiply the precision matrix by a scalar in place."""
        self._matrix *= multiplier
        self.del_solver()

    def update_matrix(self, update: np.ndarray) -> None:
        """Update the precision matrix by adding values to its diagonal.

        If which_indices is set, only updates the corresponding diagonal elements.

        Args:
            update: Vector of values to add to the diagonal elements

        Raises:
            ValueError: If update shape doesn't match or would make diagonal non-positive
        """
        if len(update) != self.shape[0]:
            msg = f"Update vector length {len(update)} does not match matrix shape {self.shape}"
            raise ValueError(msg)

        if np.allclose(update, 0):
            return

        for idx, entry in zip(self.diagonal_indices, update, strict=False):
            self._matrix.data[idx] += entry

        if np.any(self._matrix.data[self.diagonal_indices] <= 0):
            raise ValueError("Update would make a diagonal element non-positive")

        self._cholesky_is_up_to_date = False

    def update_element(self, index: int, value: float) -> None:
        """Update a single diagonal element of the precision matrix.

        If which_indices is set, only updates the corresponding element.

        Args:
            index: The updated diagonal element is self._which_indices[index]
            value: Value to add to the diagonal element

        Note:
            Updates the Cholesky factorization if it exists using a rank-1 update/downdate.
        """
        # Find diagonal element in sparse matrix
        diag_pos = self.diagonal_indices[index]

        # Convert to global index for Cholesky update
        global_index = index if self._which_indices is None else self._which_indices[index]

        # Update matrix value
        new_val = self._matrix.data[diag_pos] + value
        if new_val <= 0:
            msg = f"Update would make diagonal element at index {global_index} non-positive"
            raise ValueError(msg)

        self._matrix.data[diag_pos] = new_val

        if not self._cholesky_is_up_to_date:
            return

        # Update Cholesky factorization
        shape = (self._matrix.shape[0], 1)
        e_sparse = csc_matrix(([np.sqrt(np.abs(value))], ([global_index], [0])), shape=shape)
        self._solver.update_inplace(e_sparse, value < 0)

    def factor(self) -> None:
        """Update the Cholesky factorization of the precision matrix."""
        if self._solver is None:
            self._solver = cholesky(self._matrix)
        else:
            self._solver.cholesky_inplace(self._matrix)
        self._cholesky_is_up_to_date = True

    def del_factor(self) -> None:
        """Free the memory used by the Cholesky factorization."""
        self._solver = None
        self._cholesky_is_up_to_date = False

    def logdet(self) -> float:
        """Compute log determinant of the Schur complement.

        Returns:
            Log determinant of the Schur complement
        """
        if not self._cholesky_is_up_to_date:
            self.factor()

        # Get boolean mask for current selection
        mask = self._get_mask
        if np.all(mask):
            return self._solver.logdet()

        # Get P11 submatrix for missing indices
        P11 = self._matrix[~mask][:, ~mask]
        logdet_P11 = cholesky(P11).logdet()

        # Return logdet(P) - logdet(P11)
        return self._solver.logdet() - logdet_P11

    @cached_property
    def _get_mask(self) -> np.ndarray:
        """Get boolean mask for current selection."""
        if self._which_indices is None:
            return np.ones(self.shape[0], dtype=bool)

        # Convert key to indices array
        mask = np.zeros(self._matrix.shape[0], dtype=bool)
        mask[self._which_indices] = True
        return mask

    def _expand_vector(self, b: np.ndarray) -> np.ndarray:
        """Expand input vector to full size by padding with zeros.

        Args:
            b: Input vector

        Returns:
            Expanded vector

        Raises:
            ValueError: If input vector dimensions don't match matrix shape
        """
        b = np.asarray(b)
        if b.ndim == 1:
            b = b.reshape(-1, 1)

        if self._which_indices is None:
            if b.shape[0] != self._matrix.shape[0]:
                msg = f"Input vector has shape {b.shape}, but matrix has shape {self._matrix.shape}"
                raise ValueError(msg)
            return b

        # Convert indices to array if needed
        expected_size = len(self._which_indices)

        if b.shape[0] != expected_size:
            raise ValueError(f"Input vector has shape {b.shape}, but expected size {expected_size}")

        y = np.zeros((self._matrix.shape[0], b.shape[1]), dtype=b.dtype)
        y[self._which_indices, :] = b
        return y

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        """Schur complement matrix-vector multiplication.

        Args:
            x: Vector to multiply with precision matrix

        Returns:
            Result of matrix-vector multiplication
        """
        # Handle partial matrix multiplication
        expected_size = self.shape[0]
        if x.shape[0] != expected_size:
            raise ValueError(f"Input vector has shape {x.shape}, but expected size {expected_size}")

        if self._which_indices is None:
            return self._matrix @ x

        # Schur complement (P/P11) * x
        mask = self._get_mask
        first_term = self._matrix[mask][:, mask] @ x
        second_term = self._matrix[~mask][:, mask] @ x
        P11 = self._matrix[~mask][:, ~mask]
        second_term = cholesky(P11)(second_term)
        second_term = self._matrix[mask][:, ~mask] @ second_term
        return first_term - second_term

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        """Transposed Schur complement matrix-vector multiplication.

        Args:
            x: Vector to multiply with transposed precision matrix

        Returns:
            Result of transposed matrix-vector multiplication
        """
        # Since our matrix is symmetric, we can reuse _matvec
        return self._matvec(x.T).T if x.ndim > 1 else self._matvec(x)

    def copy(self):
        return PrecisionOperator(self._matrix, self.variant_info, self._which_indices, self._solver,
                               self._cholesky_is_up_to_date)

    def __getitem__(self, key: Union[list, slice, np.ndarray]) -> 'PrecisionOperator':
        """Sets the _which_indices class attribute.

        Args:
            key: Index, slice, or tuple of indices/slices to access. Can be:
                - List of indices
                - Array of indices
                - Boolean mask array
                - Slice object

        Returns:
            New LDGM instance with updated _which_indices
        """
        result = self.copy()
        result.set_which_indices(key)
        return result


    def set_which_indices(self, key: Union[list, slice, np.ndarray]) -> None:
        # Convert key to indices array
        if isinstance(key, slice):
            indices = np.arange(self._matrix.shape[0])[key]
        elif isinstance(key, (list, np.ndarray)):
            key = np.asarray(key)  # Convert list to array first
            if key.dtype == bool:
                indices = np.where(key)[0]  # Convert boolean mask to integer indices
            else:
                indices = key
        else:
            raise TypeError("Invalid key type. Use list, slice, or numpy array.")
        
        self._which_indices = indices

        
    def solve(self,
        b: np.ndarray,
        method: str = "direct",
        tol: float = 1e-5,
        callback: Optional[Callable] = None,
        initialization: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Solve the linear system Px = b.

        Args:
            b: Right-hand side vector or matrix
            method: Solver method ('direct' or 'pcg')
            tol: Tolerance for PCG solver
            callback: Optional callback for PCG solver
            initialization: Optional initial guess for pcg

        Returns:
            Solution vector x
        """
        y = self._expand_vector(b)

        if method == "direct":
            if not self._cholesky_is_up_to_date:
                self.factor()
            solution = self._solver(y)
        elif method == "pcg":
            if self._solver is None:
                self.factor()  # Some factorization is needed for use as a preconditioner
            if self._cholesky_is_up_to_date:
                solution = self._solver(y)
            else:
                solution = self._pcg(y, tol=tol, callback=callback, initialization=initialization)
        else:
            raise ValueError("Method must be either 'direct' or 'pcg'")

        if self._which_indices is not None:
            solution = solution[self._which_indices, :]

        return solution.reshape(b.shape)

    def _pcg(self,
            b: np.ndarray,
            tol: float = 1e-5,
            callback: Optional[Callable] = None,
            initialization: Optional[np.ndarray] = None
            ) -> np.ndarray:

        solution = np.zeros_like(b)
        for i in range(b.shape[1]):
            # Use conjugate gradient for each right-hand side
            x0 = initialization[:,i] if initialization is not None else None

            preconditioner = LinearOperator((b.shape[0], b.shape[0]), matvec=self._solver)
            x, info = cg(self, b[:, i], rtol=tol, callback=callback, x0=x0, M=preconditioner)
            if info != 0:
                raise RuntimeError(f"Conjugate gradient failed to converge for column {i}")
            solution[:, i] = x

        return solution

    def solve_Lt(self, b: np.ndarray) -> np.ndarray:
        """Solve the triangular system L'x = b, where LL' = _matrix.
        If b ~ MVN(0, I), then x ~ MVN(0, R).

        Args:
            b: Right-hand side vector

        Returns:
            Solution vector x
        """
        if not self._cholesky_is_up_to_date:
            self.factor()

        y = self._expand_vector(b)
        solution = self._solver.solve_Lt(y, use_LDLt_decomposition=False)

        if self._which_indices is not None:
            solution = solution[self._which_indices, :]
        return solution.reshape(b.shape)

    def variant_solve(self, b: np.ndarray) -> np.ndarray:
        """Computes correlation_matrix @ b where the dimension is number of variants, not number of indices.
        If two variants i,j have the same index (and are in perfect LD), b[i] and b[j] are summed, then
        solve() is called, then the solution vector is assigned the same values in entries i,j.

        Args:
            b: Right-hand side vector

        Returns:
            Solution vector z
        """
        if len(b) != len(self.variant_info):
            raise ValueError("b must have the same length as the number of variants")

        y = np.zeros(self.shape[0])
        np.add.at(y, self.variant_indices, b)
        
        y = self.solve(y)
        
        return y[self.variant_indices]
        

    def inverse_diagonal(
        self,
        method: str = "xdiag",
        initialization: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        n_samples: int = 100,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Compute the diagonal elements of the inverse of the precision matrix.

        Args:
            initialization: Optional tuple of matrices containing:
                - Initial guess for the diagonal elements
                - Initial guess for the off-diagonal elements
            method: Method to use for computing diagonal elements
                ('exact', 'hutchinson', or 'xdiag')
            n_samples: Number of probe vectors for Hutchinson's method or xdiag
            seed: Random seed for generating probe vectors

        Returns:
            Array of diagonal elements of the inverse
        """
        if method not in ["exact", "hutchinson", "xnys", "xdiag"]:
            raise ValueError(f"Unknown method: {method}")

        # Slow, exact inversion
        if method.lower() == "exact":
            if initialization is not None:
                raise ValueError("Initialization not supported for exact method")

            dense_matrix = self._matrix.toarray()
            inv_matrix = np.linalg.inv(dense_matrix)
            diag = np.diag(inv_matrix)

            return diag if self._which_indices is None else diag[self._which_indices]

        # Use a stochastic estimator
        if initialization is not None:
            v, pv = initialization
        else:
            rng = np.random.RandomState(seed)

            # Rademacher random vectors
            v = rng.choice([-1.0, 1.0], size=(self.shape[0], min(self.shape[0], n_samples)))
            pv = v.copy()

        if method.lower() == "hutchinson":
            # Solve M * y = v for each probe vector
            y = self.solve(v, method="pcg", initialization=pv)

            # Estimate diagonal elements as average element-wise product of v_i * y_i
            diag_estimate = np.mean(v * y, axis=1)

        elif method.lower() == "xdiag":
            diag_estimate, y = self._xdiag_estimator(v, initialization=pv)

        else:
            raise NotImplementedError

        # # If indices are specified, return only those elements
        # if self._which_indices is not None:
        #     diag_estimate = diag_estimate[self._which_indices]
        #     if initialization is not None:
        #         y = y[self._which_indices]

        return (diag_estimate, y) if initialization is not None else diag_estimate

    def _xdiag_estimator(self,
                        v: np.ndarray,
                        initialization: Optional[np.ndarray] = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Compute diagonal elements of the inverse using the xdiag method.
        This is a Python implementation of the MATLAB xdiag.m function
        described at https://arxiv.org/abs/2301.07825 and implemented
        at https://github.com/eepperly/XTrace/blob/main/code/xdiag.m

        Args:
            v: Matrix of probe vectors (should be Rademacher random variables)
            initialization: Initial values for self.solve(v)

        Returns:
            Tuple containing:
                - Array of diagonal elements of the inverse
                - Updated values for P \ v
        """
        n, m = v.shape

        # Y = A @ v, A = inv(self)
        Y = self.solve(v, method="pcg", initialization=initialization)

        # QR decomposition of Y
        Q, R = np.linalg.qr(Y, mode='reduced')

        # Z = A @ Q
        Z = self.solve(Q, method="pcg")
        T = Z.T @ v

        # Column-normalize inverse of R transpose
        invR = np.linalg.inv(R).T
        S = invR / np.linalg.norm(invR, axis=0, keepdims=True)

        # Compute diagonal products efficiently
        dQZ = np.sum(Q.conj() * Z, axis=1)  # diag(Q.H @ Z)
        dQSSZ = np.sum((Q @ S).conj() * (Z @ S), axis=1)  # diag((Q @ S).H @ (Z @ S))

        # For dOmQT, we need diag(v.H @ (Q @ T))
        dOmQT = np.sum(v.conj() * (Q @ T), axis=1)
        dOmY = np.sum(v.conj() * Y, axis=1)  # diag(v.H @ Y)

        # Compute S @ diag(S.H @ T)
        ST_diag = np.sum(S.conj() * T, axis=1)  # diag(S.H @ T)

        # Compute dOmQSST = diag(v.H @ (Q @ S @ diag(S.H @ T)))
        dOmQSST = np.sum(v.conj() * (Q @ S @ np.diag(ST_diag)), axis=1)

        # Final diagonal estimate
        diag_est = dQZ + (-dQSSZ + dOmY - dOmQT + dOmQSST) / m

        return np.real(diag_est), Y
