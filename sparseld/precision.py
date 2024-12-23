"""
Precision matrix operations for LDGM.

This module implements sparse precision matrix operations using scipy's LinearOperator
interface for efficient matrix-vector operations.
"""

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix, spdiags, csc_matrix
from scipy.sparse.linalg import LinearOperator, cg
from typing import Optional, Union, Tuple, Any
from dataclasses import dataclass
from sksparse.cholmod import cholesky
from functools import cached_property

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
    _matrix: csr_matrix
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
    def dtype(self):
        return self._matrix.dtype
    
    @property
    def matrix(self):
        """Get the precision matrix."""
        return self._matrix
    
    @cached_property
    def diagonal_indices(self) -> np.ndarray:
        """Get indices of diagonal elements corresponding to which_indices in the sparse matrix data array.
        
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
    
    def update_matrix(self, update: np.ndarray) -> None:
        """Update the precision matrix by adding values to its diagonal.
        
        If which_indices is set, only updates the corresponding diagonal elements.
        
        Args:
            update: Vector of values to add to the diagonal elements
            
        Raises:
            ValueError: If update shape doesn't match or if it would make a diagonal element non-positive
        """
        if len(update) != self.shape[0]:
            raise ValueError(f"Update vector length {len(update)} does not match matrix shape {self.shape}")
            
        for idx, entry in zip(self.diagonal_indices, update):
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
            raise ValueError(f"Update would make diagonal element at index {global_index} non-positive")
            
        self._matrix.data[diag_pos] = new_val
        
        if not self._cholesky_is_up_to_date:
            return
        
        # Update Cholesky factorization
        e_sparse = csc_matrix(([np.sqrt(np.abs(value))], ([global_index], [0])), shape=(self._matrix.shape[0], 1))
        self._solver.update_inplace(e_sparse, value > 0)
    
    def factor(self) -> None:
        """Update the Cholesky factorization of the precision matrix."""
        self._solver = cholesky(self._matrix)
        self._cholesky_is_up_to_date = True
    
    def logdet(self) -> float:
        """
        Compute log determinant of the Schur complement.
        
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
            
        if isinstance(self._which_indices, np.ndarray) and self._which_indices.dtype == bool:
            if len(self._which_indices) != self._matrix.shape[0]:
                raise ValueError(f"Boolean mask has length {len(self._which_indices)}, but matrix has shape {self._matrix.shape}")
            return self._which_indices
            
        # Convert integer indices to boolean mask
        mask = np.zeros(self._matrix.shape[0], dtype=bool)
        mask[self._which_indices] = True
        return mask

    def _expand_vector(self, b: np.ndarray) -> np.ndarray:
        """
        Expand input vector to full size by padding with zeros.
        
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
                raise ValueError(f"Input vector has shape {b.shape}, but matrix has shape {self._matrix.shape}")
            return b
            
        # Convert indices to array if needed
        expected_size = len(self._which_indices)
            
        if b.shape[0] != expected_size:
            raise ValueError(f"Input vector has shape {b.shape}, but expected size {expected_size}")
            
        y = np.zeros((self._matrix.shape[0], b.shape[1]), dtype=b.dtype)
        y[self._which_indices, :] = b
        return y
    
    def _matvec(self, x: np.ndarray) -> np.ndarray:
        """
        Implement matrix-vector multiplication following MATLAB's precisionMultiply.
        
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
        """
        Implement transposed matrix-vector multiplication.
        
        Args:
            x: Vector to multiply with transposed precision matrix
            
        Returns:
            Result of transposed matrix-vector multiplication
        """
        # Since our matrix is symmetric, we can reuse _matvec
        return self._matvec(x.T).T if x.ndim > 1 else self._matvec(x)
    
    def __getitem__(self, key: Union[np.ndarray, slice, Tuple[Any, Any]]) -> 'PrecisionOperator':
        """
        Sets the _which_indices class attribute.
        
        Args:
            key: Index, slice, or tuple of indices/slices to access. Can be:
                - Integer indices array
                - Boolean mask array
                - Slice object
                
        Returns:
            New LDGM instance with updated _which_indices
        """
        if isinstance(key, tuple):
            raise ValueError("Only single-axis indexing is supported")
        
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
            indices = np.array([key])
            
        # Create new LDGM instance with same matrix but updated indices
        return PrecisionOperator(self._matrix, self.variant_info, indices, self._solver, 
                               self._cholesky_is_up_to_date)
    
    def solve(self, b: np.ndarray, method: str = "direct", tol: float = 1e-5, callback=None) -> np.ndarray:
        """
        Solve linear system Px = b.
        
        Args:
            b: Right-hand side vector or matrix
            method: Solution method ('direct' or 'pcg')
            tol: Tolerance for PCG solver
            callback: Optional callback for PCG solver
            
        Returns:
            Solution vector x
        """
            
        y = self._expand_vector(b)

        if method == "direct":
            if not self._cholesky_is_up_to_date:
                self.factor()
            solution = self._solver(y)
        elif method == "pcg":
            solution = np.zeros_like(y)
            for i in range(y.shape[1]):
                # Use conjugate gradient for each right-hand side
                x, info = cg(self, y[:, i], rtol=tol, callback=callback)  # Use self for matvec
                if info != 0:
                    raise RuntimeError(f"Conjugate gradient failed to converge for column {i}")
                solution[:, i] = x
        else:
            raise ValueError("Method must be either 'direct' or 'pcg'")
            
        if self._which_indices is not None:
            solution = solution[self._which_indices, :]
            
        return solution.reshape(b.shape)
    
    def solve_L(self, b: np.ndarray) -> np.ndarray:
        """
        Solves Lx = b where L is the Cholesky factor of P = LL'.

        Args:
            b: Right-hand side vector
            
        Returns:
            Solution vector x
        """
        if not self._cholesky_is_up_to_date:
            self.factor()
            
        y = self._expand_vector(b)
        solution = self._solver.solve_L(y, use_LDLt_decomposition=False)
        
        if self._which_indices is not None:
            solution = solution[self._which_indices, :]
        return solution.reshape(b.shape)

    def inverse_diagonal(self, 
        initialization: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        method: str = "hutchinson",
        n_samples: int = 100,
        seed: Optional[int] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute the diagonal elements of the inverse of the precision matrix.
                
        Args:
            initialization: Optional tuple of matrices containing:
                - Probe vectors v
                - Initial values for P \ v, can equal v
            method: Method for computing the inverse diagonal. Options:
                - exact: Invert the full matrix and extract the diagonal
                - hutchinson: Hutchinson's method
                - xnys: the xnystrace method (https://arxiv.org/abs/2301.07825)
            n_samples: Number of probe vectors to use for Hutchinson's method
            seed: Random seed for generating probe vectors

        Returns:
            If initialization is None:
                Array of diagonal elements of the correlation matrix (inverse of precision matrix)
            If initialization is not None:
                Tuple containing:
                - Array of diagonal elements of the correlation matrix
                - Updated values for P \ v
        """
        if method not in ["exact", "hutchinson", "xnys"]:
            raise ValueError(f"Unknown method: {method}")
            
        # Get the matrix size
        n = self._matrix.shape[0]
        
        if method == "exact":
            # Convert to dense matrix and invert
            dense_matrix = self._matrix.toarray()
            inv_matrix = np.linalg.inv(dense_matrix)
            diag = np.diag(inv_matrix)
            
            return diag if self._which_indices is None else diag[self._which_indices]
            
        elif method == "hutchinson":
            # Use provided probe vectors or generate new ones
            if initialization is not None:
                v, pv = initialization
            else:
                # Set random seed if provided
                rng = np.random.RandomState(seed)
                
                # Generate Rademacher random vectors (+1/-1 with equal probability)
                v = rng.choice([-1, 1], size=(n, n_samples))
                
                # Initialize P \ v as v
                pv = v.copy()
            
            # Solve M * y = v for each probe vector
            y = self.solve(pv)
            
            # Estimate diagonal elements as average element-wise product of v_i * y_i
            diag_estimate = np.mean(v * y, axis=1)
            
            # If indices are specified, return only those elements
            if self._which_indices is not None:
                diag_estimate = diag_estimate[self._which_indices]
            
            # Return results based on whether initialization was provided
            return (diag_estimate, y) if initialization is not None else diag_estimate
            
        else:  # xnys
            raise NotImplementedError("XNYSTrace method not yet implemented")

    @staticmethod
    def load(filepath: str, snplist_path: Optional[str] = None, population: Optional[str] = None, 
             snps_only: bool = False) -> 'PrecisionOperator':
        """
        Load an LDGM from a single LD block's edgelist and snplist files.
        
        Args:
            filepath: Path to the .edgelist file or directory containing it
            snplist_path: Optional path to .snplist file or directory. If None, uses filepath
            population: Optional population name to filter files
            snps_only: Import snplist data for SNPs only (smaller memory usage)
            
        Returns:
            PrecisionOperator instance with loaded precision matrix and variant info
        """
        from .io import load_ldgm
        matrix, variant_info = load_ldgm(filepath, snplist_path, population, snps_only)
        return PrecisionOperator(matrix, variant_info)