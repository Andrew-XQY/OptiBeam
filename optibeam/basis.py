# bases.py
from __future__ import annotations
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class BasisInfo:
    name: str
    shape: Tuple[int, int]
    length: int


class BasisGenerator(ABC):
    """
    Base class for 2D orthonormal/separable-like bases over an HxW grid.

    API:
      - generator() -> generator of basis images (H,W), float32
      - components() -> generator of basis images (H,W), float32
      - component(u, v) -> one basis image
      - shape, length, name (properties)
      - resize(shape) -> update resolution
      - show_basis(save_dir=None) -> matplotlib Figure (and saved path if provided)
      - value_range: optional (min, max) tuple to remap basis values
    """

    def __init__(self, shape: Tuple[int, int], name: str, value_range: Optional[Tuple[float, float]] = None):
        self._h, self._w = int(shape[0]), int(shape[1])
        self._name = str(name)
        self._value_range = value_range

    # -------- metadata --------
    @property
    def name(self) -> str:
        return self._name

    @property
    def shape(self) -> Tuple[int, int]:
        return (self._h, self._w)

    @property
    def length(self) -> int:
        return self._h * self._w

    def info(self) -> BasisInfo:
        return BasisInfo(self._name, self.shape, self.length)

    @property
    def value_range(self) -> Optional[Tuple[float, float]]:
        """Get the target value range for basis remapping, or None if using natural range."""
        return self._value_range

    @value_range.setter
    def value_range(self, range_tuple: Optional[Tuple[float, float]]) -> None:
        """Set the target value range for basis remapping. None uses natural range."""
        self._value_range = range_tuple

    # -------- sizing --------
    def resize(self, shape: Tuple[int, int]) -> None:
        self._h, self._w = int(shape[0]), int(shape[1])
        self._on_resize()

    def _on_resize(self) -> None:
        """Hook for subclasses to invalidate caches after resize."""
        pass

    # -------- core API --------
    @abstractmethod
    def _component_raw(self, u: int, v: int) -> np.ndarray:
        """Return the (u,v) basis image in its natural value range as float32 array of shape (H,W)."""
        raise NotImplementedError

    def _remap_values(self, img: np.ndarray) -> np.ndarray:
        """Remap basis image values to target range if value_range is set."""
        if self._value_range is None:
            return img
        
        # Find current min/max
        img_min, img_max = img.min(), img.max()
        if img_max == img_min:
            # Constant image - map to midpoint of target range
            target_mid = (self._value_range[0] + self._value_range[1]) / 2.0
            return np.full_like(img, target_mid)
        
        # Linear remap: [img_min, img_max] -> [target_min, target_max]
        target_min, target_max = self._value_range
        img_normalized = (img - img_min) / (img_max - img_min)  # -> [0, 1]
        img_remapped = img_normalized * (target_max - target_min) + target_min
        return img_remapped.astype(np.float32)

    def component(self, u: int, v: int) -> np.ndarray:
        """Return the (u,v) basis image as float32 array of shape (H,W), with optional value remapping."""
        img = self._component_raw(u, v)
        return self._remap_values(img)

    def components(self) -> Generator[np.ndarray, None, None]:
        """Yield all basis images in (row-major) order: u in [0..H-1], v in [0..W-1]."""
        for u in range(self._h):
            for v in range(self._w):
                yield self.component(u, v)

    def generator(self) -> Generator[np.ndarray, None, None]:
        """Return a generator yielding all basis images. Alias for components()."""
        return self.components()

    # -------- visualization --------
    def show_basis(self, save_dir: Optional[str] = None, max_cols: Optional[int] = None):
        """
        Tile all basis images into a single figure. If save_dir is provided,
        saves as PNG with a timestamped filename inside that directory.

        Returns:
          - If save_dir is None: matplotlib Figure
          - If save_dir is provided: (Figure, saved_path_str)
        """
        imgs = list(self.components())
        n = len(imgs)
        cols = max_cols or int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))

        # Keep per-tile a decent size; adjust as you prefer.
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6), dpi=150)
        axes = np.atleast_2d(axes)

        v = max(np.max(np.abs(img)) for img in imgs) or 1.0

        for i in range(rows * cols):
            ax = axes.flat[i]
            if i < n:
                ax.imshow(imgs[i], cmap="gray", vmin=-v, vmax=v, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis("off")

        plt.tight_layout(pad=0.3)

        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            out = Path(save_dir) / f"{self.name}_{self._h}x{self._w}_{ts}.png"
            fig.savefig(out, dpi=300)
            return fig, str(out)

        return fig


# -------------------- DCT-II (orthonormal) --------------------
class DCTBasis(BasisGenerator):
    """
    2D DCT-II orthonormal basis on an HxW grid.

    Basis element:
      B_{u,v}[x,y] = α(u,H) α(v,W)
                     cos(pi * (2x+1) * u / (2H)) * cos(pi * (2y+1) * v / (2W)),
      with α(0,N)=1/sqrt(N), α(k>0,N)=sqrt(2/N).
    
    Parameters
    ----------
    shape : (H, W)
    name : str
    value_range : optional (min, max) tuple to remap basis values from natural range
    """

    def __init__(self, shape: Tuple[int, int], name: str = "DCT", value_range: Optional[Tuple[float, float]] = None):
        super().__init__(shape, name, value_range)

    @staticmethod
    def _alpha(k: int, n: int) -> float:
        return (1.0 / math.sqrt(n)) if k == 0 else math.sqrt(2.0 / n)

    def _component_raw(self, u: int, v: int) -> np.ndarray:
        H, W = self.shape
        x = np.arange(H, dtype=np.float64)
        y = np.arange(W, dtype=np.float64)

        ax = self._alpha(u, H)
        ay = self._alpha(v, W)

        # Cosine terms
        cx = np.cos(np.pi * (2.0 * x + 1.0) * u / (2.0 * H))
        cy = np.cos(np.pi * (2.0 * y + 1.0) * v / (2.0 * W))

        img = (ax * cx)[:, None] * (ay * cy)[None, :]
        return img.astype(np.float32)


# -------------------- Walsh–Hadamard (FWHT) --------------------
class WalshHadamardBasis(BasisGenerator):
    """
    2D Walsh–Hadamard basis via Sylvester/Kronecker construction.
    
    An orthogonal, complete basis for 2D image representation using
    the Fast Walsh-Hadamard Transform (FWHT). The basis consists of
    {+1, -1} patterns, making it efficient for digital signal processing.
    
    Requirements:
        - Square shape: H == W
        - Power-of-two size: n = 2^m (e.g., 4, 8, 16, 32, 64, 128, ...)
    
    The 2D basis is constructed as outer products of 1D Walsh-Hadamard
    functions, ensuring orthonormality and completeness.
    
    Parameters
    ----------
    shape : (H, W)
    value_range : optional (min, max) tuple to remap basis values from natural range
    """

    def __init__(self, shape: Tuple[int, int], value_range: Optional[Tuple[float, float]] = None):
        super().__init__(shape, name="WalshHadamard", value_range=value_range)
        self._H1d: Optional[np.ndarray] = None
        self._check_shape()
        self._build_1d()

    def _check_shape(self):
        H, W = self.shape
        if H != W:
            raise ValueError("WalshHadamardBasis requires a square shape (H == W).")
        if H & (H - 1) != 0:
            raise ValueError("WalshHadamardBasis size must be a power of two (e.g., 4, 8, 16, 32, ...).")

    def _on_resize(self) -> None:
        self._H1d = None
        self._check_shape()
        self._build_1d()

    def _build_1d(self):
        """Build the 1D Hadamard matrix via recursive Kronecker products."""
        n = self._h
        # Start with base 2x2 Hadamard matrix
        H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64)
        
        # Build up to size n via Kronecker products
        while H.shape[0] < n:
            H = np.kron(H, np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64))
        
        # Normalize to make rows orthonormal
        H = H / math.sqrt(n)
        self._H1d = H.astype(np.float64)

    def _component_raw(self, u: int, v: int) -> np.ndarray:
        """Return the (u,v) Walsh-Hadamard basis image."""
        if self._H1d is None:
            self._build_1d()
        # 2D basis via outer product of 1D basis functions
        img = np.outer(self._H1d[u], self._H1d[v])
        return img.astype(np.float32)


# -------------------- Hadamard (original) --------------------
class HadamardBasis(BasisGenerator):
    """
    2D Walsh–Hadamard basis via Sylvester construction.
    Requires a square, power-of-two shape: H == W == 2^m.

    Basis images are outer products of normalized 1D Hadamard rows.
    
    Parameters
    ----------
    shape : (H, W)
    value_range : optional (min, max) tuple to remap basis values from natural range
    """

    def __init__(self, shape: Tuple[int, int], value_range: Optional[Tuple[float, float]] = None):
        super().__init__(shape, name="Hadamard", value_range=value_range)
        self._H1d: Optional[np.ndarray] = None
        self._check_shape()
        self._build_1d()

    def _check_shape(self):
        H, W = self.shape
        if H != W:
            raise ValueError("HadamardBasis requires a square shape (H == W).")
        if H & (H - 1) != 0:
            raise ValueError("HadamardBasis size must be a power of two (e.g., 4, 8, 16, ...).")

    def _on_resize(self) -> None:
        self._H1d = None
        self._check_shape()
        self._build_1d()

    def _build_1d(self):
        n = self._h
        H = np.array([[1, 1], [1, -1]], dtype=np.float64)
        k = 1
        while (1 << k) < n:
            H = np.kron(H, np.array([[1, 1], [1, -1]], dtype=np.float64))
            k += 1
        if H.shape[0] != n:
            # n is power of two; loop should match exactly
            raise RuntimeError("Internal Hadamard build error.")
        # Row-normalize (orthonormal rows)
        H = H / math.sqrt(n)
        self._H1d = H.astype(np.float64)

    def _component_raw(self, u: int, v: int) -> np.ndarray:
        if self._H1d is None:
            self._build_1d()
        h = self._H1d
        img = np.outer(h[u], h[v])  # (n,) ⊗ (n,) -> (n,n)
        return img.astype(np.float32)


# -------------------- Hermite–Gaussian (orthonormal, discrete grid) --------------------

class HermiteGaussianBasis(BasisGenerator):
    """
    2D Hermite–Gaussian (HG) basis sampled on an HxW grid using stable
    *normalized* Hermite-function recursion.

    1D normalized Hermite functions h_n(x):
      h_0(x) = π^(-1/4) * exp(-x^2/2)
      h_1(x) = sqrt(2) * x * h_0(x)
      h_{n+1}(x) = sqrt(2/(n+1)) * x * h_n(x) - sqrt(n/(n+1)) * h_{n-1}(x)

    The 2D basis is separable: H_{u,v}(x,y) = h_u(x) * h_v(y).

    Parameters
    ----------
    shape : (H, W)
    extent : float
        Half-width of the continuous domain sampled on each axis; x∈[-extent,extent].
        Typical values: 2.5–4.0. Larger extent resolves higher orders better.
    value_range : optional (min, max) tuple to remap basis values from natural range
    """

    def __init__(self, shape: Tuple[int, int], extent: float = 3.0, value_range: Optional[Tuple[float, float]] = None):
        super().__init__(shape, name="HermiteGaussian", value_range=value_range)
        self.extent = float(extent)
        self._hx: Optional[np.ndarray] = None  # (H, H)
        self._hy: Optional[np.ndarray] = None  # (W, W)
        self._build_1d()

    def _on_resize(self) -> None:
        self._hx = None
        self._hy = None
        self._build_1d()

    def _build_1d(self):
        H, W = self.shape
        x = np.linspace(-self.extent, self.extent, H, dtype=np.float64)
        y = np.linspace(-self.extent, self.extent, W, dtype=np.float64)
        self._hx = self._hermite_functions(H - 1, x)  # shape (H, H)
        self._hy = self._hermite_functions(W - 1, y)  # shape (W, W)

    @staticmethod
    def _hermite_functions(nmax: int, x: np.ndarray) -> np.ndarray:
        """
        Return array Phi of shape (nmax+1, len(x)),
        where Phi[n] = normalized h_n(x) (orthonormal over R).
        """
        N = nmax + 1
        m = x.size
        Phi = np.zeros((N, m), dtype=np.float64)

        # h_0
        Phi[0] = (np.pi ** -0.25) * np.exp(-0.5 * x * x)

        if N == 1:
            return Phi

        # h_1
        Phi[1] = math.sqrt(2.0) * x * Phi[0]

        # Recurrence for normalized functions
        for n in range(1, nmax):
            a = math.sqrt(2.0 / (n + 1))
            b = math.sqrt(n / (n + 1))
            Phi[n + 1] = a * x * Phi[n] - b * Phi[n - 1]

        # Optional per-function normalization on the discrete grid
        # (keeps numerical drift tiny for large n on finite sampling)
        for n in range(N):
            s = np.linalg.norm(Phi[n])
            if s > 0:
                Phi[n] /= s

        return Phi

    def _component_raw(self, u: int, v: int) -> np.ndarray:
        if self._hx is None or self._hy is None:
            self._build_1d()
        hx_u = self._hx[u]  # (H,)
        hy_v = self._hy[v]  # (W,)
        img = np.outer(hx_u, hy_v)
        # Normalize to unit Frobenius norm for consistency across bases
        fnorm = np.linalg.norm(img)
        if fnorm > 0:
            img = img / fnorm
        return img.astype(np.float32)


# -------------------- Convenience constructors --------------------
def make_dct(shape: Tuple[int, int], value_range: Optional[Tuple[float, float]] = None) -> DCTBasis:
    return DCTBasis(shape, name="DCT", value_range=value_range)


def make_walsh_hadamard(n: int, value_range: Optional[Tuple[float, float]] = None) -> WalshHadamardBasis:
    """Square, power-of-two size n (e.g., 4, 8, 16, 32, ...)."""
    return WalshHadamardBasis((n, n), value_range=value_range)


def make_hadamard(n: int, value_range: Optional[Tuple[float, float]] = None) -> HadamardBasis:
    """Square, power-of-two size n (e.g., 4, 8, 16, ...)."""
    return HadamardBasis((n, n), value_range=value_range)


def make_hg(shape: Tuple[int, int], extent: float = 3.0, value_range: Optional[Tuple[float, float]] = None) -> HermiteGaussianBasis:
    return HermiteGaussianBasis(shape, extent=extent, value_range=value_range)



