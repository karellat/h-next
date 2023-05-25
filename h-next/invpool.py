import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from scipy.special import factorial
import torchvision.transforms.functional as TF
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional


def geometric_polynomials(min_coord: float,
                          max_coord: float,
                          number_samples: int,
                          power_x: int,
                          power_y: int):
    return (
            np.power(np.linspace(min_coord, max_coord, number_samples), power_x)
            *
            np.power(np.linspace(min_coord, max_coord, number_samples), power_y)[..., np.newaxis]
    )


class CentralMoments(torch.nn.Module):
    def __init__(self,
                 min_coord,
                 max_coord,
                 number_samples,
                 input_shape,
                 channels_dim=(-1, -2),
                 eps=1e-05):
        super(CentralMoments, self).__init__()
        # Views according to the channels dimension
        dim_x = channels_dim[0]
        dim_y = channels_dim[1]
        self.view_x = np.ones(len(input_shape), dtype=int)
        self.view_x[dim_x] = number_samples
        self.view_y = np.ones(len(input_shape), dtype=int)
        self.view_y[dim_y] = number_samples
        self.view_x = torch.Size(self.view_x)
        self.view_y = torch.Size(self.view_y)
        self.moment_view = torch.Size()
        self.eps = eps

        self.coords_x = torch.nn.Parameter(
            torch.from_numpy(
                np.linspace(min_coord, max_coord, number_samples, dtype=np.float64)
            ).to(torch.get_default_dtype())
            .view(self.view_x),
            requires_grad=False
        )
        self.coords_y = torch.nn.Parameter(
            torch.from_numpy(
                np.linspace(min_coord, max_coord, number_samples, dtype=np.float64)
            )
            .to(torch.get_default_dtype())
            .view(self.view_y),
            requires_grad=False
        )
        self.polynomials_01 = torch.nn.Parameter(
            (
                    torch.pow(self.coords_x, 0)
                    *
                    torch.pow(self.coords_y, 1)
            ), requires_grad=False)

        self.polynomials_10 = torch.nn.Parameter(
            (
                    torch.pow(self.coords_x, 1)
                    *
                    torch.pow(self.coords_y, 0)
            ), requires_grad=False
        )
        self.channel_dims = channels_dim

    def sum_channel(self, x, keepdim=True):
        return torch.sum(x,
                         dim=self.channel_dims,
                         keepdim=keepdim)

    def center_of_mass(self, x):
        m00 = self.sum_channel(x)
        # m00 close to zero return 0, 0

        tx = self.sum_channel(x * self.polynomials_10) / torch.clamp(m00, min=self.eps)
        ty = self.sum_channel(x * self.polynomials_01) / torch.clamp(m00, min=self.eps)
        return tx, ty

    def central_moment(self,
                       x: torch.Tensor,
                       tx: torch.Tensor,
                       ty: torch.Tensor,
                       rank_x: int,
                       rank_y: int):
        polynomial = (
                torch.pow((self.coords_x - tx), rank_x)
                *
                torch.pow((self.coords_y - ty), rank_y)
        )

        return self.sum_channel(polynomial * x, keepdim=False)


class HuInvariant1(CentralMoments):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tx, ty = self.center_of_mass(x)
        c_m20 = self.central_moment(x, tx, ty, 2, 0)
        c_m02 = self.central_moment(x, tx, ty, 0, 2)
        return c_m20 + c_m02


class HuInvariant2(CentralMoments):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tx, ty = self.center_of_mass(x)
        c_m20 = self.central_moment(x, tx, ty, 2, 0)
        c_m02 = self.central_moment(x, tx, ty, 0, 2)
        c_m11 = self.central_moment(x, tx, ty, 1, 1)
        return torch.square(c_m20 - c_m02) + 4 * torch.square(c_m11)


class HuInvariant3(CentralMoments):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tx, ty = self.center_of_mass(x)
        c_m30 = self.central_moment(x, tx, ty, 3, 0)
        c_m12 = self.central_moment(x, tx, ty, 1, 2)
        c_m21 = self.central_moment(x, tx, ty, 2, 1)
        c_m03 = self.central_moment(x, tx, ty, 0, 3)

        return torch.square(c_m30 - 3 * c_m12) + torch.square(3 * c_m21 - c_m03)


class HuInvariant4(CentralMoments):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tx, ty = self.center_of_mass(x)
        c_m30 = self.central_moment(x, tx, ty, 3, 0)
        c_m12 = self.central_moment(x, tx, ty, 1, 2)
        c_m21 = self.central_moment(x, tx, ty, 2, 1)
        c_m03 = self.central_moment(x, tx, ty, 0, 3)

        return torch.square(c_m30 + c_m12) + torch.square(c_m21 + c_m03)


class HuInvariant5(CentralMoments):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tx, ty = self.center_of_mass(x)

        c_m30 = self.central_moment(x, tx, ty, 3, 0)
        c_m12 = self.central_moment(x, tx, ty, 1, 2)
        c_m21 = self.central_moment(x, tx, ty, 2, 1)
        c_m03 = self.central_moment(x, tx, ty, 0, 3)

        return (
                (c_m30 - 3 * c_m12) * (c_m30 + c_m12) *
                (torch.square(c_m30 + c_m12) - 3 * torch.square(c_m21 + c_m03))
                + (3 * c_m21 - c_m03) * (c_m21 + c_m03) * (
                        3 * torch.square(c_m30 + c_m12) - torch.square(c_m21 + c_m03))
        )


class HuInvariant6(CentralMoments):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tx, ty = self.center_of_mass(x)
        c_m20 = self.central_moment(x, tx, ty, 2, 0)
        c_m02 = self.central_moment(x, tx, ty, 0, 2)
        c_m30 = self.central_moment(x, tx, ty, 3, 0)
        c_m12 = self.central_moment(x, tx, ty, 1, 2)
        c_m21 = self.central_moment(x, tx, ty, 2, 1)
        c_m03 = self.central_moment(x, tx, ty, 0, 3)
        c_m11 = self.central_moment(x, tx, ty, 1, 1)

        return (
                (c_m20 - c_m02) * (torch.square(c_m30 + c_m12) - torch.square(c_m21 + c_m03))
                + 4 * c_m11 * (c_m30 + c_m12) * (c_m21 + c_m03)
        )


class HuInvariant7(CentralMoments):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tx, ty = self.center_of_mass(x)
        c_m30 = self.central_moment(x, tx, ty, 3, 0)
        c_m12 = self.central_moment(x, tx, ty, 1, 2)
        c_m21 = self.central_moment(x, tx, ty, 2, 1)
        c_m03 = self.central_moment(x, tx, ty, 0, 3)

        return (
                (3 * c_m21 - c_m03) * (c_m30 + c_m12) * (torch.square(c_m30 + c_m12) - 3 * torch.square(c_m21 + c_m03))
                -
                (c_m30 - 3 * c_m12) * (c_m21 + c_m03) * (3 * torch.square(c_m30 + c_m12) - torch.square(c_m21 + c_m03))
        )


class M00(torch.nn.Module):
    def __init__(self,
                 channels_dim=(-1, -2),
                 **kwargs):
        super(M00, self).__init__()
        self.dims = channels_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x, dim=self.dims)


class GlobalMeanPool(torch.nn.Module):
    def __init__(self,
                 channels_dim=(-1, -2),
                 **kwargs):
        super(GlobalMeanPool, self).__init__()
        self.dims = channels_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=self.dims)


class ZernikePooling(torch.nn.Module):
    def __init__(self,
                 number_samples,
                 m,
                 n,
                 use_center_of_mass: bool = True,
                 channels_pad: int = 0,
                 filters_pad: float = 0.1,
                 circle_mask_pad=0.05,
                 channels_dim=(2, 3),
                 eps=1e-05):
        super(ZernikePooling, self).__init__()
        assert m >= 0, "Zernike angle \"m\" must be positive."
        self.filters_pad = filters_pad
        min_coord = -1 - filters_pad
        max_coord = 1 + filters_pad
        # Views according to the channels dimension
        dim_x = channels_dim[0]
        dim_y = channels_dim[1]
        self.view_x = np.ones(4, dtype=int)
        self.view_x[dim_x] = number_samples
        self.view_y = np.ones(4, dtype=int)
        self.view_y[dim_y] = number_samples
        self.view_x = torch.Size(self.view_x)
        self.view_y = torch.Size(self.view_y)
        self.moment_view = torch.Size()
        self.eps = eps
        self.number_samples = number_samples
        self.channels_pad = channels_pad
        self.circle_mask_pad = circle_mask_pad
        self.m = m
        self.n = n
        self.use_center_of_mass = use_center_of_mass

        coords_x = (torch.from_numpy(
            np.linspace(min_coord,
                        max_coord,
                        number_samples,
                        dtype=np.float64))
                    .to(torch.get_default_dtype())
                    .view(self.view_x))
        coords_y = (torch.from_numpy(
            np.linspace(min_coord,
                        max_coord,
                        number_samples,
                        dtype=np.float64))
                    .to(torch.get_default_dtype())
                    .view(self.view_y))
        self.polynomials_01 = torch.nn.Parameter(
            (
                    torch.pow(coords_x, 0)
                    *
                    torch.pow(coords_y, 1)
            ), requires_grad=False)

        self.polynomials_10 = torch.nn.Parameter(
            (
                    torch.pow(coords_x, 1)
                    *
                    torch.pow(coords_y, 0)
            ), requires_grad=False
        )
        self.channel_dims = channels_dim

        # Parameter precalc
        x = torch.linspace(min_coord, max_coord, number_samples)

        xx, yy = np.meshgrid(x, x, indexing='ij')
        self.xx = torch.nn.Parameter(torch.from_numpy(xx), requires_grad=False)
        self.yy = torch.nn.Parameter(torch.from_numpy(yy), requires_grad=False)
        ks = np.arange(0, np.floor_divide((n - m), 2) + 1)

        a = np.floor_divide((n + m), 2)
        b = np.floor_divide((n - m), 2)
        factor = (-1) ** ks * factorial(n - ks) / (factorial(ks) * factorial(a - ks) * factorial(b - ks))

        self.factor = torch.nn.Parameter(
            torch.from_numpy(factor[np.newaxis, np.newaxis, np.newaxis, :]).type(torch.get_default_dtype()),
            requires_grad=False)
        self.exponent = torch.nn.Parameter(
            torch.from_numpy((n - 2 * ks)[np.newaxis, np.newaxis, np.newaxis, :]).type(torch.get_default_dtype()),
            requires_grad=False)

    @torch.jit.export
    def sum_channel(self, x, keepdim=True):
        return torch.sum(x,
                         dim=self.channel_dims,
                         keepdim=keepdim)

        # Prepare parameters

    @torch.jit.export
    def center_of_mass(self, x):
        m00 = self.sum_channel(x)
        # m00 close to zero return 0, 0

        tx = self.sum_channel(x * self.polynomials_10) / torch.clamp(m00, min=self.eps)
        ty = self.sum_channel(x * self.polynomials_01) / torch.clamp(m00, min=self.eps)
        return tx, ty

    @torch.jit.export
    def get_zernike_filter(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_center_of_mass:
            tx, ty = self.center_of_mass(x)
            xx = self.xx - tx
            yy = self.yy - ty
        else:
            xx = self.xx
            yy = self.yy
        r = torch.sqrt(xx ** 2 + yy ** 2)
        radial_polynomial = torch.sum(self.factor *
                                      torch.pow(r[..., None], self.exponent),
                                      dim=-1)
        if self.m == 0:
            zernike_polynomial = torch.complex(real=radial_polynomial,
                                               imag=torch.zeros_like(radial_polynomial))
        else:
            theta = torch.arctan2(yy, xx)
            zernike_polynomial_real = radial_polynomial * torch.sin(self.m * theta)
            zernike_polynomial_imag = radial_polynomial * torch.cos(self.m * theta)
            zernike_polynomial = torch.complex(real=zernike_polynomial_real,
                                               imag=zernike_polynomial_imag)

        circular_mask = ~((xx ** 2 + yy ** 2) > (1 - self.filters_pad - self.circle_mask_pad))
        zernike_polynomial *= circular_mask
        return zernike_polynomial

    def get_example_filter(self):
        return self.get_zernike_filter(torch.ones([1, 1, self.number_samples, self.number_samples]))[0, 0]

    def plot_example_filter(self):
        fig, (ax_real, ax_imag) = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))
        Z_complex = self.get_example_filter()
        min_coord, max_coord = (-1 - self.filters_pad, 1 + self.filters_pad)
        x = torch.linspace(min_coord, max_coord, self.number_samples)
        xx, yy = torch.meshgrid(x, x, indexing='xy')
        for (Z, ax) in zip([Z_complex.real, Z_complex.imag], [ax_real, ax_imag]):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im = ax.imshow(Z, extent=[min_coord, max_coord, min_coord, max_coord], origin='lower', cmap='viridis')
            ax.contour(xx, yy, Z, alpha=0.3, colors='k')
            ax.grid(False)
            fig.colorbar(im, cax=cax, orientation='vertical')
        ax_real.set_title(f'Zm{self.m}n{self.n}_real')
        ax_imag.set_title(f'Zm{self.m}n{self.n}_imag')
        return fig

    @torch.jit.export
    def normalize_input(self, x: torch.Tensor):
        _x = x
        if self.channels_pad > 0:
            _x = torch.nn.functional.pad(_x,
                                         pad=[self.channels_pad, self.channels_pad,
                                              self.channels_pad, self.channels_pad])
        return _x

    def forward(self, x: torch.Tensor):
        # normalize and permute from [B, H, W, C] to [B, C, H, W]
        _x = self.normalize_input(x)

        zernike_polynomial = self.get_zernike_filter(_x)
        # NOTE: grad |x| is infinite at zero
        if self.m == 0:
            return self.sum_channel(zernike_polynomial.real * _x, keepdim=False)
        else:
            return self.sum_channel((zernike_polynomial * _x).abs(), keepdim=False)

    def string(self):
        return f"ZerPoolM-{self.m}N{self.n}-S{self.number_samples}P{self.filters_pad}"


class MultipleZernikePooling(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 max_rank: int,
                 variance_norm: bool = True,
                 filters_pad: int = 0.1,
                 circle_mask_pad: float = 0.05,
                 use_center_of_mass=True,
                 skip_zero_order=True,
                 mean_pool: bool = False,
                 spatial_dim=None,
                 eps=1e-05):
        super().__init__()
        if spatial_dim is None:
            spatial_dim = [1, 2]
        else:
            assert len(spatial_dim) == 2
        pooling_layers = []
        for rank in range(1 if skip_zero_order else 0, max_rank + 1):
            for angle in range(rank + 1):
                pooling_layers.append(ZernikePooling(m=angle,
                                                     n=rank,
                                                     use_center_of_mass=use_center_of_mass,
                                                     number_samples=input_size,
                                                     filters_pad=filters_pad,
                                                     circle_mask_pad=circle_mask_pad))
        self.mean_pool = mean_pool
        self.number_invariants = len(pooling_layers)
        if mean_pool:
            self.number_invariants += 1
        self.pooling_layers = torch.nn.ModuleList(pooling_layers)
        self.variance_norm = variance_norm
        self.input_size = input_size
        self.spatial_dim = spatial_dim
        self.eps = eps
        self.max_rank = max_rank

    def forward(self, x):
        assert x.shape[self.spatial_dim[0]] == self.input_size
        assert x.shape[self.spatial_dim[1]] == self.input_size
        if self.variance_norm:
            _x = x / (torch.clamp(torch.var(x, dim=self.spatial_dim, keepdim=True), min=self.eps))
        else:
            _x = x
        out_x = [torch.mean(x, dim=self.spatial_dim)] if self.mean_pool else []
        for layer in self.pooling_layers:
            out_x.append(layer(_x))
        return torch.stack(dim=1, tensors=out_x)


class ParallelZernikePooling(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 number_invariants: int,
                 variance_norm: bool = True,
                 filters_pad: int = 0.1,
                 circle_mask_pad: float = 0.05,
                 skip_zero_order=True,
                 mean_pool: bool = False,
                 spatial_dim=None,
                 eps=1e-05):
        super().__init__()
        if spatial_dim is None:
            spatial_dim = [1, 2]
        else:
            assert len(spatial_dim) == 2
        rank = 1 if skip_zero_order else 0
        angle = 0
        pooling_layers = []
        _number_invariants = (number_invariants - 1) if mean_pool else number_invariants

        while len(pooling_layers) != _number_invariants:
            self.max_rank = rank
            pooling_layers.append(ZernikePooling(m=angle,
                                                 n=rank,
                                                 number_samples=input_size,
                                                 filters_pad=filters_pad,
                                                 circle_mask_pad=circle_mask_pad))
            if angle < rank:
                angle += 1
            else:
                angle = 0
                rank += 1

        self.mean_pool = mean_pool
        self.pooling_layers = torch.nn.ModuleList(pooling_layers)
        self.variance_norm = variance_norm
        self.input_size = input_size
        self.spatial_dim = spatial_dim
        self.eps = eps

    def forward(self, x):
        assert x.shape[self.spatial_dim[0]] == self.input_size
        assert x.shape[self.spatial_dim[1]] == self.input_size
        if self.variance_norm:
            _x = x / (torch.clamp(torch.var(x, dim=self.spatial_dim, keepdim=True), min=self.eps))
        else:
            _x = x
        if self.mean_pool:
            out_x = [torch.mean(x[:, 0:1], dim=self.spatial_dim)]
            for idx, layer in enumerate(self.pooling_layers):
                out_x.append(layer(_x[:, idx + 1:idx + 2]))
        else:
            out_x = [layer(_x[:, idx:idx + 1]) for idx, layer in enumerate(self.pooling_layers)]
        return torch.cat(dim=1, tensors=out_x)


def get_single_invariant_layer(invariant_name="M00", **kwargs):
    if not hasattr(sys.modules[__name__], invariant_name):
        raise NotImplementedError(f"Unknown pooling layer \"{invariant_name}\"")
    else:
        return getattr(sys.modules[__name__], invariant_name)(**kwargs)
