import torch
from torch import nn
import numpy as np
from hnets import hnet_ops
from scipy.linalg import dft
from typing import Tuple, Dict
from numpy import newaxis
from hnets.hnet_lite import HStackMagnitudes, HNonlin, HBatchNorm


class UpscaleHConv2d(nn.Module):
    """Defining custom convolutional layer"""

    @staticmethod
    def create_circular_mask(side):
        assert side % 2 == 0
        center = ((side - 1) / 2), ((side - 1) / 2)
        radius = (side / 2)

        y, x = np.ogrid[:side, :side]
        dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

        mask = dist_from_center < radius
        return mask

    @staticmethod
    def init_weights_dict(layer_shape,
                          in_max_order: int,
                          out_max_order: int,
                          ring_count=None):
        """
        Initializes the dict of weights

        Args:
            layer_shape (list of ints): contains dimensions of the layer as a list
            in_max_order: maximum rotation order of input channels
            out_max_order: maximum rotation order of output channels
            ring_count (int): Number of rings for calculating the basis
        Returns:
            weights_dict: initialized dict of weights
        """

        rotation_orders = range(0, np.max([in_max_order, out_max_order]) + 1)
        weights_dict = {}
        for order in rotation_orders:
            if ring_count is None:
                ring_count = int(np.maximum(layer_shape[0] / 2, 2))
            sh = [ring_count, ] + list(layer_shape[2:])
            # Initializes the weights using initialization method of He
            stddev = 0.4 * np.sqrt(2.0 / np.prod(sh[:3]))
            weights_dict[order] = nn.Parameter(torch.normal(torch.zeros(*sh), stddev))
        return weights_dict

    def get_angle_samples_count(self):
        return np.maximum(np.ceil(np.pi * self.kernel_size), 101)

    @staticmethod
    def get_interpolation_weights(fs, m, n_rings, angle_samples):
        """
        Used to construct the steerable filters using Radial basis functions.
        The filters are constructed on the patches of n_rings using Gaussian
        interpolation. (Code adapted from the tf code of Worrall et al., CVPR, 2017)

        Args:
            fs (int): filter size for the H-net convolutional layer
            m (int): max. rotation order for the steerable filters
            n_rings (int): No. of rings for the steerable filters
            angle_samples (int): No. of angle samples

        Returns:
            norm_weights (numpy): contains normalized weights for interpolation
            using the steerable filters
        """

        mid = int(np.floor(fs / 2))
        # Variance of Gaussian resampling
        std_gauss = (mid / n_rings) / 2
        # We define below radii up to n_rings-0.5 (as in Worrall et al, CVPR 2017)
        radii = np.linspace(m != 0, mid - 2 * std_gauss, n_rings)
        # We define pixel centers to be at positions 0.5
        center_pt = np.asarray([fs, fs]) / 2.

        # Extracting the set of angles to be sampled

        # Choosing the sampling locations for the rings
        lin = (2 * np.pi * np.arange(angle_samples)) / angle_samples
        ring_locations = np.vstack([-np.sin(lin), np.cos(lin)])

        # Create interpolation coefficient coordinates
        coords = hnet_ops.get_l2_neighbors(center_pt, fs)

        # getting samples based on the chosen center_pt and the coords
        radii = radii[:, np.newaxis, np.newaxis, np.newaxis]
        ring_locations = ring_locations[np.newaxis, :, :, np.newaxis]
        diff = radii * ring_locations - coords[np.newaxis, :, np.newaxis, :]
        dist2 = np.sum(diff ** 2, axis=1)

        # Convert distances to weightings
        weights = np.exp(-0.5 * dist2 / (std_gauss ** 2))  # For bandwidth of 0.5

        # Normalizing the weights to calibrate the different steerable filters
        norm = np.sum(weights,
                      axis=2,
                      keepdims=True)
        assert np.alltrue(norm != 0), "Normalizing by zero weights"
        return np.divide(weights,
                         norm,
                         where=(norm != 0)
                         )

    @staticmethod
    def init_phase_dict(inp_channel_count, out_channel_count, in_max_order, out_max_order):
        """
        Initializes phase dict with phase offsets

            out_channel_count (int): number of output channels
            in_max_order: maximum rotation order of input channels
            out_max_order: maximum rotation order of output channels
        Returns:
            phase_dict: initialized dict of phase offsets
        """

        rotation_orders = range(0, np.max([in_max_order, out_max_order]) + 1)
        phase_dict = {}
        for order in rotation_orders:
            init = np.random.rand(1, 1, inp_channel_count, out_channel_count) * 2. * np.pi
            init = torch.from_numpy(init)
            phase = nn.Parameter(init)
            phase_dict[order] = phase
        return phase_dict

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 in_max_order,
                 out_max_order,
                 stride=1,
                 padding=0,
                 phase=True,
                 stddev=0.4,
                 circular_masking=False,
                 mask_shape=None,
                 norm_polar2cart=False,
                 n_rings=None):
        super().__init__()
        '''
        initializer function for Harmonic convolution lite wrapper class

        Args:
            in_channels (int): Number of input channels
            out_channels (int ): Number of output channels (int)
            kernel_size (int): dimensions of the (square) filter
            stride (tuple of ints): tuple denoting strides for h and w directions. Similar
            to the original tf code as well as the pytorch tuple standard format for stride,
            we provide a 4-size tuple here. The dimensions N and c as per convention are
            also set to 1.(default (1,1,1,1))
            padding (int): amount of implicit zero paddings on both sides
            phase (boolean): to decide whether the phase-offset term is used (default True)
            in_max_order (int): max. order of rotation to be modeled(default: 1)
            out_max_order (int): max. order of rotation to be modeled(default: 1)
            stddev (float): scaling term for He weight initialization
        '''

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.phase = phase
        self.in_max_order = in_max_order
        self.out_max_order = out_max_order
        self.stddev = stddev
        self.n_rings = n_rings
        self.shape = [kernel_size, kernel_size, self.in_channels, self.out_channels]
        self.Q = UpscaleHConv2d.init_weights_dict(self.shape,
                                                  in_max_order=self.in_max_order,
                                                  out_max_order=self.out_max_order,
                                                  ring_count=self.n_rings)
        for k, v in self.Q.items():
            self.register_parameter('weights_dict_' + str(k), v)
        if self.phase:
            self.P = UpscaleHConv2d.init_phase_dict(self.in_channels,
                                                    self.out_channels,
                                                    in_max_order=self.in_max_order,
                                                    out_max_order=self.out_max_order)

            for k, v in self.P.items():
                self.register_parameter('phase_dict_' + str(k), v)
        else:
            self.P = None

        #
        self.circular_masking = circular_masking
        if self.circular_masking:
            assert mask_shape is not None
            self.circular_mask = (
                torch.nn.Parameter(
                    torch.from_numpy(
                        UpscaleHConv2d.create_circular_mask(mask_shape)[newaxis, :, :, newaxis, newaxis, newaxis]
                    )
                    , requires_grad=False)

            )

        # low pass filters
        N = self.get_angle_samples_count()
        self.low_pass_filters = torch.nn.ParameterDict()
        for m in self.Q.keys():
            # Get the basis matrices built from the steerable filters
            weights = UpscaleHConv2d.get_interpolation_weights(self.kernel_size,
                                                               m=m,
                                                               n_rings=self.n_rings,
                                                               angle_samples=self.get_angle_samples_count())
            low_pass_filter = np.dot(dft(N)[m, :], weights).T
            _cos_comp = torch.from_numpy(np.real(low_pass_filter)).to(torch.get_default_dtype())
            _sin_comp = torch.from_numpy(np.imag(low_pass_filter)).to(torch.get_default_dtype())
            if norm_polar2cart:
                _cos_comp = torch.nn.functional.normalize(_cos_comp, dim=-1)
                _sin_comp = torch.nn.functional.normalize(_sin_comp, dim=-1)

            self.low_pass_filters[f"cos_comp_{m}"] = torch.nn.Parameter(_cos_comp,
                                                                        requires_grad=False)
            self.low_pass_filters[f"sin_comp_{m}"] = torch.nn.Parameter(_sin_comp,
                                                                        requires_grad=False)

    def get_filters(self) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Calculates filters in the form of weight matrices through performing
        single-frequency DFT on every ring obtained from sampling in the polar
        domain.

        Args:

        Returns:
            W (dict): contains the filter matrices
        """
        W = {}  # dict to store the filter matrices

        for m, r in self.Q.items():
            rsh = list(r.size())
            cos_comp = self.low_pass_filters[f'cos_comp_{m}']
            sin_comp = self.low_pass_filters[f'sin_comp_{m}']
            # Computing the projections on the rotational basis
            r = r.view(rsh[0], rsh[1] * rsh[2])
            ucos = torch.matmul(cos_comp, r).view(self.kernel_size, self.kernel_size, rsh[1], rsh[2])
            usin = torch.matmul(sin_comp, r).view(self.kernel_size, self.kernel_size, rsh[1], rsh[2])

            if self.P is not None:
                # Rotating the basis matrices
                ucos_ = torch.cos(self.P[m]) * ucos + torch.sin(self.P[m]) * usin
                usin = -torch.sin(self.P[m]) * ucos + torch.cos(self.P[m]) * usin
                ucos = ucos_
            W[m] = (ucos, usin)

        return W

    def forward(self, X):
        """
        Forward propagation function for the harmonic convolution operation

        Args:
            X (deep tensor): input feature tensor

        Returns:
            R (deep tensor): output feature tensor obtained from harmonic convolution
        """

        W = self.get_filters()
        R = hnet_ops.h_conv(X, W,
                            strides=self.stride,
                            padding=self.padding,
                            in_max_order=self.in_max_order,
                            out_max_order=self.out_max_order)
        if self.circular_masking:
            R *= self.circular_mask
        return R


class HCircularMask(nn.Module):
    def __init__(self, mask_shape):
        super().__init__()
        assert mask_shape % 2 == 0, "Only for even shapes"
        center = ((mask_shape - 1) / 2), ((mask_shape - 1) / 2)
        radius = (mask_shape / 2)

        y, x = np.ogrid[:mask_shape, :mask_shape]
        dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

        mask = dist_from_center < radius
        self.circular_mask = (
            torch.nn.Parameter(
                torch.from_numpy(
                    mask[newaxis, :, :, newaxis, newaxis, newaxis]
                )
                , requires_grad=False))

    def forward(self, x: torch.Tensor):
        return x * self.circular_mask


class UpDownHConv2d(nn.Module):
    """Defining custom convolutional layer"""

    @staticmethod
    def init_weights_dict(layer_shape,
                          in_max_order: int,
                          out_max_order: int,
                          ring_count=None):
        """
        Initializes the dict of weights

        Args:
            layer_shape (list of ints): contains dimensions of the layer as a list
            in_max_order: maximum rotation order of input channels
            out_max_order: maximum rotation order of output channels
            ring_count (int): Number of rings for calculating the basis
        Returns:
            weights_dict: initialized dict of weights
        """

        rotation_orders = range(0, np.max([in_max_order, out_max_order]) + 1)
        weights_dict = {}
        for order in rotation_orders:
            if ring_count is None:
                ring_count = int(np.maximum(layer_shape[0] / 2, 2))
            sh = [ring_count, ] + list(layer_shape[2:])
            # Initializes the weights using initialization method of He
            stddev = 0.4 * np.sqrt(2.0 / np.prod(sh[:3]))
            weights_dict[order] = nn.Parameter(torch.normal(torch.zeros(*sh), stddev))
        return weights_dict

    def get_angle_samples_count(self):
        return np.maximum(np.ceil(np.pi * self.kernel_size), 101)

    @staticmethod
    def get_interpolation_weights(fs, m, n_rings, angle_samples):
        """
        Used to construct the steerable filters using Radial basis functions.
        The filters are constructed on the patches of n_rings using Gaussian
        interpolation. (Code adapted from the tf code of Worrall et al., CVPR, 2017)

        Args:
            fs (int): filter size for the H-net convolutional layer
            m (int): max. rotation order for the steerable filters
            n_rings (int): No. of rings for the steerable filters
            angle_samples (int): No. of angle samples

        Returns:
            norm_weights (numpy): contains normalized weights for interpolation
            using the steerable filters
        """

        mid = int(np.floor(fs / 2))
        # Variance of Gaussian resampling
        std_gauss = (mid / n_rings) / 2
        # We define below radii up to n_rings-0.5 (as in Worrall et al, CVPR 2017)
        radii = np.linspace(m != 0, mid - 2 * std_gauss, n_rings)
        # We define pixel centers to be at positions 0.5
        center_pt = np.asarray([fs, fs]) / 2.

        # Extracting the set of angles to be sampled

        # Choosing the sampling locations for the rings
        lin = (2 * np.pi * np.arange(angle_samples)) / angle_samples
        ring_locations = np.vstack([-np.sin(lin), np.cos(lin)])

        # Create interpolation coefficient coordinates
        coords = hnet_ops.get_l2_neighbors(center_pt, fs)

        # getting samples based on the chosen center_pt and the coords
        radii = radii[:, np.newaxis, np.newaxis, np.newaxis]
        ring_locations = ring_locations[np.newaxis, :, :, np.newaxis]
        diff = radii * ring_locations - coords[np.newaxis, :, np.newaxis, :]
        dist2 = np.sum(diff ** 2, axis=1)

        # Convert distances to weightings
        weights = np.exp(-0.5 * dist2 / (std_gauss ** 2))  # For bandwidth of 0.5

        # Normalizing the weights to calibrate the different steerable filters
        norm = np.sum(weights,
                      axis=2,
                      keepdims=True)
        assert np.alltrue(norm != 0), "Normalizing by zero weights"
        return np.divide(weights,
                         norm,
                         where=(norm != 0)
                         )

    @staticmethod
    def init_phase_dict(inp_channel_count, out_channel_count, in_max_order, out_max_order):
        """
        Initializes phase dict with phase offsets

            out_channel_count (int): number of output channels
            in_max_order: maximum rotation order of input channels
            out_max_order: maximum rotation order of output channels
        Returns:
            phase_dict: initialized dict of phase offsets
        """

        rotation_orders = range(0, np.max([in_max_order, out_max_order]) + 1)
        phase_dict = {}
        for order in rotation_orders:
            init = np.random.rand(1, 1, inp_channel_count, out_channel_count) * 2. * np.pi
            init = torch.from_numpy(init)
            phase = nn.Parameter(init)
            phase_dict[order] = phase
        return phase_dict

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 scale_factor: int,
                 in_max_order,
                 out_max_order,
                 stride=1,
                 padding=0,
                 phase=True,
                 stddev=0.4,
                 n_rings=None):
        super().__init__()
        '''
        initializer function for Harmonic convolution lite wrapper class

        Args:
            in_channels (int): Number of input channels
            out_channels (int ): Number of output channels (int)
            kernel_size (int): dimensions of the (square) filter
            stride (tuple of ints): tuple denoting strides for h and w directions. Similar
            to the original tf code as well as the pytorch tuple standard format for stride,
            we provide a 4-size tuple here. The dimensions N and c as per convention are
            also set to 1.(default (1,1,1,1))
            padding (int): amount of implicit zero paddings on both sides
            phase (boolean): to decide whether the phase-offset term is used (default True)
            in_max_order (int): max. order of rotation to be modeled(default: 1)
            out_max_order (int): max. order of rotation to be modeled(default: 1)
            stddev (float): scaling term for He weight initialization
        '''
        assert scale_factor >= 1
        self.scale_factor = scale_factor

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.phase = phase
        self.in_max_order = in_max_order
        self.out_max_order = out_max_order
        self.stddev = stddev
        self.n_rings = n_rings
        self.shape = [kernel_size, kernel_size, self.in_channels, self.out_channels]
        self.Q = UpscaleHConv2d.init_weights_dict(self.shape,
                                                  in_max_order=self.in_max_order,
                                                  out_max_order=self.out_max_order,
                                                  ring_count=self.n_rings)
        for k, v in self.Q.items():
            self.register_parameter('weights_dict_' + str(k), v)
        if self.phase:
            self.P = UpscaleHConv2d.init_phase_dict(self.in_channels,
                                                    self.out_channels,
                                                    in_max_order=self.in_max_order,
                                                    out_max_order=self.out_max_order)

            for k, v in self.P.items():
                self.register_parameter('phase_dict_' + str(k), v)
        else:
            self.P = None

        # low pass filters
        N = self.get_angle_samples_count()
        self.low_pass_filters = torch.nn.ParameterDict()
        for m in self.Q.keys():
            # Get the basis matrices built from the steerable filters
            weights = UpscaleHConv2d.get_interpolation_weights(self.kernel_size,
                                                               m=m,
                                                               n_rings=self.n_rings,
                                                               angle_samples=self.get_angle_samples_count())
            low_pass_filter = np.dot(dft(N)[m, :], weights).T
            cos_comp = torch.nn.Parameter(torch.from_numpy(np.real(low_pass_filter)).to(torch.get_default_dtype()),
                                          requires_grad=False)
            sin_comp = torch.nn.Parameter(torch.from_numpy(np.imag(low_pass_filter)).to(torch.get_default_dtype()),
                                          requires_grad=False)
            self.low_pass_filters[f"cos_comp_{m}"] = cos_comp
            self.low_pass_filters[f"sin_comp_{m}"] = sin_comp

    def get_filters(self) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Calculates filters in the form of weight matrices through performing
        single-frequency DFT on every ring obtained from sampling in the polar
        domain.

        Args:

        Returns:
            W (dict): contains the filter matrices
        """
        W = {}  # dict to store the filter matrices

        for m, r in self.Q.items():
            rsh = list(r.size())
            cos_comp = self.low_pass_filters[f'cos_comp_{m}']
            sin_comp = self.low_pass_filters[f'sin_comp_{m}']
            # Computing the projections on the rotational basis
            r = r.view(rsh[0], rsh[1] * rsh[2])
            ucos = torch.matmul(cos_comp, r).view(self.kernel_size, self.kernel_size, rsh[1], rsh[2])
            usin = torch.matmul(sin_comp, r).view(self.kernel_size, self.kernel_size, rsh[1], rsh[2])

            if self.P is not None:
                # Rotating the basis matrices
                ucos_ = torch.cos(self.P[m]) * ucos + torch.sin(self.P[m]) * usin
                usin = -torch.sin(self.P[m]) * ucos + torch.cos(self.P[m]) * usin
                ucos = ucos_
            W[m] = (ucos, usin)

        return W

    def forward(self, x: torch.Tensor):
        """
        Forward propagation function for the harmonic convolution operation

        Args:
            X (deep tensor): input feature tensor [Batch, Height, Width, Orders, Complex, Channels]

        Returns:
            R (deep tensor): output feature tensor obtained from harmonic convolution
        """
        # x
        input_size = x.shape[1]
        shp = x.shape
        up_reshaped_x = (x
                         .view(None, shp[1], shp[2], shp[3] * shp[4] * shp[5])
                         .permute([0, 3, 1, 2])
                         )
        # Upscale shape mini-batch x channels x [optional depth] x [optional height] x width.
        upscale_x = (
            torch.nn.functional.interpolate(up_reshaped_x, scale_factor=self.scale_factor, mode='bilinear')
            .permute([0, 2, 3, 1])
            .view([None, input_size * self.scale_factor, input_size * self.scale_factor, shp[3:]])
        )
        W = self.get_filters()
        R = hnet_ops.h_conv(upscale_x, W,
                            strides=self.stride,
                            padding=self.padding,
                            in_max_order=self.in_max_order,
                            out_max_order=self.out_max_order)
        down_reshaped_x = (R
                           .view(None, shp[1], shp[2], shp[3] * shp[4] * shp[5])
                           .permute([0, 3, 1, 2])
                           )
        result_x = (
            torch.nn.functional.interpolate(R, size=input_size, mode='bilinear')
            .permute([0, 2, 3, 1])
            .view([None, input_size, input_size, shp[3:]])
        )
        return R


class HOrderDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        assert 0 < p < 1, f"Variable p should be {p}"
        self.dropout_layer = nn.Dropout3d(p=p, inplace=True)
        self.order_dimension = 3

    def forward(self, x):
        assert x.ndim == 6, "Expecting input as [Batch, Height, Width, Orders, Complex, Channels]"
        for i in range(x.shape[3]):
            self.dropout_layer(x[:, :, :, i])
        return x


class HOrderAlphaDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        assert 0 < p < 1, f"Variable p should be {p}"
        self.dropout_layer = nn.FeatureAlphaDropout(p=p, inplace=True)
        self.order_dimension = 3

    def forward(self, x):
        assert x.ndim == 6, "Expecting input as [Batch, Height, Width, Orders, Complex, Channels]"
        for i in range(x.shape[3]):
            self.dropout_layer(x[:, :, :, i])
        return x
