# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tensor implementation."""
import numpy as np

from mindspore import log as logger
from mindspore.communication.management import get_rank, get_group_size
from . import dtype as mstype
from ._register_for_tensor import tensor_operator_registry
from .._c_expression import MetaTensor
from .._c_expression import Tensor as Tensor_
from .._checkparam import Validator as validator

__all__ = ['Tensor', 'MetaTensor', 'RowTensor', 'SparseTensor']
np_types = (np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64, np.float16,
            np.float32, np.float64, np.bool_)


class Tensor(Tensor_):
    """
    Tensor is used for data storage.

    Tensor inherits tensor object in C++.
    Some functions are implemented in C++ and some functions are implemented in Python.

    Args:
        input_data (Tensor, float, int, bool, tuple, list, numpy.ndarray): Input data of the tensor.
        dtype (:class:`mindspore.dtype`): Input data should be None, bool or numeric type defined in `mindspore.dtype`.
            The argument is used to define the data type of the output tensor. If it is None, the data type of the
            output tensor will be as same as the `input_data`. Default: None.

    Outputs:
        Tensor, with the same shape as `input_data`.

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> # initialize a tensor with input data
        >>> t1 = Tensor(np.zeros([1, 2, 3]), mindspore.float32)
        >>> assert isinstance(t1, Tensor)
        >>> assert t1.shape == (1, 2, 3)
        >>> assert t1.dtype == mindspore.float32
        ...
        >>> # initialize a tensor with a float scalar
        >>> t2 = Tensor(0.1)
        >>> assert isinstance(t2, Tensor)
        >>> assert t2.dtype == mindspore.float64
    """

    def __init__(self, input_data=None, dtype=None, shape=None, init=None):
        # If input data is numpy number, convert it to np array
        if isinstance(input_data, np_types):
            input_data = np.array(input_data)

        if ((input_data is not None and init is None) or (input_data is None and init is not None)) is False:
            raise TypeError("input_data and init can not be None at the same time.")

        # If input_data is tuple/list/numpy.ndarray, it's support in check_type method.
        if init is None:
            validator.check_value_type('input_data', input_data, (Tensor_, np.ndarray, list, tuple, float, int, bool),
                                       'Tensor')
            valid_dtypes = (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64,
                            np.float16, np.float32, np.float64, np.bool_)
            if isinstance(input_data, np.ndarray) and input_data.dtype not in valid_dtypes:
                raise TypeError(f"For Tensor, the input_data is a numpy array, "
                                f"but it's data type is not in supported list:\
                                {list(i.__name__ for i in valid_dtypes)}.")
            if isinstance(input_data, (tuple, list)):
                if np.array(input_data).dtype not in valid_dtypes:
                    raise TypeError(f"For Tensor, the input_data is {input_data} that contain unsupported element.")
            if dtype is not None:
                validator.check_type_name('dtype', dtype, mstype.number_type + (mstype.bool_,), "Tensor")

            if isinstance(input_data, np.ndarray) and (not input_data.flags['FORC']):
                input_data = np.ascontiguousarray(input_data)
            if dtype is None:
                Tensor_.__init__(self, input_data)
            else:
                Tensor_.__init__(self, input_data, dtype)
        else:
            Tensor_.__init__(self, dtype, shape)
        self._virtual_flag = False
        self.init = init

    def __deepcopy__(self, memodict):
        new_obj = Tensor(self)
        new_obj.init = self.init
        new_obj._virtual_flag = self._virtual_flag # pylint:disable=w0212
        return new_obj

    def __repr__(self):
        Tensor_.data_sync(self, False)
        return Tensor_.__repr__(self)

    def __add__(self, other):
        out = tensor_operator_registry.get('__add__')(self, other)
        return out

    def __eq__(self, other):
        if not isinstance(other, (int, float, Tensor)):
            return False
        #  bool type is not supported for `Equal` operator in backend.
        if self.dtype == mstype.bool_ or (isinstance(other, Tensor) and other.dtype == mstype.bool_):
            if isinstance(other, Tensor):
                return Tensor(np.array(self.asnumpy() == other.asnumpy()))
            return Tensor(np.array(self.asnumpy() == other))
        return tensor_operator_registry.get('__eq__')(self, other)

    def __ne__(self, other):
        if not isinstance(other, (int, float, Tensor)):
            return True
        #  bool type is not supported for `NotEqual` operator in backend.
        if self.dtype == mstype.bool_ or (isinstance(other, Tensor) and other.dtype == mstype.bool_):
            return Tensor(np.array(self.asnumpy() != other.asnumpy()))
        return tensor_operator_registry.get('__ne__')(self, other)

    def __hash__(self):
        return hash(id(self))

    def __mul__(self, other):
        out = tensor_operator_registry.get('__mul__')(self, other)
        return out

    def __neg__(self):
        out = tensor_operator_registry.get('__neg__')(self)
        return out

    def __bool__(self):
        data = self.asnumpy()
        if data.shape == ():
            return bool(data)
        if data.shape == (1,):
            return bool(data[0])
        raise ValueError("The truth value of an array with several elements is ambiguous.")

    def __pos__(self):
        return self

    def __iadd__(self, other):
        return self.__add__(other)

    def __radd__(self, other):
        out = tensor_operator_registry.get('__add__')(self, other)
        return out

    def __imul__(self, other):
        return self.__mul__(other)

    def __rmul__(self, other):
        out = tensor_operator_registry.get('__mul__')(self, other)
        return out

    def __truediv__(self, other):
        out = tensor_operator_registry.get('__truediv__')(self, other)
        return out

    def __rtruediv__(self, other):
        out = tensor_operator_registry.get('__truediv__')(other, self)
        return out

    def __sub__(self, other):
        out = tensor_operator_registry.get('__sub__')(self, other)
        return out

    def __isub__(self, other):
        return self.__sub__(other)

    def __rsub__(self, other):
        out = tensor_operator_registry.get('__sub__')(other, self)
        return out

    def __lt__(self, other):
        out = tensor_operator_registry.get('__lt__')(self, other)
        return out

    def __le__(self, other):
        out = tensor_operator_registry.get('__le__')(self, other)
        return out

    def __getitem__(self, index):
        if isinstance(index, int) and not isinstance(index, bool) and self.shape and index >= self.shape[0]:
            raise IndexError("index {} is out of bounds for axis 0 with size {}".format(index, self.shape[0]))
        out = tensor_operator_registry.get('__getitem__')(self, index)
        return out

    def __setitem__(self, index, value):
        out = tensor_operator_registry.get('__setitem__')(self, index, value)
        self.assign_value(out)
        return self

    def __gt__(self, other):
        out = tensor_operator_registry.get('__gt__')(self, other)
        return out

    def __ge__(self, other):
        out = tensor_operator_registry.get('__ge__')(self, other)
        return out

    def __len__(self):
        out = tensor_operator_registry.get('shape')(self)
        if out:
            return out[0]
        raise TypeError("Not support len of a 0-D tensor")


    def __mod__(self, other):
        return tensor_operator_registry.get('__mod__')(self, other)

    def __imod__(self, other):
        return self.__mod__(other)

    def __rmod__(self, other):
        return tensor_operator_registry.get('__mod__')(other, self)

    def __pow__(self, other):
        return tensor_operator_registry.get('__pow__')(self, other)

    def __floordiv__(self, other):
        return tensor_operator_registry.get('__floordiv__')(self, other)

    def __ifloordiv__(self, other):
        return self.__floordiv__(other)

    def __rfloordiv__(self, other):
        return tensor_operator_registry.get('__floordiv__')(other, self)

    def __str__(self):
        if self.dtype == mstype.type_none:
            return "Unknown Tensor type!"
        return str(self.asnumpy())

    @property
    def shape(self):
        """The shape of tensor is a tuple."""
        return self._shape

    @property
    def dtype(self):
        """The dtype of tensor is a mindspore type."""
        return self._dtype

    @property
    def size(self):
        """The size reflects the total number of elements in tensor."""
        return self._size

    @property
    def ndim(self):
        """The ndim of tensor is an integer."""
        return len(self._shape)

    @property
    def has_init(self):
        """tensor is inited."""
        return self.init is not None

    @property
    def itemsize(self):
        """The length of one tensor element in bytes."""
        return self._itemsize

    @property
    def strides(self):
        """The tuple of bytes to step in each dimension when traversing a tensor."""
        return self._strides

    @property
    def nbytes(self):
        """The total number of bytes taken by the tensor."""
        return self._nbytes

    @property
    def T(self):
        """The transposed tensor."""
        return self.transpose()

    @property
    def virtual_flag(self):
        """Mark tensor is virtual."""
        return self._virtual_flag

    @virtual_flag.setter
    def virtual_flag(self, value):
        """The setter of virtual_flag."""
        if not isinstance(value, bool):
            raise TypeError("virtual_flag must be bool.")
        self._virtual_flag = value

    @staticmethod
    def from_numpy(array):
        """Convert numpy array to Tensor without copy data."""
        return Tensor(Tensor_.from_numpy(array))

    def asnumpy(self):
        """Convert tensor to numpy array."""
        return Tensor_.asnumpy(self)

    def all(self, axis=(), keep_dims=False):
        """
        Check all array elements along a given axis evaluate to True.

        Args:
            axis (Union[None, int, tuple(int)): Dimensions of reduction,
                when axis is None or empty tuple, reduce all dimensions.
                Default: (), reduce all dimensions.
            keep_dims (bool): Whether to keep the reduced dimensions.
                Default : False, don't keep these reduced dimensions.

        Returns:
            Tensor, has the same data type as x.
        """

        if axis is None:
            axis = ()
        return tensor_operator_registry.get('all')(keep_dims)(self, axis)

    def any(self, axis=(), keep_dims=False):
        """
        Check any array element along a given axis evaluate to True.

        Args:
            axis (Union[None, int, tuple(int)): Dimensions of reduction,
                when axis is None or empty tuple, reduce all dimensions.
                Default: (), reduce all dimensions.
            keep_dims (bool): Whether to keep the reduced dimensions.
                Default : False, don't keep these reduced dimensions.

        Returns:
            Tensor, has the same data type as x.
        """

        if axis is None:
            axis = ()
        return tensor_operator_registry.get('any')(keep_dims)(self, axis)

    def view(self, *shape):
        r"""
        Reshape the tensor according to the input shape.

        Args:
            shape (Union(tuple[int], \*int)): Dimension of the output tensor.

        Returns:
            Tensor, has the same dimension as the input shape.
        """
        if not shape:
            raise ValueError("The shape variable should not be empty")
        if isinstance(shape[0], tuple):
            if len(shape) != 1:
                raise ValueError(f"Only one tuple is needed, but got {shape}")
            shape = shape[0]
        return tensor_operator_registry.get('reshape')()(self, shape)

    def expand_as(self, x):
        """
        Expand the dimension of target tensor to the dimension of input tensor.

        Args:
            shape (Tensor): The input tensor. The shape of input tensor must obey
                the broadcasting rule.

        Returns:
            Tensor, has the same dimension as input tensor.
        """
        return tensor_operator_registry.get('broadcast_to')(x.shape)(self)

    def abs(self):
        """
        Return absolute value element-wisely.

        Returns:
            Tensor, has the same data type as x.
        """
        return tensor_operator_registry.get('abs')()(self)

    def mean(self, axis=(), keep_dims=False):
        """
        Reduces a dimension of a tensor by averaging all elements in the dimension.

        Args:
            axis (Union[None, int, tuple(int), list(int)]): Dimensions of reduction,
                when axis is None or empty tuple, reduce all dimensions.
                Default: (), reduce all dimensions.
            keep_dims (bool): Whether to keep the reduced dimensions.
                Default : False, don't keep these reduced dimensions.

        Returns:
            Tensor, has the same data type as x.
        """
        if axis is None:
            axis = ()
        return tensor_operator_registry.get('mean')(keep_dims)(self, axis)

    def transpose(self, *axes):
        """
        Returns a view of the array with axes transposed.

        For a 1-D array this has no effect, as a transposed vector is simply the
        same vector. For a 2-D array, this is a standard matrix transpose. For an
        n-D array, if axes are given, their order indicates how the axes are permuted
        (see Examples). If axes are not provided and a.shape = (i[0], i[1],...
        i[n-2], i[n-1]), then a.transpose().shape = (i[n-1], i[n-2], ... i[1], i[0]).

        Args:
            axes(Union[None, tuple(int), list(int), n ints], optional):
                None or no argument: reverses the order of the axes.
                Tuple of ints: i in the j-th place in the tuple means a’s i-th
                    axis becomes a.transpose()’s j-th axis.
                n ints: this form is intended simply as a `convenience alternative
                    to the tuple form.

        Returns:
            Tensor, has the same dimension as input tensor, with axes suitably permuted.
        """
        perm = validator.check_transpose_axis(axes, self.ndim)
        return tensor_operator_registry.get('transpose')()(self, perm)

    def reshape(self, *shape):
        """
        Gives a new shape to an array without changing its data.

        Args:
            shape(Union[int, tuple(int), list(int)]): The new shape should be compatible
            with the original shape. If an integer, then the result will be a 1-D
            array of that length. One shape dimension can be -1. In this case, the
            value is inferred from the length of the array and remaining dimensions.

        Returns:
            reshaped_tensor(Tensor): This will be a new view object if possible;
            otherwise, it will be a copy.
        """
        new_shape = validator.check_reshape_shp(shape)
        return tensor_operator_registry.get('reshape')()(self, new_shape)

    def ravel(self):
        """
        Returns a contiguous flattened tensor.
        A 1-D tensor, containing the elements of the input, is returned.

        Returns:
            Tensor, has the same data type as x.
        """
        reshape_op = tensor_operator_registry.get('reshape')()
        return reshape_op(self, (-1,))

    def flatten(self, order='C'):
        """
        Returns a copy of the array collapsed into one dimension.

        Args:
            order (str, optional): Can choose between `C` and `F`. `C` means to
            flatten in row-major (C-style) order. ‘F’ means to flatten in column-major
            (Fortran- style) order. Only `C` and `F` are supported.

        Returns:
            Tensor, has the same data type as x.
        """
        reshape_op = tensor_operator_registry.get('reshape')()
        trans_op = tensor_operator_registry.get('transpose')()

        order = validator.check_flatten_order(order)
        if order == 'C':
            return reshape_op(self, (-1,))

        perm = tuple(range(self.ndim-1, -1, -1))
        return reshape_op(trans_op(self, perm), (-1,))

    def swapaxes(self, axis1, axis2):
        """
        Interchanges two axes of a tensor.

        Args:
            axis1 (int): First axis.
            axis2 (int): Second axis.

        Returns:
            Transposed tensor, has the same data type as the original tensor x.
        """
        axis1, axis2 = validator.check_swapaxes_axis((axis1, axis2), self.ndim)

        if axis1 == axis2:
            return self
        if axis1 > axis2:
            axis1, axis2 = axis2, axis1

        perm = tuple(range(0, self.ndim))
        new_perm = None
        if axis2 + 1 < self.ndim:
            new_perm = perm[0:axis1] + perm[axis2:axis2+1] + \
                perm[axis1+1:axis2] + perm[axis1:axis1+1] + perm[axis2+1:]
        else:
            new_perm = perm[0:axis1] + perm[axis2:axis2+1] + \
                perm[axis1+1:axis2] + perm[axis1:axis1+1]

        return tensor_operator_registry.get('transpose')()(self, new_perm)

    def squeeze(self, axis=None):
        """
        Removes single-dimensional entries from the shape of an tensor.

        Args:
            axis: Union[None, int, list(int), tuple(list)]. Default is None.

        Returns:
            Tensor, with all or a subset of the dimensions of length 1 removed.
        """
        if axis is None:
            return tensor_operator_registry.get('squeeze')(self)
        new_shape = validator.prepare_shape_for_squeeze(self.shape, axis)
        return tensor_operator_registry.get('reshape')()(self, new_shape)

    def astype(self, dtype, copy=True):
        """
        Returns a copy of the array, cast to a specified type.

        Args:
            dtype(Union[mstype.dtype, numpy.dtype, str]): Designated tensor dtype,
            can be in format of np.float32, mstype.float32 or `float32`. Default
            is mstype.float32.

            copy(bool, optional): By default, astype always returns a newly allocated
                tensor. If this is set to false, the input tensor is returned instead
                of a copy if possible.

        Returns:
            Tensor, with the designated dtype.
        """
        dtype = validator.check_astype_dtype(dtype)
        if not copy and dtype == self.dtype:
            return self
        return tensor_operator_registry.get('cast')(self, dtype)


    def init_data(self, slice_index=None, shape=None, opt_shard_group=None):
        """
        Get the tensor format data of this Tensor.

        Args:
            slice_index (int): Slice index of a parameter's slices.
                It is used when initialize a slice of a parameter, it guarantees that devices
                using the same slice can generate the same tensor.
            shape (list[int]): Shape of the slice, it is used when initialize a slice of the parameter.
            opt_shard_group(str): Optimizer shard group which is used in auto or semi auto parallel mode
                to get one shard of a parameter's slice.
        """
        if self.init is None:
            raise TypeError("init_data must be set Tensor.init, init can't be None")

        if shape is None:
            shape = self.shape

        try:
            arr = np.ndarray(shape, dtype=mstype.dtype_to_nptype(self.dtype))
        except ValueError:
            msg = "Error shape={}".format(shape)
            logger.error(msg)
            raise ValueError(msg)

        class seed_context:
            '''set and restore seed'''

            def __init__(self, init):
                self.init = init
                from .seed import get_seed
                global_seed = get_seed()
                self._np_seed = np.random.get_state()[1][0]
                self.need_set_seed = ((slice_index is not None) and (global_seed is None))

            def __enter__(self):
                if self.need_set_seed:
                    self.seed = self.init.seed
                    np.random.seed(slice_index)
                    self.init.seed = slice_index

            def __exit__(self, ptype, value, trace):
                if self.need_set_seed:
                    np.random.seed(self._np_seed)
                    self.init.seed, _ = self.seed

        with seed_context(self.init):
            self.init(arr)
        data = np.array(arr)
        if opt_shard_group:
            rank = get_rank(opt_shard_group)
            size = get_group_size(opt_shard_group)
            data = np.split(data, size)[rank]
        return Tensor(data, dtype=self.dtype)


    def to_tensor(self, slice_index=None, shape=None, opt_shard_group=None):
        """Return init_data()."""
        logger.warning("WARN_DEPRECATED: The usage of to_tensor is deprecated."
                       " Please use init_data")
        return self.init_data(slice_index, shape, opt_shard_group)


class RowTensor:
    """
    A sparse representation of a set of tensor slices at given indices.

    An RowTensor is typically used to represent a subset of a larger
    tensor dense of shape [L0, D1, .. , DN] where L0 >> D0.

    The values in indices are the indices in the first dimension of the slices
    that have been extracted from the larger tensor.

    The dense tensor dense represented by an RowTensor slices has
    `dense[slices.indices[i], :, :, :, ...] = slices.values[i, :, :, :, ...]`.

    RowTensor can only be used in the `Cell`'s construct method.

    It is not supported in pynative mode at the moment.

    Args:
        indices (Tensor): A 1-D integer Tensor of shape [D0].
        values (Tensor): A Tensor of any dtype of shape [D0, D1, ..., Dn].
        dense_shape (tuple): An integer tuple which contains the shape
            of the corresponding dense tensor.

    Returns:
        RowTensor, composed of `indices`, `values`, and `dense_shape`.

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> class Net(nn.Cell):
        ...     def __init__(self, dense_shape):
        ...         super(Net, self).__init__()
        ...         self.dense_shape = dense_shape
        ...     def construct(self, indices, values):
        ...         x = RowTensor(indices, values, self.dense_shape)
        ...         return x.values, x.indices, x.dense_shape
        >>>
        >>> indices = Tensor([0])
        >>> values = Tensor([[1, 2]], dtype=ms.float32)
        >>> out = Net((3, 2))(indices, values)
        >>> print(out[0])
        [[1. 2.]]
        >>> print(out[1])
        [0]
        >>> print(out[2])
        (3, 2)
    """

    def __init__(self, indices, values, dense_shape):
        "Init RowTensor"
        self.__indices = indices
        self.__values = values
        self.__dense_shape = dense_shape

    @property
    def indices(self):
        return self.__indices

    @property
    def values(self):
        return self.__values

    @property
    def dense_shape(self):
        return self.__dense_shape


class SparseTensor:
    """
    A sparse representation of a set of nonzero elememts from a tensor at given indices.

    SparseTensor can only be used in the `Cell`'s construct method.

    Pynative mode not supported at the moment.

    For a tensor dense, its SparseTensor(indices, values, dense_shape) has
    `dense[indices[i]] = values[i]`.

    Args:
        indices (Tensor): A 2-D integer Tensor of shape `[N, ndims]`,
            where N and ndims are the number of values and number of dimensions in
            the SparseTensor, respectively.
        values (Tensor): A 1-D tensor of any type and shape `[N]`, which
            supplies the values for each element in indices.
        dense_shape (tuple): A integer tuple of size `ndims`,
            which specifies the dense_shape of the sparse tensor.

    Returns:
        SparseTensor, composed of `indices`, `values`, and `dense_shape`.

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> class Net(nn.Cell):
        ...     def __init__(self, dense_shape):
        ...         super(Net, self).__init__()
        ...         self.dense_shape = dense_shape
        ...     def construct(self, indices, values):
        ...         x = SparseTensor(indices, values, self.dense_shape)
        ...         return x.values, x.indices, x.dense_shape
        >>>
        >>> indices = Tensor([[0, 1], [1, 2]])
        >>> values = Tensor([1, 2], dtype=ms.float32)
        >>> out = Net((3, 4))(indices, values)
        >>> print(out[0])
        [1. 2.]
        >>> print(out[1])
        [[0 1]
         [1 2]]
        >>> print(out[2])
        (3, 4)
    """

    def __init__(self, indices, values, dense_shape):
        "Init SparseTensor"
        self.__indices = indices
        self.__values = values
        self.__dense_shape = dense_shape

    @property
    def indices(self):
        return self.__indices

    @property
    def values(self):
        return self.__values

    @property
    def dense_shape(self):
        return self.__dense_shape


def _vm_compare(*args):
    """Implement `vm_compare` for tensor."""
    obj_str = args[-1]
    if obj_str == "shape":
        fn = getattr(args[0].asnumpy(), obj_str)
        return fn
    if len(args) == 2:
        fn = getattr(args[0].asnumpy(), obj_str)
        return Tensor(fn())
    if isinstance(args[0], Tensor):
        fn = getattr(args[0].asnumpy(), obj_str)
        y = args[1].asnumpy() if isinstance(args[1], Tensor) else args[1]
    else:
        obj_str = "__r" + obj_str[2:]
        fn = getattr(args[1].asnumpy(), obj_str)
        y = args[0]
    return Tensor(np.array(fn(y)))


tensor_operator_registry.register('vm_compare', _vm_compare)
