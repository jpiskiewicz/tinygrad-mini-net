from tinygrad.nn import Conv2d, ConvTranspose2d, BatchNorm
from tinygrad.tensor import Tensor
from typing import Tuple


class ExpandSqueeze:
  def __init__(self, in_channels: int, kernel_size: int):
    self.depthwise = Conv2d(in_channels, in_channels, kernel_size, padding="same", groups=in_channels)
    self.pointwise = Conv2d(in_channels, in_channels, 1)
    self.bn = BatchNorm(in_channels)

  def __call__(self, x: Tensor) -> Tensor: return self.bn(self.pointwise(self.depthwise(x.relu())))


class DualMultiscaleResidual:
  def __init__(self, in_channels: int):
    self.expandSqueeze1 = ExpandSqueeze(in_channels, 3)
    self.expandSqueeze2 = ExpandSqueeze(in_channels, 5)
    self.conv = Conv2d(in_channels, in_channels, 1)
    self.bn = BatchNorm(in_channels)

  def __call__(self, x: Tensor) -> Tensor:
    x1 = self.expandSqueeze2(x).add(self.expandSqueeze1(x))
    x2 = self.bn(self.conv(x1.relu())).add(x1)
    es2 = self.expandSqueeze2(x2)
    return self.expandSqueeze1(x2).add(es2).add(self.bn(self.conv(x))).relu()


class EncoderBlock:
  def __init__(self, in_channels: int):
    out_channels = in_channels * 2
    self.dmr = DualMultiscaleResidual(in_channels)
    self.strided_conv = Conv2d(in_channels, out_channels, 3, 2, 1)

  def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
    x = self.dmr(x)
    return x, self.strided_conv(x)


class DecoderBlock:
  def __init__(self, in_channels: int):
    out_channels = in_channels // 2
    self.transpose_conv = ConvTranspose2d(in_channels, out_channels, 3, 2, 1, 1)
    self.dmr = DualMultiscaleResidual(out_channels)

  def __call__(self, x: Tensor) -> Tensor: return self.dmr(self.transpose_conv(x))


class Net:
  def __init__(self):
    self.conv1 = Conv2d(1, 8, 3, padding="same")
    self.b1 = BatchNorm(8)
    self.e1 = EncoderBlock(8)
    self.conv2 = Conv2d(8, 8, 1)
    self.e2 = EncoderBlock(16)
    self.conv3 = Conv2d(16, 16, 1)
    self.b2 = BatchNorm(16)
    self.dmr = DualMultiscaleResidual(32)
    self.d1 = DecoderBlock(32)
    self.d2 = DecoderBlock(16)
    self.conv4 = Conv2d(8, 1, 1)
    self.b3 = BatchNorm(1)

  def __call__(self, x: Tensor) -> Tensor:
    print(x.shape, self.e1(self.b1(self.conv1(x)))[1].shape)
    dmr1, x = self.e1(self.b1(self.conv1(x)))
    dmr2, x = self.e2(x)
    x = self.d2(self.d1(self.dmr(x)).add(self.b2(self.conv3(dmr2)))).add(self.b1(self.conv2(dmr1)))
    x1 = self.conv4(self.b3(self.conv4(x)).relu()).sigmoid()
    assert isinstance(x1, Tensor)
    return x1