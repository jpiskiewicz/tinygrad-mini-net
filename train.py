from tinygrad.tensor import Tensor
from net import Net
from dataset import Dataset


def dice_loss(prediction: Tensor, target: Tensor) -> Tensor:
  e = 1e-6
  intersection = prediction.mul(target).sum()
  return Tensor.ones().sub(intersection.add(e).div(prediction.sum().add(target.sum().add(e))))


def jaccard_loss(prediction: Tensor, target: Tensor) -> Tensor:
  return Tensor.ones().sub(prediction.mul(target).sum().div(prediction.sum().add(target.sum())))


if __name__ == "__main__":
  net = Net()
  dataset = Dataset("../UNet/tinygrad-UNet/dataset.safetensors")
  print(dice_loss(dataset.images[0], dataset.masks[0]).numpy())
  print(jaccard_loss(dataset.images[0], dataset.masks[0]).numpy())
  print(dataset.images[0].shape, dataset.masks[0].shape)
  print(dataset.images[0].binary_crossentropy(dataset.masks[0]).numpy())
  # I don't want to complicate the network by introducing another layer with
  # object IDs so we should remove this layer from the masks before feeding them to the network.
