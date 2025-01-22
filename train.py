#!/usr/bin/env python3

from tinygrad.tensor import Tensor
from net import Net
from dataset import Dataset


def dice_loss(prediction: Tensor, target: Tensor) -> Tensor:
  e = 1e-6
  intersection = prediction.mul(target).sum()
  return Tensor.ones().sub(intersection.add(e).div(prediction.sum().add(target.sum().add(e))))


def jaccard_loss(prediction: Tensor, target: Tensor) -> Tensor:
  return Tensor.ones().sub(prediction.mul(target).sum().div(prediction.sum().add(target.sum())))


def total_loss(prediction: Tensor, target: Tensor) -> Tensor:
  return dice_loss(prediction, target).add(jaccard_loss(prediction, target)).add(prediction.binary_crossentropy(target))


if __name__ == "__main__":
  net = Net()
  dataset = Dataset("../UNet/tinygrad-UNet/dataset.safetensors")
  dataset.remove_object_ids()
  print(total_loss(net(dataset.images[Tensor.zeros(1, dtype="int")]), dataset.masks[Tensor.zeros(1, dtype="int")]).numpy())