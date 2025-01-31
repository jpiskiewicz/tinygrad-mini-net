#!/usr/bin/env python3

from tinygrad.tensor import Tensor
from typing import Callable
from net import Net
from dataset import Dataset, TOTAL_EXAMPLES
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad.engine.jit import TinyJit


def dice_loss(prediction: Tensor, target: Tensor) -> Tensor:
  e = 1e-6
  intersection = prediction.mul(target).sum()
  return Tensor.ones().sub(intersection.add(e).div(prediction.sum().add(target.sum().add(e))))


def jaccard_loss(prediction: Tensor, target: Tensor) -> Tensor:
  return Tensor.ones().sub(prediction.mul(target).sum().div(prediction.sum().add(target.sum())))


def total_loss_factory() -> Callable[[Tensor, Tensor], Tensor]:
  last_loss = Tensor.ones()
  def f(prediction: Tensor, target: Tensor) -> Tensor:
    nonlocal last_loss
    last_loss = last_loss.mul(dice_loss(prediction, target).add(jaccard_loss(prediction, target)).add(prediction.binary_crossentropy(target)))
    return last_loss
  return f


if __name__ == "__main__":
  net = Net()
  opt = Adam(get_parameters(net), lr=0.0001)
  dataset = Dataset("../UNet/tinygrad-UNet/dataset.safetensors")
  dataset.remove_object_ids()
  total_loss = total_loss_factory()

  @TinyJit
  @Tensor.train()
  def step() -> Tensor:
    samp = Tensor.randint(1, high=TOTAL_EXAMPLES)
    batch, truth = dataset.images[samp], dataset.masks[samp]
    out = net(batch)
    loss = total_loss(out, truth).backward()
    opt.step()
    return loss

  for s in range(100 * TOTAL_EXAMPLES):
    loss = step()
    print(f"step: {s}; loss: {loss}")

    if s % 10000: net.save_state(s)