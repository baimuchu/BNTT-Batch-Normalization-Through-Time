# BNTT-Batch-Normalization-Through-Time[Reproduction]
Group 16

This repository contains the source code associated with [arXiv preprint arXiv:2010.01729][arXiv preprint arXiv:2010.01729]

Accepted to Frontiers in Neuroscience (2021)

[arXiv preprint arXiv:2010.01729]: https://arxiv.org/abs/2010.01729

## Introduction

Spiking Neural Networks (SNNs) have recently emerged as an alternative to deep learning owing to sparse, asynchronous and binary event (or spike) driven processing, that can yield huge energy efficiency benefits on neuromorphic hardware. However, training high-accuracy and low-latency SNNs from scratch suffers from non-differentiable nature of a spiking neuron. To address this training issue in SNNs, we revisit batch normalization and propose a temporal Batch Normalization Through Time (BNTT) technique. Most prior SNN works till now have disregarded batch normalization deeming it ineffective for training temporal SNNs. Different from previous works, our proposed BNTT decouples the parameters in a BNTT layer along the time axis to capture the temporal dynamics of spikes. The temporally evolving learnable parameters in BNTT allow a neuron to control its spike rate through different time-steps, enabling low-latency and low-energy training from scratch. We conduct experiments on CIFAR-10, CIFAR-100, Tiny-ImageNet and event-driven DVS-CIFAR10 datasets. BNTT allows us to train deep SNN architectures from scratch, for the first time, on complex datasets with just few 25-30 time-steps. We also propose an early exit algorithm using the distribution of parameters in BNTT to reduce the latency at inference, that further improves the energy-efficiency.

## Reproduction Result

|                 |Dataset           | Time-step | Accuracy (%) | Accuracy (%) (Paper Original) |
|-----------------|------------------|-----------|--------------|---------------------------|
| BNTT            |CIFAR-10          | 25        | 84.6         | 90.5                      |
| BNTT+early exit | CIFAR-10         | 20        | 89.5         | 90.3                      |
| BNTT(FC)        | Sequential MNIST | 25        | 98.6         | 96.6                      |

## Citation
 
Please consider citing our paper:
 ```
 @article{kim2020revisiting,
  title={Revisiting Batch Normalization for Training Low-latency Deep Spiking Neural Networks from Scratch},
  author={Kim, Youngeun and Panda, Priyadarshini},
  journal={arXiv preprint arXiv:2010.01729},
  year={2020}
}
 ```
 
 