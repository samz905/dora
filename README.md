# DoRA: Weight-Decomposed Low-Rank Adaptation

This repository implements **Weight-Decomposed Low-Rank Adaptation (DoRA)**, a novel parameter-efficient fine-tuning technique that outperforms the popular LoRA (Low-Rank Adaptation) method by decomposing pre-trained weights into magnitude and directional components.

## Overview

DoRA is an advanced adaptation technique introduced in [Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353) that enhances the learning capacity and training stability of LoRA by decomposing the pre-trained weight into two components:

- **Magnitude component**: Captures the norm of weight vectors
- **Directional component**: Represents the direction of weight vectors

This decomposition allows DoRA to achieve better performance compared to LoRA while maintaining similar computational efficiency.

## Key Features

- **Complete implementation** of both LoRA and DoRA layers from scratch
- **Comparative analysis** between vanilla training, LoRA, and DoRA
- **Practical demonstration** using MNIST dataset with multilayer perceptron
- **Performance comparison** showing DoRA's superior training dynamics
- **Modular design** for easy integration into existing models

## Implementation Details

### LoRA Layer
The LoRA implementation decomposes weight updates into two low-rank matrices A and B:
```python
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        # Low-rank decomposition: W + α·B·A
```

### DoRA Layer
DoRA extends LoRA by additionally learning magnitude parameters:
```python
class LinearWithDoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        # Weight decomposition: m * (W + α·B·A) / ||W + α·B·A||
```

## Requirements

```python
torch
torchvision
numpy
```

## Usage

1. **Clone the repository**:
```bash
git clone https://github.com/samz905/dora
cd DoRA
```

2. **Install dependencies**:
```bash
pip install torch torchvision numpy
```

3. **Run the notebook**:
```bash
jupyter notebook DoRA.ipynb
```

## References

- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

## Contributing

Feel free to open issues and pull requests for improvements and bug fixes.

## License

This project is open-source and available under the MIT License.
