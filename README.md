# Maximal Brain Damage Without Data or Optimization: Disrupting Neural Networks via Sign-Bit Flips

[Paper](https://arxiv.org/pdf/2502.07408) | [Demo Notebook](demo.ipynb)

This repository provides an implementation of the sign-bit attacks introduced in *Maximal Brain Damage Without Data or Optimization: Disrupting Neural Networks via Sign-Bit Flips*. It includes a demonstration notebook with two examples:

- image classification with EfficientNet-B0
- language generation with Qwen3-30B-A3B-Thinking-2507

<p>
  <img src="./assets/dalmat.png" alt="Figure 1 from the paper" width="900">
</p>

## Repository

- `demo.ipynb`: end-to-end demonstration notebook
- `src/deep_neural_lesion/`: attack helpers used by the notebook
- `assets/`: the dalmatian image used in the notebook and the teaser figure from the paper

## Getting started

```bash
uv sync
uv run jupyter notebook demo.ipynb
```

## What the notebook covers

### Image classification

The image section uses EfficientNet-B0, available in the `timm` model repository, on a dalmatian image. It shows the clean prediction, then the effect of DNL, then the effect of 1P-DNL.

### Language generation

The language section uses `Qwen3-30B-A3B-Thinking-2507`. It first generates with the clean model and then reruns the same prompt after DNL and 1P-DNL.

## Citation

If you find this work useful, please cite:

```bibtex
@article{galil2025maximal,
  title={Maximal Brain Damage Without Data or Optimization: Disrupting Neural Networks via Sign-Bit Flips},
  author={Galil, Ido and Kimhi, Moshe and El-Yaniv, Ran},
  journal={Transactions on Machine Learning Research},
  year={2025},
  url={https://arxiv.org/pdf/2502.07408}
}
```
