# UI-RFT: Reinforcement Fine-Tuning for GUI Grounding

<font size=4><div align='center' > [[📖 Paper](https://arxiv.org/abs/2412.17589)] [[🤗 Checkpoints](https://huggingface.co/henryhe0123/UI-RFT-3B)] [[🤗 Datasets](https://huggingface.co/datasets/henryhe0123/UI-128)]</div></font>

## 🔥 Overview

We introduce **UI-RFT**, the first framework utilizing rule-based RL to enhance VLMs' GUI grounding capabilities.

This work is the course project of CS3316 Reinforcement Learning.

<a href="">
  <img src="assets/benchmark.png" alt="ben" >
</a>

## 📌 Takeaway

- Reinforced fine-tuning with only 128 high-quality samples significantly enhances GUI grounding.
- GUI grounding is a fundamental visual ability in VLMs, improved without needing long reasoning chains.

## ⚙️ Usage

### Training

To train VLM with verl:

```bash
./train.sh
```

### Evaluation

To test VLM on ScreenSpot:

```bash
python ./screenspot/test.py
```

To test VLM on ScreenSpot-Pro:

```bash
python ./screenspot/test-pro.py
```

## 🙏 Acknowledgement

We would like to express our sincere gratitude to Yan Ma for his invaluable and highly insightful discussions.
