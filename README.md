---
title: Fresh or Rotten
emoji: 🍎
colorFrom: green
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Produce freshness classifier (EfficientNet-B0)
---

# Fresh or Rotten

Classifies produce as fresh or rotten from a photo or a live camera feed.

- **Model:** EfficientNet-B0 fine-tuned for binary classification, 97.2% test accuracy (F1 97.2%, AUC 0.997)
- **Baselines:** majority class 66.4%, Random Forest on classical features 94.8%
- **Produce covered:** apple, banana, bell pepper, bitter gourd, capsicum, carrot, cucumber, mango, okra, orange, potato, strawberry, tomato

Source code, training scripts, and the full evaluation: https://github.com/arnavmahale/DL-CV
