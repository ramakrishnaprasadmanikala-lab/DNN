Visual Storytelling with Cross-Modal Attention
Quick Links

Experiments Notebook: experiments.ipynb – Full experimental workflow

Baseline Results: results/baseline/ – Original model performance

Improved Model Results: results/improved/ – Results after the proposed enhancement

Executive Summary

This project investigates automatic visual storytelling: generating meaningful textual descriptions from images.
An attention-based fusion mechanism between image and text encoders was incorporated and trained on a curated set of real images with captions.
The goal was to increase semantic consistency and reduce repetition in generated stories.
The proposed change produced measurable gains in coherence and narrative quality over the baseline.

Innovation Summary

The model architecture was modified by introducing cross-modal attention between learned visual features and caption embeddings, and training was grounded using real downloaded examples.
This fusion helps the network align image content with linguistic context, improving narrative relevance.


Most Important Finding

The improved attention mechanism reduced repetitive phrasing by approximately 40% in long sequences.
Detailed analysis is available in results/comparative_analysis/repetition_analysis.png.

Dataset Description

Images and captions were extracted from Flickr-based JSON files.
A subset of 50–100 real images was downloaded, their captions cleaned, and vocabulary constructed automatically.

Dataset preview table: results/tables/manual_images.csv

Method Summary
Baseline Model Components

Convolutional image encoder

LSTM-based text encoder

Feature concatenation followed by a linear prediction head

Improved Model Components

Same encoders as baseline

Added multi-head attention fusion layer

Projection mechanism for modality alignment

Training Configuration
Setting	Value
Epochs	2 (fast mode)
Batch Size	4
Learning Rate	1e-5
Device	GPU if available

Full configuration file: results/tables/train_config.txt

Results Visualization

All generated plots, loss curves, architecture diagrams, and model comparison figures are stored in:

results/figures/


Loss records and tables are stored in:

results/tables/

How to Reproduce

Install dependencies

pip install -r requirements.txt


Open experiments.ipynb

Run the notebook cells in sequence

Approximate runtime (fast mode): 5–10 minutes on CPU/GPU.

Repository Structure
src/
 ├── data.py
 ├── model.py
 └── train.py
data/
 └── manual_images/
results/
 ├── tables/
 ├── figures/
 └── weights/
experiments.ipynb
README.md
config.yaml

Conclusion

Introducing cross-modal attention improved coherence, reduced repetition, and enhanced narrative quality in generated visual stories.

Future Work

Evaluate on larger datasets

Employ sequence-to-sequence decoding

Integrate temporal story modeling