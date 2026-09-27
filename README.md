# RT-1 (PyTorch)

PyTorch runtime for the OXE [RT-1](https://arxiv.org/abs/2212.06817) (TF)
policy used by [SimplerEnv](https://github.com/google-research/simpler_env):
`rt1_pytorch.RT1Inference` is a drop-in replacement for SimplerEnv's
TensorFlow class with zero TF at runtime.

## Setup

The USE-large (v5) language embedder is vendored as the git submodule
[`universal_sentence_encoder`](universal_sentence_encoder/)
(paragduttaiisc/universal_sentence_encoder_pytorch). Its `use_v5_embed.pt`
weights (~1.4 GB) are tracked with **Git LFS**, so clone with LFS and
recurse-submodules:

```bash
git lfs install
git clone --recurse-submodules https://github.com/paragduttaiisc/RT1-Pytorch.git
cd RT1-Pytorch
pip install -r universal_sentence_encoder/requirements.txt
```

If you already have a clone:

```bash
git submodule update --init
cd universal_sentence_encoder && git lfs pull && cd ..
```

> Without git-lfs the submodule's `use_v5_embed.pt` checks out as a small
> text pointer file and loading fails.

RT-1 policy weights live in `models/` (not tracked in git; see `main.py`
for the expected filenames, e.g. `models/rt1_logits_pt_model.pt`).

## Run

```bash
python main.py   # SimplerEnv google_robot_pick_coke_can, RT-1 Converged
```
