"""archsheet spec for <MODEL NAME>.

Run:  python -m archsheet archsheet_spec.py -o docs/archsheet/<model> --png

build() is executed under `torch.device("meta")`, so *everything created inside it is
allocated on the meta device*: instantiate the model and make the example inputs here
and nothing touches real memory (a 70B model is fine).
"""
import torch

# from my_project.model import MyModel, MyConfig      # <- import the real model code


# ── 1. Symbols ────────────────────────────────────────────────────────────────
# Every integer that should appear as a *name* in the diagram instead of a number.
# The tool replaces concrete dims by these names (and by simple expressions of them:
# h·dh, 4d, dh/2, b·s, s+1 ...). RULES:
#   * every value must be UNIQUE (b=2 and kv=2 would be indistinguishable)
#   * dynamic dims (batch, seq, image size ...) must not equal any config dim:
#     use odd/prime values such as b=3, s=17 (and s must satisfy model constraints,
#     e.g. divisible by a patch size / window size if the model requires it)
#   * (value, "description") tuples are shown in the legend
B, S = 3, 17


def build():
    # cfg = MyConfig(...)                  # tiny/real config both work on meta
    # model = MyModel(cfg)
    model = ...

    return dict(
        model=model,
        # kwargs for forward (use `args=(...)` for positional)
        inputs={
            "input_ids": torch.zeros(B, S, dtype=torch.long),
        },
        symbols={
            "b": (B, "batch"),
            "s": (S, "sequence length"),
            # "d":  (cfg.hidden_size, "d_model"),
            # "h":  (cfg.num_attention_heads, "query heads"),
            # "kv": (cfg.num_key_value_heads, "kv heads"),
            # "dh": (cfg.head_dim, "head dim"),
            # "f":  (cfg.intermediate_size, "d_ff"),
            # "v":  (cfg.vocab_size, "vocab"),
        },
        title="<Model name>",
        # run=lambda m: m.some_other_entrypoint(...),   # optional: custom call instead of model(**inputs)
    )
