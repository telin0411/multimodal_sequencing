# Multimodal Event Sequencing

## To Train
```bash
sh xxx.sh
```

## To Evaluate
```bash
# Comment out the `--do_train`.
sh xxx.sh
```

# Process

1. Put the image pretrained files at
2. **Change the `WIKIHOW_DATA_ROOT` to yours.**


# Using Multi-Ref sets.


# Using Our Trained Model Weights
1. Change the `--output_dir` to the path you store the pretrained weights.
2. Change the `--iters_to_eval` to `best`.
