# Understanding Multimodal Procedural Knowledge by Sequencing Multimodal Instructional Manuals

## Data

Please download the data from these links: [WikiHow](https://drive.google.com/file/d/1WpBJ0ChBNaZeJxT-3rk7pdhEpIapi40V/view?usp=sharing), and [RecipeQA](https://drive.google.com/file/d/1MtXrN7ux-Ht98I8iEBRv7SM3c_TOQQlX/view?usp=sharing), and put the tar.gz files under the folder `data` to untar them. They should contain everything you need.

You can also use the notebook `demo_data.ipynb` to randomly inspect some data samples.

Download the necessary model weights from [here](https://drive.google.com/file/d/1qOuTU5RoHPncqo4kgivZ0c9xeyGQqQNE/view?usp=sharing) and untar it. Put the `pretrained_models` in this path (i.e., `.`) as they are downloaded LM pretrained weights that are required for our training.

## Requirements

You can run the pip installation with the provided `requirements.txt`, however, it encompasses all the libraries I installed for my particular conda environment, the most important packages are:
```bash
torch=1.8.0+cu111
transformers=3.4.0
tensorboardx=1.2
tensorboard=2.3.0
opencv-python=4.4.0.44
```
Other packages may not be that important and you can install them upon required from running the codes.

## To Train

The below is the sample pretraining script
```bash
sh scripts/wikihow_pretrain.sh
```
And the finetuning script:
```bash
sh scripts/wikihow_finetune.sh
```

## To Evaluate
For evaluation, simply comment out the `--do_train` flag in each script and make sure the `--iters_to_eval` and the `--output_dir` (and `--output_root` if you set it) are the correct paths you would like to evaluate on.
```bash
# Comment out the `--do_train`.
sh scripts/wikihow_finetune.sh
```

## Suggested Process

1. We recommend executing the `wikihow_image_only_pretrain.sh` first to obtain a good visual encoder, however, this step can be skipped if end-point performance is not of big concern.
2. Put the image pretrained files at some path and use them with the flag `--clip_visual_model_weights` for loading.
3. **Change the `WIKIHOW/RECIPEQA_DATA_ROOT`** in `datasets/wikihow.py` and `datasets/recipeQA.py` if you do not save them directly under the `data` folder.
4. Execute the training. FYI in our experiments we use A100 type GPUs.
5. Note that the split with `human` in the name are human inspected golden-test-set instead of a random split test set.

## Using Multi-Ref sets.

Simply change the tag in the `--eval_splits` to something like `test-acl22_human_multiref_multimodal` to fit your needs.

# Using Our Trained Model Weights
1. Change the `--output_dir` to the path you store the pretrained weights `paper_weights`.
2. Change the `--iters_to_eval` to `best`.

## Citation

Please cite our work using:
```
@inproceedings{wu2022procedural,
  title = {Understanding Multimodal Procedural Knowledge by Sequencing Multimodal Instructional Manuals},
  author = {Wu, Te-Lin and Spangher, Alex and Alipoormolabashi, Pegah and Freedman, Marjorie and Weischedel, Ralph and Peng, Nanyun},
  booktitle = {Proceedings of the Conference of the 60th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year = {2022}
}
```

Please contact [Te-Lin Wu](mailto:telinwu@g.ucla.edu) should you have any questions, thanks!
