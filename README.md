# Pose_Estimation

Use ViTPose to do pose estimation

ViTPose paper: https://arxiv.org/abs/2204.12484
VitPose++ paper: https://arxiv.org/abs/2212.04246v3
ViTPose docs: https://huggingface.co/docs/transformers/model_doc/vitpose.

Large portions of this code just comes from ViTPose-transformers code [here](https://huggingface.co/spaces/hysts/ViTPose-transformers/blob/main/app.py). This was largely to write a batch tool without gradio in the mix.

## Execution

```sh
conda env create -f ./environment.yaml
conda activate pose_estimation
pip install python-magic-bin==0.4.14 # this does not work on first install. So just install on activation and it should work
python main.py -p "./tests/data/pexels-stockphotoartist-1034859.jpg"
```

## Test Images

Data is from [pexels](https://www.pexels.com), a free-use/stock photo site.

## TODO

- Add tests
- Cleanup tox
