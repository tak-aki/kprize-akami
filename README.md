# kprize-akami

This is the solution code for the Kaggle competition [Konwinski Prize](https://www.kaggle.com/competitions/konwinski-prize).

## Structure

The structure is designed to manage code, libraries, and submit notebooks, referencing [smly's workflow](https://ho.lc/blog/kaggle_code_submission/).

```
.
├── codes  # Contains the actual code to be executed. Treated as a dataset on Kaggle
├── deps   # Creates necessary libraries for submission as files. Can be imported from submit notebook by uploading and running as a kernel on Kaggle
└── sub    # Contains submission notebooks
```

## Environment Setup

### Installing Dependencies

```bash
uv sync
```

### Downloading Pre-trained Difficulty Model

```bash
cd input
kaggle datasets download kami634/kprize-akami-difficulty-model
unzip kprize-akami-difficulty-model.zip -d kprize-akami-difficulty-model
```

## Training Environment

The following machine environment was used for training:

- **GPU**: NVIDIA A100 80GB
- **Cloud**: Google Cloud
- **Instance Type**: a2-ultragpu-1g (12 vCPU, 6 cores, 170 GB memory)

## Local Training of Difficulty Model

```bash
uv run python -m local.train.exp004.run exp=70b_003
```

## Local Execution

```bash
uv run python -m local.main
```

## Submission Steps

1. To reflect changes to codes in the dataset, execute the following command in the `./codes` directory

   ```bash
   kaggle d version -m 'update' -r zip
   ```

2. If there are changes to libraries required for submission, edit `./deps/kprize-akami-deps.ipynb` and execute the following command in the `./deps` directory

   ```bash
   kaggle k push
   ```

3. To upload and run the submit notebook, execute the following command in the `./sub` directory
   ```bash
   kaggle k push
   ```

## References

- Final submission code: https://www.kaggle.com/code/kami634/kprize-akami-sub-kami?scriptVersionId=226698839
