# MedNLP-Multitask

Transformer-based Clinical Text Analysis

Authors: Ke Wang, Carla Malo, Esther Tan

## Dataset

- **Source**: [argilla/medical-domain](https://huggingface.co/datasets/argilla/medical-domain)
- **Description**: Medical domain dataset for NLP tasks

## Environment Setup

This project uses separate Python environments for different tasks to ensure package compatibility.

Tasks 1 to 3 are therefore kept separated due to incompatible dependency versions across tasks.

### Task 1 & Task 3 Environment

1. Create the environment from the script:

`
bash env_scripts/build_env_tasks_1_3.sh
`

2. Activate the environment:

   `conda activate task3-nlp`

3. If needed, run the following command to turn it into a kernel for the jupyter notebook:

   `python -m ipykernel install --user --name=task3-nlp`

*Alternatively*, the .yml configuration can be used:

```
conda env create --file env_scripts/task3_env.yml
conda activate task3-nlp
python -m spacy download en_core_web_md
```

### Task 2 Environment

1. Create the environment from the script:

`
bash env_scripts/build_env_task_2.sh
`

2. Activate the environment:

   `conda activate task2-nlp`

3. If needed, run the following command to turn it into a kernel for the jupyter notebook:

   `python -m ipykernel install --user --name=task2-nlp`

*Alternatively*, the .yml configuration can be used:

```
conda env create --file env_scripts/task2_env.yml
conda activate task2-nlp
python -m spacy download en_core_web_md
```

### Notes

Always activate the appropriate environment before running the corresponding task:

    `conda activate task3-nlp`for tasks 1 and 3
    
    `conda activate task2-nlp` for task 2
