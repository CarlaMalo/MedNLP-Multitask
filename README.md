# MedNLP-Multitask

Transformer-based Clinical Text Analysis

Authors: Ke Wang, Carla Malo, Esther Tan

## Dataset

- **Source**: [argilla/medical-domain](https://huggingface.co/datasets/argilla/medical-domain)
- **Description**: Medical domain dataset for NLP tasks

## Environment Setup

### Unified Environment YAML (Foy any Task) - (CUDA 11.8)

To run any of the tasks, you can use the following .yml (configured for CUDA 11.8):

```
conda env create --file env_scripts/tasks_env.yml
conda activate tasks-nlp
python -m spacy download en_core_web_md
```

*Alternatively*, we provide the following scripts for reference:

#### Task 1 & Task 3 Environment Script - (CUDA 11.8)

1. Create the environment from the script:

`
bash env_scripts/build_env_tasks_1_3.sh
`

2. Activate the environment:

   `conda activate task3-nlp`

3. If needed, run the following command to turn it into a kernel for the jupyter notebook:

   `python -m ipykernel install --user --name=task3-nlp`


#### Task 2 Environment Script - (CPU friendly)

1. Create the environment from the script:

`
bash env_scripts/build_env_task_2.sh
`

2. Activate the environment:

   `conda activate task2-nlp`

3. If needed, run the following command to turn it into a kernel for the jupyter notebook:

   `python -m ipykernel install --user --name=task2-nlp`

### Notes

Always activate the appropriate environment before running the corresponding task:

    `conda activate tasks-nlp` for any task

    `conda activate task3-nlp` for tasks 1 and 3
    
    `conda activate task2-nlp` for task 2
