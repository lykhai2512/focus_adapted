# FOCUS: Flexible Optimizable Counterfactual Explanations for Tree Ensembles

This repository is the official implementation of the [AAAI 2022 paper "FOCUS: Flexible Optimizable Counterfactual Explanations for Tree Ensembles"](https://arxiv.org/abs/1911.12199). 

## Requirements

To install requirements:

```setup
conda env create --file environment.yml
```

>📋 This will create a conda environment called tensorflow-py3


## Using FOCUS to generate counterfactual explanations

To train FOCUS for each dataset, run the following commands:

```train
python main.py --sigma=1.0 --temperature=1.0 --distance_weight=0.01 --lr=0.001 --opt=adam --model_name=<MODEL_NAME> --data_name=<DATA NAME> --model_type=<MODEL TYPE> --distance_function=<euclidean/cosine/etc>
```

>📋  This will create another folder in the main directory called 'results', where the results files will be stored.


## Pre-trained Models

The pretrained models are available in the models folder

| Dataset         | Model Type | Model File Name                          | Data File Name               |
|----------------|------------|------------------------------------------|------------------------------|
| cf_compas_num  | ADA        | model_ada_cf_compas_num_iter100_depth2_lr0.1 | cf_compas_num_data_test.tsv |
| cf_compas_num  | DT         | model_dt_cf_compas_num_depth4               | cf_compas_num_data_test.tsv |
| cf_compas_num  | RF         | model_rf_cf_compas_num_iter500_depth4       | cf_compas_num_data_test.tsv |
| cf_heloc       | ADA        | model_ada_cf_heloc_iter100_depth8_lr0.1     | cf_heloc_data_test.tsv      |
| cf_heloc       | DT         | model_dt_cf_heloc_depth4                    | cf_heloc_data_test.tsv      |
| cf_heloc       | RF         | model_rf_cf_wine_iter500_depth4             | cf_heloc_data_test.tsv      |
| cf_shop2       | ADA        | model_ada_cf_shop2_iter100_depth8_lr0.1     | cf_shop2_data_test.tsv      |
| cf_shop2       | DT         | model_dt_cf_shop2_depth4                    | cf_shop2_data_test.tsv      |
| cf_shop2       | RF         | model_rf_cf_wine_iter500_depth4             | cf_shop2_data_test.tsv      |
| cf_wine        | ADA        | model_ada_cf_shop2_iter100_depth8_lr0.1     | cf_wine_data_test.tsv       |
| cf_wine        | DT         | model_dt_cf_shop2_depth4                    | cf_wine_data_test.tsv       |
| cf_wine        | RF         | model_rf_cf_wine_iter500_depth4             | cf_wine_data_test.tsv       |

