#!/bin/bash

seed=0
soft_dim=10
soft_token=5
perf_dim=5

fit_lr=0.01
fit_epoch=500

query_ucb_beta=0.25
query_ws_gamma=0.1
query_ws_alpha=0.1

query_cma_bounds='-1,1'
query_cma_popsize=50
query_cma_iteration=10

query_budget=20
query_batch_size=5

tuning_model='llama-30b'
target_model='gpt-35'
target_dataset='negation'
source_histories='active_to_passive,falcon-40b|larger_animal,falcon-40b|sentiment,falcon-40b|singular_to_plural,falcon-40b|active_to_passive,falcon-7b|larger_animal,falcon-7b|sentiment,falcon-7b|singular_to_plural,falcon-7b|active_to_passive,llama-2-70b|larger_animal,llama-2-70b|sentiment,llama-2-70b|singular_to_plural,llama-2-70b|active_to_passive,llama-65b|larger_animal,llama-65b|sentiment,llama-65b|singular_to_plural,llama-65b'


python3 main.py --seed=${seed} --soft_dim=${soft_dim} --soft_token=${soft_token} --perf_dim=${perf_dim} --fit_lr=${fit_lr} --fit_epoch=${fit_epoch} --query_ucb_beta=${query_ucb_beta} --query_ws_gamma=${query_ws_gamma} --query_ws_alpha=${query_ws_alpha} --query_cma_bounds=${query_cma_bounds} --query_cma_popsize=${query_cma_popsize} --query_cma_iteration=${query_cma_iteration} --query_budget=${query_budget} --query_batch_size=${query_batch_size} --tuning_model=${tuning_model} --target_model=${target_model} --target_dataset=${target_dataset} --source_histories=${source_histories}
