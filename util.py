import ot
import re
import json
import math
import time
import random
import string
import argparse
import dill as pk
import numpy as np
import pandas as pd
import surprise as sp
from cmaes import CMA
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances, euclidean_distances

import torch
import torch.nn as nn
import torch.optim as optim
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import *
from model import *
from kernel import *


def parse_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int)
    parser.add_argument('--soft_dim', type=int)
    parser.add_argument('--soft_token', type=int)
    parser.add_argument('--perf_dim', type=int)
    
    parser.add_argument('--fit_lr', type=float)
    parser.add_argument('--fit_epoch', type=int)
    
    parser.add_argument('--query_ucb_beta', type=float)
    parser.add_argument('--query_ws_gamma', type=float)
    parser.add_argument('--query_ws_alpha', type=float)
    
    parser.add_argument('--query_cma_bounds', type=str)
    parser.add_argument('--query_cma_popsize', type=int)
    parser.add_argument('--query_cma_iteration', type=int)
    
    parser.add_argument('--query_budget', type=int)
    parser.add_argument('--query_batch_size', type=int)
    
    parser.add_argument('--tuning_model', type=str)
    parser.add_argument('--target_model', type=str)
    parser.add_argument('--target_dataset', type=str)
    parser.add_argument('--source_histories', type=str)
    
    args = parser.parse_args()
    args.query_cma_bounds = [int(each) for each in args.query_cma_bounds.split(',')]
    args.source_histories = [tuple([each for each in pair.split(',')]) for pair in args.source_histories.split('|')]
    
    return args


def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    
def myprint(text, file):
    
    print(time.strftime("%Y %b %d %a, %H:%M:%S: ", time.localtime()) + text, file=open(file, 'a'), flush=True)


def init_record(args, info):
    
    record = {info.RECORD_ARGS:{}, info.RECORD_OBSERVATION:{}, info.RECORD_MODEL:{}}
    record[info.RECORD_ARGS]['Seed'] = args.seed
    record[info.RECORD_ARGS]['Soft Dimension'] = args.soft_dim
    record[info.RECORD_ARGS]['Soft Token Number'] = args.soft_token
    record[info.RECORD_ARGS]['Performance Dimension'] = args.perf_dim
    
    record[info.RECORD_ARGS]['Fitting Learning Rate'] = args.fit_lr
    record[info.RECORD_ARGS]['Fitting Epoch'] = args.fit_epoch
    
    record[info.RECORD_ARGS]['Query UCB Beta'] = args.query_ucb_beta
    record[info.RECORD_ARGS]['Query WS-CMA-ES Gamma'] = args.query_ws_gamma
    record[info.RECORD_ARGS]['Query WS-CMA-ES Alpha'] = args.query_ws_alpha
    
    record[info.RECORD_ARGS]['Query CMA-ES Bounds'] = args.query_cma_bounds
    record[info.RECORD_ARGS]['Query CMA-ES Population Size'] = args.query_cma_popsize
    record[info.RECORD_ARGS]['Query CMA-ES Iteration'] = args.query_cma_iteration
    
    record[info.RECORD_ARGS]['Query Budget'] = args.query_budget
    record[info.RECORD_ARGS]['Query Batch Size'] = args.query_batch_size
    
    record[info.RECORD_ARGS]['Tuning Model'] = args.tuning_model
    record[info.RECORD_ARGS]['Target Model'] = args.target_model
    record[info.RECORD_ARGS]['Target Dataset'] = args.target_dataset
    record[info.RECORD_ARGS]['Source Histories'] = args.source_histories
    
    myprint('Trial Hyperparameters', info.FILE_STDOUT)
    for k, v in record[info.RECORD_ARGS].items():
        if k != 'Source Histories':
            myprint(f'{k} = {v}', info.FILE_STDOUT)
        
    myprint('Source Histories = ', info.FILE_STDOUT)
    for idx in range(len(args.source_histories))[::2]:
        text = f'{args.source_histories[idx]}'
        if idx != len(args.source_histories)-1: text += f', {args.source_histories[idx+1]}'
        myprint(text, info.FILE_STDOUT)
    myprint('-'*20, info.FILE_STDOUT)
    
    return record


def load(args, info):
    
    all_softs = pk.load(open(info.FILE_SOFT, 'rb'))
    all_scores = json.load(open(info.FILE_SCORE, 'r'))
    
    source_scores = []
    for dataset_, model_ in args.source_histories:
        for hard_, score_ in all_scores[dataset_][model_].items():
            if (dataset_, hard_) not in all_softs: continue
            soft_ = all_softs[(dataset_, hard_)].flatten()
            source_scores.append([dataset_, model_, hard_, score_, soft_])
    target_dataset = Dataset(info, args.target_dataset)
    
    target_model_name = info.model2name[args.target_model]
    if 'gpt' in args.target_model:
        target_model = Model_OpenAI(target_model_name)
    else:
        tokenizer_ = AutoTokenizer.from_pretrained(target_model_name, use_fast=False)
        model_ = AutoModelForCausalLM.from_pretrained(target_model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map='auto')
        target_model = Model_HF(tokenizer_, model_)
        
    tuning_model_name = info.model2name[args.tuning_model]
    tokenizer_ = AutoTokenizer.from_pretrained(tuning_model_name, use_fast=False)
    model_ = AutoModelForCausalLM.from_pretrained(tuning_model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map='auto')
    projector_ = torch.nn.Linear(args.soft_dim, args.soft_token*model_.get_input_embeddings().weight.shape[1], bias=False).to(model_.device)
    tuning_model = Model_HF(tokenizer_, model_, projector_)
    
    for model_ in [target_model, tuning_model]:
        if type(model_) == Model_OpenAI: continue
        for param in model_.model.parameters():
            param.requires_grad = False
            
    for param in tuning_model.projector.parameters():
        torch.nn.init.uniform_(param, -1, 1)
        param.requires_grad = False
    
    return source_scores, target_dataset, target_model, tuning_model


def get_soft_instruct(args, info, source_scores, tuning_model):

    projector = tuning_model.projector.weight.detach()
    high_softs = torch.from_numpy(np.vstack([high_soft for _, _, _, _, high_soft in source_scores])).to(projector.device).T
    low_softs = torch.linalg.lstsq(projector, high_softs)[0].T.cpu().numpy()
    
    for each, low_soft in zip(source_scores, low_softs):
        each[-1] = low_soft
        
    return source_scores


def get_perf_rep(args, info, source_scores):
    
    best_perf = defaultdict(lambda: defaultdict(int))
    unique_datasets, unique_models = set(), set()
    for dataset_, model_, _, score_, _ in source_scores:
        best_perf[dataset_][model_] = max(score_, best_perf[dataset_][model_])
        unique_datasets.add(dataset_)
        unique_models.add(model_)
        
    dataset2did = {dataset_:did for did, dataset_ in enumerate(unique_datasets)}
    model2mid = {model_:mid for mid, model_ in enumerate(unique_models)}
        
    best_perf_df = {'datasetID':[], 'modelID':[], 'score':[]}
    for dataset_ in best_perf:
        for model_, score_ in best_perf[dataset_].items():
            best_perf_df['datasetID'].append(dataset2did[dataset_])
            best_perf_df['modelID'].append(model2mid[model_])            
            best_perf_df['score'].append(score_)
    
    best_perf_df = pd.DataFrame(best_perf_df)
    reader = sp.Reader(rating_scale=(0,1))
    performance = sp.Dataset.load_from_df(best_perf_df, reader)
    
    svd = sp.SVD(n_factors=args.perf_dim)
    svd.fit(performance.build_full_trainset())
    
    dataset_pers = np.concatenate([svd.pu, np.expand_dims(svd.bu, axis=1)], axis=-1)
    model_pers = np.concatenate([svd.qi, np.expand_dims(svd.bi, axis=1)], axis=-1)
    
    dataset2per = {info.dataset2IDs[dataset_]: dataset_pers[did] for dataset_, did in dataset2did.items()}
    model2per = {info.model2IDs[model_]: model_pers[mid] for model_, mid in model2mid.items()}
    
    if info.dataset2IDs[args.target_dataset] not in dataset2per: 
        dataset2per[info.dataset2IDs[args.target_dataset]] = np.mean(dataset_pers, axis=0)
    if info.model2IDs[args.target_model] not in model2per:
        model2per[info.model2IDs[args.target_model]] = np.mean(model_pers, axis=0)    
    
    return dataset2per, model2per


def get_perf_sim(info, source_scores, dataset2per, model2per):
    
    hard_dataset = defaultdict(lambda: defaultdict(int))
    for dataset_, model_, hard_, score_, _ in source_scores:
        hard_dataset[hard_][(dataset_, model_, score_)] += 1
        
    def _hard_features(hard_):
    
        frequency = np.array(list(hard_dataset[hard_].values()), dtype=float)
        frequency /= frequency.sum()
        
        feature_dataset, feature_model, feature_score = [], [], []
        for dataset_, model_, score_ in hard_dataset[hard_].keys():
            feature_dataset.append(dataset2per[info.dataset2IDs[dataset_]])
            feature_model.append(model2per[info.model2IDs[model_]])
            feature_score.append(score_)
            
        feature_dataset = np.stack(feature_dataset)
        feature_model = np.stack(feature_model)
        feature_score = np.expand_dims(np.array(feature_score), axis=1)
        
        return frequency, feature_dataset, feature_model, feature_score
    
    def _performance_similarity(hard1, hard2):
        
        frequency1, feature_dataset1, feature_model1, feature_score1 = _hard_features(hard1)
        frequency2, feature_dataset2, feature_model2, feature_score2 = _hard_features(hard2)
        
        cost = cosine_distances(feature_dataset1, feature_dataset2) + \
               cosine_distances(feature_model1, feature_model2) + \
               euclidean_distances(feature_score1, feature_score2)
        dist = ot.emd2(frequency1, frequency2, cost)
        sim = 1 / (1+dist)
        
        return sim
    
    hard_list = list(hard_dataset.keys())
    hard_pair2per_sim = {}
    for hid1, hard1 in enumerate(hard_list):
        for hard2 in hard_list[hid1+1:]:
            hard_pair2per_sim[tuple(sorted((hard1, hard2)))] = _performance_similarity(hard1, hard2)
    
    return hard_pair2per_sim


def get_sem_rep(args, info, source_scores, tuning_model):
    
    best_hard, hard_list = {}, set()
    for dataset_, _, hard_, score_, _ in source_scores:
        hard_list.add(hard_)
        if dataset_ not in best_hard or score_ > best_hard[dataset_][1]: 
            best_hard[dataset_] = (hard_, score_)
    hard2datasets = defaultdict(set)
    for dataset_, (hard_, _) in best_hard.items():
        hard2datasets[hard_].add(dataset_)
    
    hard2sem, dataset2sem = {}, {}
    for hard_ in hard_list:
        hard_sem_rep = tuning_model.encode_instruction(hard_)
        hard2sem[hard_] = hard_sem_rep
        for dataset_ in hard2datasets[hard_]:
            dataset2sem[info.dataset2IDs[dataset_]] = hard_sem_rep
    
    if info.dataset2IDs[args.target_dataset] not in dataset2sem: 
        dataset2sem[info.dataset2IDs[args.target_dataset]] = np.mean(np.stack(list(dataset2sem.values())), axis=0)
    
    return dataset2sem, hard2sem


def get_taskID(info, dataset_name, model_name):
    
    taskID = info.dataset2IDs[dataset_name] * info.num_all_models + info.model2IDs[model_name]
    
    return taskID


def get_ucb(args, gp, train_X):
    
    gp.eval()
    with torch.no_grad():
        
        mgd = gp(train_X)
        ucb = mgd.mean + args.query_ucb_beta * mgd.variance.sqrt()
        ucb = ucb.cpu().numpy()
        
    return ucb


def get_gp(args, info, source_scores, dataset2per, dataset2sem, model2per, hard_pair2per_sim, hard2sem):
    
    dataset2per_ = torch.zeros((info.num_all_datasets, list(dataset2per.values())[0].shape[0])).to(info.DEVICE_GPU)
    dataset2sem_ = torch.zeros((info.num_all_datasets, list(dataset2sem.values())[0].shape[0])).to(info.DEVICE_GPU)
    for did in dataset2per:
        dataset2per_[did] = torch.from_numpy(dataset2per[did])
        dataset2sem_[did] = torch.from_numpy(dataset2sem[did])

    model2per_ = torch.zeros((info.num_all_models, list(model2per.values())[0].shape[0])).to(info.DEVICE_GPU)
    for mid in model2per:
        model2per_[mid] = torch.from_numpy(model2per[mid])

    hard_list = []
    hard_sem_list = torch.zeros((len(source_scores), list(hard2sem.values())[0].shape[0]))
    soft_list = torch.zeros((len(source_scores), source_scores[0][-1].shape[0])).to(info.DEVICE_GPU)
    train_X = torch.zeros((len(source_scores), args.soft_dim+1)).to(info.DEVICE_GPU)
    train_Y = torch.zeros(len(source_scores)).to(info.DEVICE_GPU)

    for idx, (dataset_, model_, hard_, score_, soft_) in enumerate(source_scores):
        hard_list.append(hard_)
        hard_sem_list[idx] = torch.from_numpy(hard2sem[hard_])
        soft_list[idx] = torch.from_numpy(soft_)
        train_X[idx, 0] = get_taskID(info, dataset_, model_)
        train_X[idx, 1:] = torch.from_numpy(soft_)
        train_Y[idx] = score_

    hard_sem_matrix = torch.from_numpy(cosine_similarity(hard_sem_list)).to(info.DEVICE_GPU)
    hard_per_matrix = torch.eye(len(hard_list)).to(info.DEVICE_GPU)
    for hid1, hard1 in enumerate(hard_list):
        for hid2, hard2 in enumerate(hard_list):
            if hid1 >= hid2: continue
            if hard1 == hard2: per_sim = 1
            else: per_sim = hard_pair2per_sim[tuple(sorted((hard1, hard2)))]
            hard_per_matrix[hid1, hid2] = hard_per_matrix[hid2, hid1] = per_sim
            
    likelihood = GaussianLikelihood().to(info.DEVICE_GPU)
    gp = GP(info, dataset2per_, dataset2sem_, model2per_, hard_per_matrix, hard_sem_matrix, soft_list, train_X, train_Y, likelihood).to(info.DEVICE_GPU)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    
    mll.train()
    optimizer = optim.Adam(gp.parameters(), lr=args.fit_lr)
    for epoch_idx in range(args.fit_epoch):
        loss = -mll(gp(train_X), train_Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return gp, train_X


def query(args, info, target_dataset, target_model, tuning_model, gp, train_X, query_num):
    
    myprint('-'*20, info.FILE_STDOUT)
    myprint('Collect the UCB of the training data', info.FILE_STDOUT)
    train_ucb = get_ucb(args, gp, train_X)

    myprint('Calculate the task similarities', info.FILE_STDOUT)
    target_taskID = get_taskID(info, args.target_dataset, args.target_model)
    taskIDs = torch.unique(torch.cat([train_X[:,0], torch.tensor([target_taskID]).to(train_X.device)]), sorted=True)
    taskID_idx = torch.where(taskIDs == target_taskID)[0].item()

    gp.eval()
    with torch.no_grad():
        covar_task_dataset = gp.covar_module_dataset(taskIDs // gp.num_all_models).evaluate()
        covar_task_model = gp.covar_module_model(taskIDs % gp.num_all_models).evaluate()
        task_sim = covar_task_dataset * covar_task_model
        task_sim = task_sim[taskID_idx].cpu().numpy()
        
    myprint('Select Top-Performing Soft Instructions', info.FILE_STDOUT)
    if target_taskID not in train_X[:,0]: task_sim[taskID_idx] = 0
    task_sim /= task_sim.sum()
    task_gammas = np.rint(train_X.shape[0] * args.query_ws_gamma * task_sim).astype(int)

    selected_Xs = []
    for taskID, task_gamma in zip(taskIDs, task_gammas):
        if task_gamma == 0: continue

        taskID_indices = torch.where(train_X[:,0] == taskID)[0].cpu()
        task_ucb = train_ucb[taskID_indices]
        task_selected_indices = np.argsort(-task_ucb)[:task_gamma]
        task_selected_X = train_X[taskID_indices, 1:][task_selected_indices].cpu()
        
        selected_Xs.append(task_selected_X)
    selected_Xs = np.vstack(selected_Xs)

    myprint('Cluster the Selected Soft Instructions', info.FILE_STDOUT)
    kmeans = KMeans(n_clusters=args.query_batch_size, n_init='auto').fit(selected_Xs)
    batch_labels = kmeans.labels_
    
    myprint('Warm Start CMA-ES', info.FILE_STDOUT)
    cmas = []
    num_dim = selected_Xs.shape[1]
    bounds = np.tile(np.array(args.query_cma_bounds), (num_dim,1))

    for i in np.unique(batch_labels):
        batch_indices = np.where(batch_labels==i)[0]
        batch_Xs = selected_Xs[batch_indices]
        num_selected = batch_Xs.shape[0]

        ws_mean = batch_Xs.mean(0)
        batch_Xs -= ws_mean
        cov = np.matmul(batch_Xs[:,:,np.newaxis], batch_Xs[:,np.newaxis,:]).mean(0)
        cov += args.query_ws_alpha**2 * np.eye(num_dim)

        det_cov = np.linalg.det(cov)
        ws_sigma = math.pow(det_cov, 1/2/num_selected)
        ws_cov = cov / math.pow(det_cov, 1/num_selected)

        cma = CMA(mean=ws_mean, sigma=ws_sigma, cov=ws_cov, bounds=bounds, population_size=args.query_cma_popsize)
        cmas.append(cma)

    myprint('Identify the Soft Instruction with the Highest UCB', info.FILE_STDOUT)
    target_taskIDs = torch.tensor([target_taskID] * cma.population_size).unsqueeze(1).to(train_X.device)
    best_queries = []
    for idx, cma in enumerate(cmas):

        iteration = 0
        best_query = (None, 0)
        while not cma.should_stop() and iteration < args.query_cma_iteration:

            iter_X = torch.from_numpy(np.stack([cma.ask() for _ in range(cma.population_size)])).to(train_X.device)
            iter_X = torch.hstack([target_taskIDs, iter_X]).float()
            iter_ucb = get_ucb(args, gp, iter_X)
            cma.tell([(x.cpu().numpy(), -ucb) for x, ucb in zip(iter_X[:,1:], iter_ucb)])

            iter_best_idx = np.argmax(iter_ucb)
            if best_query[1] < iter_ucb[iter_best_idx]: best_query = (iter_X[iter_best_idx,1:], iter_ucb[iter_best_idx])
            iteration += 1
        best_queries.append(best_query)
    
    myprint('Evaluate the Identified Soft Instructions', info.FILE_STDOUT)
    examples = target_dataset.splits['train'][:info.num_demo]
    answers = [example['answer'] for example in target_dataset.splits['val']]
    predict = target_model.generate_prediction if target_dataset.task_type == 'generation' else target_model.classify_prediction
    
    myprint('-'*20, info.FILE_STDOUT)
    best_hard = ('', 0)
    observed_scores, predictions_all = [], []
    for idx, (soft_, ucb_) in enumerate(sorted(best_queries, key=lambda x:x[1], reverse=True)):
    
        hard_ = tuning_model.generate_instruction_prepend(examples, soft_, target_dataset.task_type).strip()
        query_num += 1
        myprint(f'Query Num {query_num} | Generated Hard Instruction: {hard_}', info.FILE_STDOUT)
        
        predictions = [predict(example, hard_) for example in target_dataset.splits['val']]
        score = round(evaluate(answers, predictions), 3)
        myprint(f'Query Num {query_num} | UCB: {ucb_:.4f} | Validation Score: {score:.4f}', info.FILE_STDOUT)

        best_hard = update(best_hard, (hard_, score))
        observed_scores.append([args.target_dataset, args.target_model, hard_, score, soft_.cpu().numpy()])
        predictions_all.append(predictions)
        if query_num >= args.query_budget: break
    
    return best_hard, observed_scores, predictions_all, query_num


def test(info, target_dataset, target_model, best_hard):
    
    myprint(f'Best Identified Hard Instruction: {best_hard}', info.FILE_STDOUT)
    predict = target_model.generate_prediction if target_dataset.task_type == 'generation' else target_model.classify_prediction
    predictions_test = [predict(example, best_hard) for example in target_dataset.splits['test']]
    
    answers = [example['answer'] for example in target_dataset.splits['test']]
    score = round(evaluate(answers, predictions_test), 3)
    myprint(f'Test Score: {score:.4f}', info.FILE_STDOUT)    
    myprint('-'*20, info.FILE_STDOUT)
    
    return predictions_test


def update(best_hard1, best_hard2):
    
    if best_hard1[1] > best_hard2[1]: return best_hard1
    elif best_hard1[1] < best_hard2[1]: return best_hard2
    elif len(best_hard1[0]) < len(best_hard2[0]): return best_hard1
    else: return best_hard2


def exact_match(answer, prediction):
    
    def _white_space_fix(text):
        return " ".join(text.split())

    def _remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def _remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def _lower(text):
        return text.lower()
    
    def _normalize_text(text):
        return _white_space_fix(_remove_articles(_remove_punc(_lower(text)))).strip()
    
    if isinstance(answer, list):
        return any(exact_match(each, prediction) for each in answer)
    return _normalize_text(str(answer)) == _normalize_text(str(prediction))
    
    
def evaluate(answers, predictions):
    
    scores = []
    for answer, prediction in zip(answers, predictions):
        scores.append(exact_match(answer, prediction))
    score = np.mean(scores)
    
    return score