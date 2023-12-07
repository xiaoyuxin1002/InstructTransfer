import warnings
import dill as pk

from info import *
from util import *


def main():
    
    args = parse_args()
    info = Info(args)
    warnings.filterwarnings("ignore")

    myprint('='*20, info.FILE_STDOUT)
    myprint('Start Transfer Tuning of Hard Instructions for Black-Box Language Models', info.FILE_STDOUT)
    myprint('-'*20, info.FILE_STDOUT)
    set_seed(args.seed)
    record = init_record(args, info)
    
    myprint('Load the Source Histories and the Target Task', info.FILE_STDOUT)
    source_scores, target_dataset, target_model, tuning_model = load(args, info)

    myprint('Match the Learned Soft Instructions', info.FILE_STDOUT)
    source_scores = get_soft_instruct(args, info, source_scores, tuning_model)
    record[info.RECORD_OBSERVATION]['Source Observations'] = source_scores

    best_hard, query_round, query_num = ('', 0), 0, 0
    while query_num < args.query_budget:
        
        query_round += 1
        myprint('-'*20, info.FILE_STDOUT)
        myprint(f'Start Query Round {query_round}', info.FILE_STDOUT)
        myprint('-'*20, info.FILE_STDOUT)
    
        myprint('Collaborative Filtering to Get the Performance Representations', info.FILE_STDOUT)
        dataset2per, model2per = get_perf_rep(args, info, source_scores)
    
        myprint('Optimal Transport to Get the Performance Similarities', info.FILE_STDOUT)
        hard_pair2per_sim = get_perf_sim(info, source_scores, dataset2per, model2per)

        myprint('Language Model Encoding to Get the Semantic Representations', info.FILE_STDOUT)
        dataset2sem, hard2sem = get_sem_rep(args, info, source_scores, tuning_model)
        
        myprint('Initiate and Fit a Gaussian Process', info.FILE_STDOUT)
        gp, train_X = get_gp(args, info, source_scores, dataset2per, dataset2sem, model2per, hard_pair2per_sim, hard2sem)
        
        myprint('Start Querying with UCB and WS-CMA-ES', info.FILE_STDOUT)
        best_hard_, observed_scores, predictions_all, query_num = query(args, info, target_dataset, target_model, tuning_model, gp, train_X, query_num)
        
        source_scores += observed_scores
        best_hard = update(best_hard, best_hard_)
        record[info.RECORD_MODEL][f'GP Model Round {query_round}'] = gp
        record[info.RECORD_OBSERVATION][f'Queried Observations Round {query_round}'] = observed_scores
        record[info.RECORD_OBSERVATION][f'Queried Predictions Round {query_round}'] = predictions_all
        
    myprint('-'*20, info.FILE_STDOUT)
    myprint('Evaluate the Best Identified Hard Instruction on the Test Set', info.FILE_STDOUT)
    predictions_test = test(info, target_dataset, target_model, best_hard[0])
    record[info.RECORD_OBSERVATION][f'Test Predictions'] = predictions_test

    myprint('Save the Tuning Records', info.FILE_STDOUT)
    pk.dump(record, open(info.FILE_RECORD, 'wb'), -1)
    myprint('='*20, info.FILE_STDOUT)
    

if __name__=='__main__':
    main()