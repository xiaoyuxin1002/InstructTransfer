import os
import time


class Info:
    
    def __init__(self, args):
        
        self.num_demo = 5
        self.num_train = 50
        self.num_val = 20
        self.num_test = 100
        
        self.DIR_DATA = os.path.join(os.getcwd(), 'data')
        self.DIR_DATABASE = os.path.join(self.DIR_DATA, 'InstructBench')
        self.DIR_INPUT = os.path.join(self.DIR_DATABASE, 'input')
        self.DIR_OUTPUT = os.path.join(self.DIR_DATABASE, 'output')
        self.DIR_SOFT = os.path.join(self.DIR_DATABASE, 'soft')
        self.DIR_TUNING = os.path.join(self.DIR_DATABASE, 'tuning')
        
        self.FILE_INSTRUCTION = os.path.join(self.DIR_DATABASE, 'instruction.json')
        self.FILE_SCORE = os.path.join(self.DIR_DATABASE, 'score.json')
        self.FILE_SOFT = os.path.join(self.DIR_SOFT, f'{args.tuning_model}-{args.soft_token}.pkl')
        
        output_filename = time.strftime("%Y-%b-%d-%a-%H-%M-%S", time.localtime())
        self.FILE_STDOUT = os.path.join(self.DIR_TUNING, f'{output_filename}.txt')
        self.FILE_RECORD = os.path.join(self.DIR_TUNING, f'{output_filename}.pkl')
        
        self.RECORD_ARGS = 'Args'
        self.RECORD_OBSERVATION = 'Observation'
        self.RECORD_MODEL = 'Model'
        self.DEVICE_GPU = 'cuda:0'
        
        self.model2name = {'gpt-35': 'gpt-3.5-turbo-0613', 'gpt-4': 'gpt-4-0613', 
                           'llama-2-70b': 'upstage/Llama-2-70b-instruct-v2', 'llama-65b': 'upstage/llama-65b-instruct', 
                           'llama-30b': 'upstage/llama-30b-instruct', 'falcon-40b': 'tiiuae/falcon-40b-instruct', 
                           'falcon-7b': 'tiiuae/falcon-7b-instruct'}
        
        self.dataset2IDs = {'active_to_passive':0, 'ag_news':1, 'anli':2, 'antonyms':3, 'boolq':4, 
                            'cause_and_effect':5, 'cosmos_qa':6, 'diff':7, 'first_word_letter':8, 'hellaswag':9, 
                            'imdb':10, 'informal_to_formal':11, 'larger_animal':12, 'letters_list':13, 'negation':14, 
                            'nq_open':15, 'num_to_verbal':16, 'rhymes':17, 'second_word_letter':18, 'sentence_similarity':19, 
                            'sentiment':20, 'singular_to_plural':21, 'sum':22, 'synonyms':23, 'translation_en-de':24, 
                            'translation_en-es':25, 'translation_en-fr':26, 'trivia_qa':27, 'tweet_emotion':28, 'word_in_context':29}
        self.model2IDs = {'gpt-35':0, 'gpt-4':1, 'llama-2-70b':2, 'llama-65b':3, 'llama-30b':4, 'falcon-40b':5, 'falcon-7b':6}
        self.num_all_datasets = len(self.dataset2IDs)
        self.num_all_models = len(self.model2IDs)
        
        self.task2dataset = {'classification': ['ag_news', 'anli', 'boolq', 'cause_and_effect', 
                                                'cosmos_qa', 'hellaswag', 'imdb', 'larger_animal', 
                                                'sentence_similarity', 'sentiment', 'tweet_emotion', 'word_in_context'], 
                             'generation': ['active_to_passive', 'antonyms', 'diff', 'first_word_letter', 'informal_to_formal', 
                                            'letters_list', 'negation', 'nq_open', 'num_to_verbal', 'rhymes', 
                                            'second_word_letter', 'singular_to_plural', 'sum', 'synonyms', 
                                            'translation_en-de', 'translation_en-es', 'translation_en-fr', 'trivia_qa']}
        self.dataset2task = {dataset:task for task, datasets in self.task2dataset.items() for dataset in datasets}