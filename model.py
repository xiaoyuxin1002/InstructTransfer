import os
import time
import torch
import openai
import numpy as np


answer2str = lambda answer: str(answer[0]) if isinstance(answer, list) else str(answer)
option2str = lambda option: '\n'.join([f'({idx}) {each}' for idx, each in enumerate(option)])

demo_oneshot = lambda question, answer: f'Question:\n{question}\n\nAnswer:\n{answer2str(answer)}\n\n'
context = 'I gave a friend an instruction and five inputs. The friend read the instruction and wrote an output for every one of the inputs.\nHere are the input-output pairs:\n\n'

prompt_inst = lambda demo: f'{demo}The instruction was'
prompt_gen = lambda instruction, question: f'Instruction:\n{instruction}\n\nQuestion:\n{question}\n\nAnswer:\n'
prompt_cla = lambda instruction, question, option: f'Instruction:\n{instruction}\n\nQuestion:\n{question}\n\nOption:\n{option2str(option)}\n\nAnswer:\n'

message_hf = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, succinct, and polite answers to the user\'s questions.'
message_openai = 'You are a helpful assistant.'

template_hf = lambda prompt: f'### System:\n{message_hf}\n\n### User:\n{prompt}\n\n### Assistant:\n'
template_openai = lambda prompt: [{'role':'system', 'content':message_openai}, {'role':'user', 'content':prompt}]


class Model_HF:
    
    def __init__(self, tokenizer, model, projector=None):
        
        self.tokenizer = tokenizer
        self.model = model
        self.projector = projector
        
    def generate_instruction_vanilla(self, examples, task_type):
        
        input_ = self._prompt_inst(examples, task_type, if_context=True)
        input_ids = self._input_hard(input_)
        output_ = self._output_generate_hard(input_ids)
        
        return output_
    
    def generate_instruction_prepend(self, examples, soft_instruct, task_type):
        
        input_ = self._prompt_inst(examples, task_type)
        input_ids = self._input_hard(input_)
        input_embeds = self._input_soft(input_ids, soft_instruct, True)
        output_ = self._output_generate_soft(input_embeds)
        
        return output_
    
    def encode_instruction(self, hard_instruct):
        
        input_ids = self._input_hard(hard_instruct)
        rep = self._output_representation(input_ids)
        
        return rep
    
    def discover_instruction_prepend(self, examples, soft_full, hard_instruct, task_type):
        
        input_ = self._prompt_inst(examples, task_type)
        input_ids = self._input_hard(input_)
        input_embeds = self._input_soft(input_ids, soft_full, False)
        label_ids = self._input_hard(hard_instruct)
        loss = self._output_backprop(input_embeds, label_ids)
        
        return loss
    
    def generate_prediction(self, example, hard_instruct):
        
        input_ = self._prompt_gen(hard_instruct, example)
        input_ids = self._input_hard(input_)
        output_ = self._output_generate_hard(input_ids)
        
        return output_
    
    def classify_prediction(self, example, hard_instruct):
        
        input_ = self._prompt_cla(hard_instruct, example)
        input_ids = self._input_hard(input_)
        option_ids = [self._input_hard(option_) for option_ in example['option']]
        output_ = self._output_classify(input_ids, option_ids)
        
        return output_
    
    def _prompt_inst(self, examples, task_type, if_context=False):
        
        demo = ''.join([demo_oneshot(example['question'], example['answer'] if task_type == 'generation' else example['option'][example['answer']]) for example in examples])
        input_ = context + prompt_inst(demo) if if_context else prompt_inst(demo)
        input_ = template_hf(input_)
        
        return input_
    
    def _prompt_gen(self, hard_instruct, example):
        
        input_ = template_hf(prompt_gen(hard_instruct, example['question']))
        
        return input_
    
    def _prompt_cla(self, hard_instruct, example):
        
        input_ = template_hf(prompt_cla(hard_instruct, example['question'], example['option']))
        
        return input_
    
    def _input_hard(self, input_):
        
        input_ids = self.tokenizer(input_, return_tensors='pt').input_ids.to(self.model.device)
        
        return input_ids
    
    def _input_soft(self, input_ids, soft_instruct, if_project):
        
        input_embeds = self.model.get_input_embeddings()(input_ids)
        if if_project: soft_instruct = self.projector(soft_instruct)
        soft_instruct = soft_instruct.reshape(1, -1, input_embeds.shape[-1])
        input_embeds = torch.cat([soft_instruct, input_embeds], dim=1).type(self.model.dtype)
            
        return input_embeds
    
    def _output_representation(self, input_ids):
        
        with torch.no_grad():
            output_ = self.model(input_ids=input_ids, output_hidden_states=True)
        rep = output_.hidden_states[-1][0, -1, :].cpu().numpy()
        
        return rep
    
    def _output_backprop(self, input_embeds, label_ids):
        
        past_key_values = self.model(inputs_embeds=input_embeds, use_cache=True).past_key_values
        loss = self.model(input_ids=label_ids, labels=label_ids, past_key_values=past_key_values).loss
        
        return loss
    
    def _output_generate_hard(self, input_ids):
        
        output_ids = self.model.generate(input_ids=input_ids, max_new_tokens=100)
        output_ = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        
        return output_
    
    def _output_generate_soft(self, input_embeds):
        
        output_ids = self.model.generate(inputs_embeds=input_embeds, max_new_tokens=100)
        output_ = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return output_
    
    def _output_classify(self, input_ids, option_ids):
        
        with torch.no_grad():
            output_ = self.model(input_ids=input_ids, use_cache=True)
        past_key_values = output_.past_key_values
        past_last_logit = output_.logits[:, -1, :]
        
        perplexities = []
        for option_id in option_ids:
            with torch.no_grad():
                option_ = self.model(input_ids=option_id, past_key_values=past_key_values)
            option_logits = torch.cat([past_last_logit.unsqueeze(1), option_.logits], axis=1)[0, :-1, :]
            option_probs = torch.softmax(option_logits, dim=-1)[range(option_id.shape[1]), option_id[0]]
            perplexities.append(-torch.mean(torch.log(option_probs)).item())
        output_ = str(np.argmin(perplexities))
        
        return output_
    
    
class Model_OpenAI:
    
    def __init__(self, model):
        
        self.model = model
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
    def generate_instruction_vanilla(self, examples, task_type):
        
        input_ = self._prompt_inst(examples, task_type)
        output_ = self._output_generate(input_)
        
        return output_
        
    def generate_prediction(self, example, hard_instruct):
        
        input_ = self._prompt_gen(hard_instruct, example)
        output_ = self._output_generate(input_)
        if 'content_filter' in output_ or 'content filter' in output_: output_ = ''
        
        return output_
    
    def classify_prediction(self, example, hard_instruct):
        
        input_ = self._prompt_cla(hard_instruct, example)
        output_ = self._output_generate(input_)
        if 'content_filter' in output_ or 'content filter' in output_: output_ = ''
        else:
            if_digit = [letter.isdigit() for letter in output_]
            output_ = int(output_[if_digit.index(True)]) if True in if_digit else ''
        
        return output_
    
    def _prompt_inst(self, examples, task_type):
        
        demo = ''.join([demo_oneshot(example['question'], example['answer'] if task_type == 'generation' else example['option'][example['answer']]) for example in examples])
        input_ = template_openai(context + prompt_inst(demo))
        
        return input_
        
    def _prompt_gen(self, hard_instruct, example):
        
        input_ = template_openai(prompt_gen(hard_instruct, example['question']))
        
        return input_
    
    def _prompt_cla(self, hard_instruct, example):
        
        input_ = template_openai(prompt_cla(hard_instruct, example['question'], example['option']))
        
        return input_
        
    def _output_generate(self, input_):
        
        try:
            output_ = openai.ChatCompletion.create(model=self.model, messages=input_)
            output_ = output_['choices'][0]['message']['content']
            return output_
        
        except (openai.error.APIError, openai.error.ServiceUnavailableError, openai.error.Timeout) as e:
            retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
            time.sleep(retry_time)
            return self._output_generate(input_)