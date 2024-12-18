import re

from evaluate import load
from lm_eval.base import Task


class GSM8k(Task):
    
    DATASET_PATH = 'openai/gsm8k'

    DATASET_NAME = 'main'

    def __init__(self):
        super().__init__(
            stop_words=[],
            requires_execution=True,
        )
        dataset = self.dataset["train"].select(range(500))
        # extract answer value, and transform the answer into normal format 
        def extract_ans_val(example):
            ref_ans = example['answer']
            ref_ans = re.sub(r"<<.*?>>", "", ref_ans, flags=re.DOTALL)
            val_idx = ref_ans.index('####')
            ans_str = ref_ans[val_idx+5:].replace(',', '')
            example['ans_val'] = int(ans_str)
            example['answer'] = ref_ans[:val_idx - 1]
            return example
        new_ds = dataset.map(extract_ans_val)
        self.dataset = new_ds

    def get_dataset(self):
        return self.dataset

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        prompt = doc['question'] + '\nPlease reason step by step, and put your final answer within \\boxed{}.\n'
        return prompt

    def get_full_data(self, doc):
        return self.get_prompt(doc) + doc['answer']

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return doc['ans_val']

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation."""
        return generation

    @staticmethod
    def _extract_ans(text:str):
        pattern = r"\\boxed{(.*?)}"
        matches = re.findall(pattern, text)    
        if not matches:
            return None
        last_match = matches[-1]
        try:
            number = int(last_match)
            return number
        except ValueError:
            return None

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(int)
            list of int containing refrence answers
        """
        pass_info = []
        for gens, ref in zip(generations, references):
            gen = gens[0]
            gen_ans = self._extract_ans(gen)
            if gen_ans is not None and gen_ans == ref:
                pass_info.append(1)
            else:
                pass_info.append(0)
        
        result = {'solve-rate':sum(pass_info) / len(pass_info)}
        return result, pass_info
    
    def get_general_prompt(self):
        return 'Solve a math problem, please think step by step.\n' 