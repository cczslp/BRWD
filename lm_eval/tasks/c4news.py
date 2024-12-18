from lm_eval.base import Task
from datasets import load_dataset
import nltk
nltk.download('punkt')


class C4News(Task):
    '''
    We load c4 dataset from a processed load json file containing 5000 records.
    The prompt we used is the first two sentences of a news article.
    Users can modify this class to adapt to other datasource.
    '''

    DATASET_PATH = "json"

    DATA_FILE = "/data/c4_processed.json"

    def __init__(self):
        stop_words = []
        super().__init__(stop_words, requires_execution=False)
        self.need_eval = False

    def prepare_ds(self):
        dataset = load_dataset(path=self.DATASET_PATH, data_files=self.DATA_FILE, 
                               split='train').select(range(500))
        def truncate_text(example):
            words = nltk.word_tokenize(example["human_completion"])
            truncated = " ".join(words[:120])
            example["human_completion"] = truncated
            return example
        truncated_dataset = dataset.map(truncate_text)
        self.dataset = truncated_dataset

    def get_dataset(self):
        return self.dataset
    
    def get_prompt(self, doc):
        return doc['input']
    
    def get_full_data(self, doc):
        return f"{doc['input']} {doc['human_completion']}"
    
    def get_reference(self, doc):
        return doc['human_completion']
    
    def get_general_prompt(self):
        return "There are several sentences extracted from a news report."
    
    def postprocess_generation(self, generation, idx):
        return generation
    
    def process_results(self, generations, references):
        return None