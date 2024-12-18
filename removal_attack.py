import libcst as cst
import requests
import random
import json
import string
import argparse
from tqdm import tqdm
from lm_eval import tasks

# collect local variables by traversing AST
class LocalVarsCollector(cst.CSTVisitor):
    
    def __init__(self):
        self.stack:list[str] = []
        self.nameSet:set[tuple[str]] = set()
        self.funcNameSet:set[tuple[str]] = set()
        self.nonLocalNameSet:set[str] = set()
        self.inFunction = False

    def clear_states(self):
        self.stack.clear()
        self.nameSet.clear()
        self.funcNameSet.clear()
        self.nonLocalNameSet.clear()
        self.inFunction = False

    def visit_Name(self, node:cst.Name):
        self.stack.append(node.value)
        if self.inFunction and (node.value not in self.nonLocalNameSet):
            self.nameSet.add(tuple(self.stack))
        else:
            self.nonLocalNameSet.add(node.value)

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        self.stack.append(node.name.value)
        self.inFunction = True
        funcName = tuple(self.stack + [node.name.value])
        self.funcNameSet.add(funcName)
        return True
    
    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:
        self.stack.append(node.name.value)
        return True

    def visit_Call(self, node: cst.Call):
        if isinstance(node.func, cst.Name):
            funcName = tuple(self.stack + [node.func.value])
        elif isinstance(node.func, cst.Attribute):
            funcName = tuple(self.stack + [node.func.attr.value])
        self.funcNameSet.add(funcName)

    def leave_Name(self, node):
        self.stack.pop()
    
    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        self.stack.pop()
        self.inFunction = False

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        self.stack.pop()
    
    def get_local_vars(self):
        return list(self.nameSet - self.funcNameSet)

# rename local variables
class RandomRenameLocalVariables(cst.CSTTransformer):

    def __init__(self, replace_ratio, seed:int=12):
        self.stack:list[str] = []
        self.replace_ratio = replace_ratio
        self.replace_map = None
        random.seed(seed)

    def clear_states(self):
        self.stack.clear()
        if self.replace_map:
            self.replace_map.clear()

    def set_local_vars(self, local_vars:list[tuple[str]]):
        if local_vars:
            replace_num = max(int(self.replace_ratio * len(local_vars)), 1)
            replace_list = random.sample(local_vars, replace_num)
            self.replace_map = {key:self._generate_random_name(2, 5) for key in replace_list}
        else:
            self.replace_map = {}

    def visit_Name(self, node):
        self.stack.append(node.value)

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        self.stack.append(node.name.value)
    
    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:
        self.stack.append(node.name.value)

    def leave_Name(self, original_node, updated_node):
        key = tuple(self.stack)
        self.stack.pop()
        if key in self.replace_map:
            return updated_node.with_changes(
                value=self.replace_map[key]
            )
        return updated_node

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.BaseStatement | cst.FlattenSentinel[cst.BaseStatement] | cst.RemovalSentinel:
        self.stack.pop()
        return updated_node

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.BaseStatement | cst.FlattenSentinel[cst.BaseStatement] | cst.RemovalSentinel:
        self.stack.pop()
        return updated_node

    def _generate_random_name(self, min_len, max_len):
        length = random.randint(min_len, max_len)
        return ''.join(random.choices(string.ascii_lowercase, k=length))

# base class for removal attack
class RemovalAttacker:

    def __init__(self, task_name, out_dir, new_res_file, 
                 old_res_file:str='machine_code.json', eval_res_file='evaluation_results.json'):
        self.out_dir = out_dir
        self.old_res_path = f'{out_dir}/{old_res_file}'
        self.new_res_path = f'{out_dir}/{new_res_file}'
        self.eval_res_path = f'{out_dir}/{eval_res_file}'
        self.task = tasks.get_task(task_name)
        self.ds = self.task.get_dataset()

    def load_generation_results(self) -> tuple[list[str], list[str]]:
        with open(self.old_res_path) as fp:
            generations = json.load(fp)
        generations = [gen[0] for gen in generations]

        task = self.task
        ds = self.ds
        n_samples = len(generations)
        prompt_contents = [task.get_prompt(ds[sid]) for sid in range(n_samples)]

        for idx in range(n_samples):
            pr:str = prompt_contents[idx]
            gen:str = generations[idx]
            assert gen.startswith(pr), 'generation should start with the prompt!'
            generations[idx] = gen[len(pr):]
        return prompt_contents, generations

    def filter_by_syntax(self, generations) -> tuple[list[int], list[str]]:
        with open(self.eval_res_path) as fp:
            eval_res:dict = json.load(fp)['pass_info']
        
        valid_ids = []
        for k, v in eval_res.items():
            cur_idx = int(k)
            result:str = v[0][1]['result']
            if result == 'passed' or result == 'failed: ' or result.startswith('failed: Test'):
                valid_ids.append(cur_idx)

        valid_gens = [generations[idx] for idx in valid_ids]
        return valid_ids, valid_gens

    def save_attack_res(self, prompt_contents, valid_ids, attacked_gens) -> None:
        assert len(valid_ids) == len(attacked_gens), 'the number of attacked generations should be equal to that of valid index'
        
        orig_ids = []
        prompts = []
        for j in range(len(valid_ids)):
            orig_idx = valid_ids[j]
            orig_ids.append(orig_idx)
            prompt = prompt_contents[orig_idx]
            prompts.append(prompt)
            attacked_gens[j] = prompt + attacked_gens[j]
        attack_res = {'orig_ids':orig_ids,
                      'prompts':prompts,
                      'attacked_gens':attacked_gens}

        with open(self.new_res_path, 'w') as fp:
            json.dump(attack_res, fp)
        
    def attack_single_text(self, text:str) -> str:
        pass

    def attack(self):
        prompts, generations = self.load_generation_results()
        valid_ids, valid_gens = self.filter_by_syntax(generations)
        attacked_gens = []

        for gen_text in tqdm(valid_gens):
            attacked_gens.append(self.attack_single_text(gen_text))
        self.save_attack_res(prompts, valid_ids, attacked_gens)
        
# variable rename attack
class SubstitutionAttacker(RemovalAttacker):

    def __init__(self, replace_ratio, seed=12, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.varsCollector = LocalVarsCollector()
        self.renameTransformer = RandomRenameLocalVariables(replace_ratio, seed)

    def attack_single_text(self, text: str) -> str:
        self.varsCollector.clear_states()
        self.renameTransformer.clear_states()

        module = cst.parse_module(text)
        module.visit(self.varsCollector)
        self.renameTransformer.set_local_vars(self.varsCollector.get_local_vars())
        new_module = module.visit(self.renameTransformer)

        return new_module.code

# rephrase code using codepal api, this attack is only applicable to relatively long code (200+ tokens)
# since watermark on a very short code is easily removed after rephrasing
class RephraseAttacker(RemovalAttacker):
    url = "https://api.codepal.ai/v1/code-rephraser/query"

    def __init__(self, api_key, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_key = api_key
        self.headers = {'Authorization':f'Bearer {api_key}'}

    def attack_single_text(self, text: str) -> str:
        payload = {'code':text}

        try:
            response = requests.post(self.url, headers=self.headers, data=payload)
            if response.status_code == 201:
                response_data = response.json()
                err = response_data['error']
                if err is not None:
                    print(err)
                    return ''
                return response_data['result']
        except requests.exceptions.RequestException as e:
            print(e)

        return ''



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--method',
        type=str,
        default='sub',
        help="attack method"
    )
    parser.add_argument(
        '--api_key',
        type=str,
    )
    parser.add_argument(
        '--task_name',
        type=str,
        default='mbpp'
    )
    parser.add_argument(
        '--replace_ratio',
        type=float,
        default=0.5
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=12
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='outputs/mbpp_kgw_outputs'
    )
    parser.add_argument(
        '--new_res_file',
        type=str,
        default='var_replaced_code.json'
    )
    args = parser.parse_args()

    if args.method == 'sub':
        attacker = SubstitutionAttacker(replace_ratio=args.replace_ratio, seed=args.seed,
                                    task_name=args.task_name, 
                                    out_dir=args.out_dir, 
                                    new_res_file=args.new_res_file)
    elif args.method == 'rephrase':
        attacker = RephraseAttacker(
            api_key=args.api_key,
            task_name=args.task_name,
            out_dir=args.out_dir,
            new_res_file=args.new_res_file
        )

    attacker.attack()