import os
import json

import datasets
import torch
import transformers
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from torch.distributed.elastic.multiprocessing.errors import record

from lm_eval.arguments import EvalArguments
from lm_eval.generator import Generator
from lm_eval.tasks import ALL_TASKS


def parse_args():
    parser = HfArgumentParser(EvalArguments)

    parser.add_argument(
        "--model",
        default="/data/starcoder2-7b",
        help="Model to evaluate, provide a repo name in Hugging Face hub or a local path",
    )
    parser.add_argument(
        "--use_auth_token",
        action="store_true",
        help="Use the token generated when running `huggingface-cli login` (necessary for private model).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Use a model with custom code, this requires executing code by the author of the model.",
    )
    parser.add_argument(
        "--task",
        choices=ALL_TASKS,
        help=f"Evaluation tasks from {ALL_TASKS}",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation on each worker, can be larger for HumanEval",
    )
    parser.add_argument(
        "--max_length_generation",
        type=int,
        default=512,
        help="Maximum length of generated sequence (prompt+generation)",
    )
    parser.add_argument(
        "--max_new_toks",
        type=int,
        default=1024,
        help="Maximum number of generated tokens",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help="Model precision, from: fp32, fp16 or bf16",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to solve and evaluate from the benchmark",
    )
    parser.add_argument(
        "--postprocess",
        action="store_false",
        help="Postprocess model outputs before execution, always on except during generation tests",
    )
    parser.add_argument(
        "--allow_code_execution",
        action="store_true",
        help="Allow code evaluation to execute external/untrusted Python code on your machine",
    )
    parser.add_argument(
        "--load_generations_path",
        type=str,
        default=None,
        help="Path of file with previously generated solutions, if provided generation is skipped and only evaluation is done",
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default="outputs",
        help="Directory to save the results",
    )
    parser.add_argument(
        "--save_generations",
        action="store_true",
        help="Whether to save code generations",
    )
    parser.add_argument(
        "--save_generations_path",
        type=str,
        default="generations.json",
        help="Path for saving the code generations",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Gamma for WLLM",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help="Delta for WLLM",
    )
    parser.add_argument(
        "--detect_human_code",
        action="store_true",
    )
    parser.add_argument(
        "--generation_only",
        action="store_true",
    )
    parser.add_argument(
        "--attacked_gen_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--metric_output_path",
        type=str,
        default='evaluation_results.json',
    )
    parser.add_argument(
        "--by_only",
        action='store_true',
        help='whether to use only bayesian detector'
    )
    parser.add_argument(
        "--dip_only",
        action='store_true',
        help='whether to use only bayesian detector'
    )
    parser.add_argument(
        "--wo_model",
        action='store_true',
        help='whether to load model for detection'
    )
    parser.add_argument(
        "--gen_method",
        type=str,
        default='kgw',
        help='watermark method used in generation'
    )
    parser.add_argument(
        "--ent_thresh",
        type=float,
        default=0.6,
        help='entropy threshold for sweet'
    )
    parser.add_argument(
        "--by_topk",
        type=int,
        default=200,
        help='bayesian detection hyperparameter'
    )
    parser.add_argument(
        "--no_prompt_detect",
        action='store_true',
        help='whether to use no prompt detection'
    )
    parser.add_argument(
        "--detection_temp",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--no_rep_n",
        type=int,
        default=None,
    )
    return parser.parse_args()

@record
def main():
    args = parse_args()
    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()

    assert args.task is not None
    task_name = args.task

    accelerator = Accelerator()
    if accelerator.is_main_process:
        print(f"Selected Task: {task_name}")

    os.makedirs(args.outputs_dir, exist_ok=True)
    args.save_generations_path = os.path.join(args.outputs_dir, args.save_generations_path)
    args.metric_output_path = os.path.join(args.outputs_dir, args.metric_output_path)

    tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            # trust_remote_code=args.trust_remote_code,
            # use_auth_token=args.use_auth_token,
            truncation_side="left",
            padding_side="right",
        )
    if not tokenizer.eos_token:
        if tokenizer.bos_token:
            tokenizer.eos_token = tokenizer.bos_token
            print("bos_token used as eos_token")
        else:
            raise ValueError("No eos_token or bos_token found")
    tokenizer.pad_token = tokenizer.eos_token
    
    dict_precisions = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    if args.wo_model:
        model = None
    else:
        print(f"Loading tokenizer and model (in {args.precision})")
        model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=dict_precisions[args.precision],
                # trust_remote_code=args.trust_remote_code,
                # use_auth_token=args.use_auth_token,
            )

    generator = Generator(accelerator, model, tokenizer, args)
    if args.generation_only:
        if accelerator.is_main_process:
            print("generation mode only")
        generations, references = generator.generate_text(task_name)
        if accelerator.is_main_process:
            with open(args.save_generations_path, "w") as fp:
                json.dump(generations, fp)
                print(f"generations were saved at {args.save_generations_path}")
    else:
        results = {}
        # merge old results
        if os.path.exists(args.metric_output_path):
            results = json.load(open(args.metric_output_path))
        
        if args.by_only:
            new_results = generator.eval_bayesian_detect(task_name)
        elif args.dip_only:
            new_results = generator.eval_dip_detect(task_name)
        else:
            new_results = generator.evaluate(task_name)
        
        results["config"] = vars(args)
        results.update(new_results)
        
        dumped = json.dumps(results, indent=2)
        with open(args.metric_output_path, "w") as f:
            f.write(dumped)



if __name__ == '__main__':
    main()