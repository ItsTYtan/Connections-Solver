import json
import subprocess
import sys
from dotenv import load_dotenv
from datasets import concatenate_datasets, load_dataset
import os
import wandb
from generate_splits import generate_train_test_split

load_dotenv()
wandb.login(
    key=os.getenv("WANDB_API_KEY"),
    relogin=True,
    verify=False
)

###########################################
# ABLATION PARAMETERS
###########################################
params = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "learning-rate": 1e-5,
    "peft-config" : {
        "rank": 8,
        "alpha": 32,
        "dropout": 0.1,
    },
    "comments": "4 epochs instead of 1",
}

###########################################
# TRAINING PARAMETERS
###########################################
gpu = "0"
# gpu_utilization = 1
epochs = 4
save_steps = 200
per_device_train_batch_size = 1
gradient_accumulation_steps = 1
logging_steps = 10

learning_rate = params["learning-rate"]

experiment_no = len(os.listdir("./models"))
output_dir = "models/" + "ablation-" + str(experiment_no)

############### PEFT CONFIG ###############
use_lora = True



###########################################
# DATASET CONFIGURATION
###########################################
dataset, _ = generate_train_test_split()

###########################################
# WANDB PARAMETERS
###########################################
run_name = os.path.basename(output_dir)

wandb.init(
    project="connections-solver",
    name=run_name,
    tags=[params["model"], str(params["learning-rate"]), "peft"],
    notes=params["comments"]
)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel

    # torch.cuda.set_per_process_memory_fraction(gpu_utilization, 0)
    model = AutoModelForCausalLM.from_pretrained(
        params["model"],
        device_map='auto'
    )
    
    if (use_lora):
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False,     
            r=params["peft-config"]["rank"], 
            lora_alpha=params["peft-config"]["alpha"], 
            lora_dropout=params["peft-config"]["dropout"]
        )
        model = get_peft_model(model, peft_config)
    
    tokenizer = AutoTokenizer.from_pretrained(params["model"])

    training_args = SFTConfig(
        report_to="wandb",
        run_name=params["model"],
        logging_steps=logging_steps,
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,  
        gradient_accumulation_steps=gradient_accumulation_steps, 
        save_steps=save_steps,
        learning_rate=learning_rate,
        num_train_epochs=epochs
    )

    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        args=training_args,
    )

    try:
        trainer.train()

    except Exception as e:
        print(e)
        sys.exit(1)
    
    # Write ablation params to config file
    os.makedirs(os.path.dirname(output_dir + "/" + "ablation_params.json"), exist_ok=True)
    with open(output_dir + "/" + "ablation_params.json", "w") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)