import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from model import Text2Mol
from dataset import MoleculeDataset, MoleculeGraphDataset
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import torch
from transformers import AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from NoamOpt import NoamOpt
import os
from utils import set_seed, is_valid_smiles, calculate_similarity

import selfies as sfs
from rdkit import Chem, rdBase
import wandb

import warnings

warnings.filterwarnings('ignore')
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
rdBase.DisableLog('rdApp.error')

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12317'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def main(rank, args):
    set_seed(42)
    if torch.backends.mps.is_available():
        mac_run = True
    else:
        mac_run = False
    if not mac_run:
        setup(rank, args.world_size)

    run_name = "text2mol_" + args.text_encoder + "_" + args.molecule_decoder + "_" + args.dataset_name
    if mac_run or dist.get_rank() == 0:
        if args.use_wandb:
            print("Using wandb for logging.")
            try:
                wandb.login(key=args.wandb_token)
            except:
                pass
            run = wandb.init(project=run_name, config=args, name=run_name)
        else:
            print("Not using wandb for logging.")
    
    corrected_selfies = []
    corrected_smiles = []
    corrected_corpus = []
    import csv
    with open("data/train.txt") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, fieldnames=['cid', 'SMILES', 'description'])
        for n, line in enumerate(reader):
            try:
                smiles = Chem.MolToSmiles(Chem.MolFromSmiles(line['SMILES']))
                selfies = sfs.encoder(smiles)
            except:
                continue
            corrected_smiles.append(smiles)
            corrected_selfies.append(selfies)
            corrected_corpus.append(line['description'])
    train_data = list(zip(corrected_corpus, corrected_selfies, corrected_smiles))

    corrected_smiles = []
    corrected_selfies = []
    corrected_corpus = []
    with open("data/validation.txt") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, fieldnames=['cid', 'SMILES', 'description'])
        for n, line in enumerate(reader):
            try:
                smiles = Chem.MolToSmiles(Chem.MolFromSmiles(line['SMILES']))
                selfies = sfs.encoder(smiles)
            except:
                continue
            corrected_smiles.append(smiles)
            corrected_selfies.append(selfies)
            corrected_corpus.append(line['description'])
    
    valid_data = list(zip(corrected_corpus, corrected_selfies, corrected_smiles))

    molecule_tokenizer = AutoTokenizer.from_pretrained("huggingface/MolGen")
    cls_idx = molecule_tokenizer.cls_token_id
    eos_idx = molecule_tokenizer.eos_token_id
    mask_idx = molecule_tokenizer.mask_token_id
    pad_idx = molecule_tokenizer.pad_token_id


    model = Text2Mol(args.text_encoder, args.freeze_encoder, args.molecule_decoder)
    total_params = sum(p.numel() for p in model.parameters())

    # Trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Untrainable parameters 
    untrainable_params = total_params - trainable_params

    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Untrainable Parameters: {untrainable_params}")

    device = torch.device("mps") if not torch.cuda.is_available() else torch.device("cuda:{}".format(rank))
    model.to(device)
    model = DDP(model, device_ids=[device])
    train_dataset = MoleculeDataset(train_data)
    valid_dataset = MoleculeDataset(valid_data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer_adamw = torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    optimizer = NoamOpt(optimizer=optimizer_adamw, model_size=model.MolGen.config.hidden_size, factor=args.lr_factor, warmup=args.warm_up_steps)

    best_loss = 100
    best_similarity = 0
    epoch = 500
    for epoch_num in range(epoch):
        train_loss = []
        model.train()
        train_invalid_selfies = 0
        train_similarities = []
        for batch_idx, batch in enumerate(train_dataloader):
            input = batch
            selfies = input['selfies']
            selfies_tokens = molecule_tokenizer(selfies, return_tensors="pt", padding=True, truncation=True).to(device)
            selfies_ids = selfies_tokens['input_ids']
            input['prev_tokens'] = selfies_ids[:, :-1]

            selfies_label = selfies_ids[:, 1:]
            output_logits = model(input)
            eos_indices = []
            for g in output_logits.argmax(-1):
                eos_position = torch.nonzero(g == eos_idx, as_tuple=True)[0]
                if len(eos_position) > 0:
                    first_eos_index = eos_position[0]
                else:
                    first_eos_index = g.shape[0]
                eos_indices.append(first_eos_index)
            selfies_output = [molecule_tokenizer.decode(output_logits.argmax(-1)[g][:eos_indices[g]]).replace(" ", "") for g in
                              range(output_logits.argmax(-1).shape[0])]
            for i in range(output_logits.shape[0]):
                try:
                    smiles_output = sfs.decoder(selfies_output[i])
                except:
                    train_invalid_selfies += 1
                    continue
                if is_valid_smiles(smiles_output) and smiles_output != "":
                    similarity = calculate_similarity(smiles_output, input['smiles'][i])
                    if similarity is None:
                        train_invalid_selfies += 1
                        continue
                    train_similarities.append(similarity)
                else:
                    train_invalid_selfies += 1
            loss = criterion(output_logits.transpose(1, 2), selfies_label)
            loss = loss / args.accumulation_steps
            loss.backward()
            if (batch_idx + 1) % args.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            train_loss.append(loss.detach().cpu().numpy())
        train_loss = torch.tensor(np.mean(train_loss), dtype=torch.float32).to(device)
        train_similarity = torch.tensor(np.mean(train_similarities), dtype=torch.float32).to(device)
        train_invalid_selfies = torch.tensor(train_invalid_selfies).to(device)

        if not mac_run:
            dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_similarity, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_invalid_selfies, op=dist.ReduceOp.SUM)
        
        if args.skip_valid != True:
            if epoch_num % args.eval_epoch == 0:
                model.eval()
                with torch.no_grad():
                    test_invalid_selfies = 0
                    test_similarities = []
                    for batch_idx, batch in enumerate(valid_dataloader):
                        input = batch
                        output, sequence = model.sample_ar(input, temp=1, cls_idx=cls_idx, greedy=False)
                        eos_indices = []
                        for g in sequence:
                            eos_position = torch.nonzero(g == eos_idx, as_tuple=True)[0]
                            if len(eos_position) > 0:
                                first_eos_index = eos_position[0]
                            else:
                                first_eos_index = g.shape[0] - 1
                            eos_indices.append(first_eos_index)
                        sequence_np = sequence.detach().cpu().numpy()
                        selfies_output = []
                        for i in range(sequence_np.shape[0]):
                            line = sequence_np[i][:eos_indices[i]]
                            selfies_output.append(molecule_tokenizer.decode(line))
                        
                        for i in range(output.shape[0]):
                            try:
                                smiles_output = sfs.decoder(selfies_output[i])
                            except:
                                test_invalid_selfies += 1
                                continue
                            if is_valid_smiles(smiles_output) and smiles_output != "":
                                if batch_idx == 0:
                                    print(input['description'][i], input['smiles'][i], smiles_output)
                                similarity = calculate_similarity(smiles_output, input['smiles'][i])
                                if similarity is None:
                                    test_invalid_selfies += 1
                                    continue
                                test_similarities.append(similarity)
                            else:
                                test_invalid_selfies += 1    
                        
                test_similarity = torch.tensor(np.mean(test_similarities), dtype=torch.float32).to(device)
                test_invalid_selfies = torch.tensor(test_invalid_selfies).to(device)

                if not mac_run:
                    dist.all_reduce(test_similarity, op=dist.ReduceOp.SUM)
                    dist.all_reduce(test_invalid_selfies, op=dist.ReduceOp.SUM)
                
                if args.use_wandb:
                    wandb.log({"test_similarity": test_similarity / args.world_size,
                            "test invalid selfies number": test_invalid_selfies / args.world_size,})
                if test_similarity / args.world_size > best_similarity:
                    adapter_dict = {'attn': model.chemical_adapter_attn.state_dict(), 
                                    'ffn': model.chemical_adapter_ffn.state_dict(),
                                    'step': optimizer.last_epoch}
                    if args.freeze_encoder != True:
                        adapter_dict['encoder'] = model.encoder.state_dict(),
                    torch.save(adapter_dict, 'model_parameters_' + args.text_encoder + "_" + args.molecule_decoder + "_" + args.dataset_name + '.pth')
                    best_similarity = test_similarity / args.world_size 


        if mac_run or dist.get_rank() == 0:
            print("For epoch ", epoch_num)
            print("training loss: ", (train_loss / args.world_size).detach().cpu().numpy())
            print("training similarity: ", (train_similarity / args.world_size).detach().cpu().numpy())
            print("training invalid selfies number: ",
                  (train_invalid_selfies / args.world_size).detach().cpu().numpy())
            if train_loss / args.world_size < best_loss:
                adapter_dict = {'attn': model.chemical_adapter_attn.state_dict(), 
                                'ffn': model.chemical_adapter_ffn.state_dict(),
                                'step': optimizer.last_epoch}
                if args.freeze_encoder != True:
                    adapter_dict['encoder'] = model.encoder.state_dict(),
                torch.save(adapter_dict, 'model_parameters_' + args.text_encoder + "_" + args.molecule_decoder + "_" + args.dataset_name + '.pth')
                best_loss = train_loss / args.world_size

            if args.use_wandb:
                wandb.log({"training loss": train_loss / args.world_size,
                           "train_similarity": train_similarity / args.world_size,
                           "training invalid selfies number": train_invalid_selfies / args.world_size,})

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="ChEBI-20")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--warm_up_steps', type=int, default=4000)
    parser.add_argument("--lr_factor", type=float, default=1.0)
    parser.add_argument("--text_encoder", type=str, default="ChemT5")
    parser.add_argument("--molecule_decoder", type=str, default="MolGen")
    parser.add_argument("--freeze_encoder", type=bool, default=True)
    parser.add_argument("--use_wandb", type=bool, default=False)
    parser.add_argument("--wandb_token", type=str)
    parser.add_argument("--skip_valid", type=bool, default=True)
    parser.add_argument("--eval_epoch", type=int, default=5)
    args = parser.parse_args()
    if torch.backends.mps.is_available():
        args.world_size = 1
        main(0, args)
    else:
        args.world_size = torch.cuda.device_count()
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size)

