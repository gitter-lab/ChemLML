import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

from distutils.ccompiler import new_compiler
import pandas as pd
from model import Text2Mol
from dataset import MoleculeDataset, MoleculeGraphDataset
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import torch
from transformers import BertTokenizer, AutoTokenizer
from utils import calculate_similarity, is_valid_smiles, set_seed
from NoamOpt import NoamOpt
import json

import selfies as sfs
from rdkit import Chem, rdBase
import pickle
import random
import wandb

# Disable error messages from rdkit
import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
rdBase.DisableLog('rdApp.error')

def main(args):
    set_seed(42)
    if torch.backends.mps.is_available():
        mac_run = True
    else:
        mac_run = False

    run_name = "text2mol_" + args.text_encoder + "_" + args.molecule_decoder+ "_" + args.dataset_name + "_test"

    if args.use_wandb:
        print("Using wandb for logging.")
        try:
            wandb.login(key=args.wandb_key)
        except:
            pass
        run = wandb.init(project=run_name, config=args)
        # run = wandb.init(project=run_name, config={"epochs": 100, "learning_rate": 0.005, "batch_size": 8}, name="Trained on CheBI-20 ChemT5 MolGen7B " + dataset_name)
    else:
        print("Not using wandb for logging.")
    corrected_smiles = []
    corrected_selfies = []
    corrected_corpus = []
    if args.dataset_name == "ChEBI-20":
        import csv
        with open("ChEBI-20/test.txt") as f:
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
    
        test_data = list(zip(corrected_corpus, corrected_selfies, corrected_smiles))
    
    elif args.dataset_name == "PubChem_filtered":
        with open("PubChem/PubChem_filtered.json", "r") as f:
            dataset = json.load(f)
        for i in range(len(dataset)):
            try:
                smiles = Chem.MolToSmiles(Chem.MolFromSmiles(dataset[i]['smiles']))
                selfies = sfs.encoder(smiles)
            except Exception as error:
                print(error)
                continue
            corrected_smiles.append(smiles)
            corrected_selfies.append(selfies)
            corrected_corpus.append(dataset[i]['input'])
        data = list(zip(corrected_corpus, corrected_selfies, corrected_smiles))
        test_data = random.sample(data, 3000)
    elif args.dataset_name == "PubChem_unfiltered":
        with open("PubChem/PubChem_unfiltered.pkl", "rb") as f:
            dataset = pickle.load(f)
        corrected_corpus = [dataset[i]['description'] for i in range(len(dataset))]
        corrected_selfies = [dataset[i]['SELFIES'] for i in range(len(dataset))]
        corrected_smiles = [dataset[i]['SMILES'] for i in range(len(dataset))]

        data = list(zip(corrected_corpus, corrected_selfies, corrected_smiles))
        random.shuffle(data)
        test_data = data[int(len(data) * 0.99):]
    

    molecule_tokenizer = AutoTokenizer.from_pretrained("zjunlp/MolGen-Large")

    cls_idx = molecule_tokenizer.cls_token_id
    eos_idx = molecule_tokenizer.eos_token_id
    mask_idx = molecule_tokenizer.mask_token_id
    pad_idx = molecule_tokenizer.pad_token_id

    model = Text2Mol(args.text_encoder, args.molecule_decoder)
    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Untrainable parameters 
    untrainable_params = total_params - trainable_params

    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Untrainable Parameters: {untrainable_params}")

    device = torch.device("mps") if not torch.cuda.is_available() else torch.device("cuda:0")
    
    model_weight = torch.load('model_parameters_' + args.text_encoder + "_" + args.molecule_decoder + "_" + args.dataset_name + '.pth', map_location="cpu")
    if args.freeze_encoder != True:
        model.encoder.load_state_dict(model_weight['encoder'], strict=False)
    model.structural_adapter_attn.load_state_dict(model_weight['attn'], strict=False)
    model.structural_adapter_ffn.load_state_dict(model_weight['ffn'], strict=False)
    
    model.to(device)
    batch_size = args.batch_size
    test_dataset = MoleculeDataset(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    descriptions = []
    ground_truth = []
    output_smiles = []
    
    model.eval()
    with torch.no_grad():
        test_invalid_selfies = 0
        test_similarities = []
        for batch_idx, batch in enumerate(test_dataloader):
            input = batch
            selfies = input['selfies']
            input['description']  = ["Write in SMILES the described molecule: " + i for i in input['description']]
            input['description'] = [i.replace(" with data available.", ".") for i in input['description']]
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
            descriptions.extend(input['description'])
            for i in range(output.shape[0]):
                ground_truth.append(input['smiles'][i])
                try:
                    smiles_output = sfs.decoder(selfies_output[i])
                except:
                    output_smiles.append("None")
                    test_invalid_selfies += 1
                    continue
                if is_valid_smiles(smiles_output) and smiles_output != "":
                    output_smiles.append(smiles_output)
                    similarity = calculate_similarity(smiles_output, input['smiles'][i])
                    if similarity is None:
                        test_invalid_selfies += 1
                        continue
                    test_similarities.append(similarity)
                else:
                    output_smiles.append("None")
                    test_invalid_selfies += 1
            
            if args.use_wandb:
                wandb.log({"test_similarity": np.mean(test_similarities),
                        "training invalid selfies number": test_invalid_selfies,
                        "Validity": len(test_similarities)/(len(test_similarities)+test_invalid_selfies)})
    with open(args.text_encoder + "_" + args.molecule_decoder +"_" + args.dataset_name + ".txt", 'w') as f:
        f.write('description' + '\t' + 'ground truth' + '\t' + 'output' + '\n')
        for desc, rt, ot in zip(descriptions, ground_truth, output_smiles):
            desc = desc.replace("\n", "")
            f.write(desc + '\t' + rt + '\t' + ot + '\n')

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="CheBI-20")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--text_encoder", type=str, default="ChemT5")
    parser.add_argument("--molecule_decoder", type=str, default="MolGen")
    parser.add_argument("--freeze_encoder", type=bool, default=True)
    parser.add_argument("--use_wandb", type=bool, default=False)
    parser.add_argument("--wandb_key", type=str)
    args = parser.parse_args()
    main(args)
