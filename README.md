# Dataset
- ChEBI-20 dataset is collected by MolT5. *[Translation between Molecules and Natural Language](https://arxiv.org/abs/2204.11817)*
- PubChem dataset is collected by MoleculeSTM. *[Multi-modal Molecule Structure-text Model for Text-based Retrieval and Editing](https://arxiv.org/abs/2212.10789)*
- Mol-Instruction followed the same data collection process. *[Mol-Instructions: A Large-Scale Biomolecular Instruction Dataset for Large Language Models](https://arxiv.org/abs/2306.08018)*


# Code and Model
- The result evaluation code is from MolT5 repo: https://github.com/blender-nlp/MolT5
- We use the following models in huggingface:
  MolT5: https://huggingface.co/laituan245/molt5-base-smiles2caption  
  Text+Chem T5: https://huggingface.co/GT4SD/multitask-text-and-chemistry-t5-base-standard  
  MolGen: https://huggingface.co/zjunlp/MolGen-large  
  MolGen-7B: https://huggingface.co/zjunlp/MolGen-7b  
  Fine-tuned LLaMA2-7B: https://huggingface.co/zjunlp/llama2-molinst-molecule-7b  
  SCIBERT: https://huggingface.co/allenai/scibert_scivocab_uncased  
  Galactica series: https://huggingface.co/facebook/galactica-125m  

# Run the code
Simply run *python train_ChEBI-20.py* to train on ChEBI-20 dataset.  
Parameters include:  
**accumulation_steps**: Accumulation steps for gradient accumulation.  
**warm_up_steps** and **lr_factor**: Parameters for Noam Optimizer.  
**text_encoder**: Text encoder for ChemLML, "ChemT5" for encoder from Text + ChemT5, "SciBERT" for SciBERT model and "galactica-" + ["125m", "1.3b", "6.7b"] for different scales of Galactica model.  
**molecule_decoder**: Molecule decoder for ChemLML, "MolGen" and "MolGen-7B" for MolGen regular and MolGen7B model.  
**freeze_encoder**: To freeze the text encoder or not.
**use_wandb**: To use wandb or not.  
**wandb_key**: Your wandb key.  
**skip_valid** To skip the validation set or not.  
**eval_epoch**: Evaluate on validation set every **eval_epoch** epochs.

For test, simply run *python text.py* to test on different datasets.
The only different is the **dataset_name** parameters, select from "ChEBI-20",  "PubChem_filtered" or "PubChem_unfiltered".

To evaluate the result, run *python evaluation/fingerprint_metrics.py* and specify the text file.

# Citation


