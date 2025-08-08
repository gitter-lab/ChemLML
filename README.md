# Chemical Language Model Linker
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13925649.svg)](https://zenodo.org/doi/10.5281/zenodo.13925649)

## Running the code
First, install the dependencies from `requirements.txt`, preferably in a virtual environment.
For instance, with conda run
```
conda create -n chemlml python=3.9
conda activate chemlml
pip install -r requirements.txt
```

Then, run `python train_ChEBI-20.py` to train on the ChEBI-20 dataset.
Parameters include:  
- **accumulation_steps**: Accumulation steps for gradient accumulation.  
- **warm_up_steps** and **lr_factor**: Parameters for Noam Optimizer.  
- **text_encoder**: Text encoder for ChemLML, "ChemT5" for encoder from Text + ChemT5, "SciBERT" for SciBERT model and "galactica-" + ["125m", "1.3b", "6.7b"] for different scales of Galactica model.  
- **molecule_decoder**: Molecule decoder for ChemLML, "MolGen" and "MolGen-7B" for default MolGen and MolGen7B model.  
- **freeze_encoder**: To freeze the text encoder or not.
- **use_wandb**: To use wandb or not.  
- **wandb_key**: Your wandb key.  
- **skip_valid** To skip the validation set or not.  
- **eval_epoch**: Evaluate on validation set every **eval_epoch** epochs.

Run `python test.py` to test on different datasets.
The only difference is the **dataset_name** parameters.
Select from "ChEBI-20", "PubChem_filtered", or "PubChem_unfiltered".
PubChem_unfiltered must be downloaded from Zenodo into the PubChem subdirectory first.

To evaluate the result, run `python evaluation/fingerprint_metrics.py` and specify the test file.

The pretrained ChemLML models are available on [Zenodo](https://zenodo.org/doi/10.5281/zenodo.11661517).

## Citation
[Chemical Language Model Linker: blending text and molecules with modular adapters](https://arxiv.org/abs/2410.20182)  
Yifan Deng, Spencer S. Ericksen, Anthony Gitter (2024)  
arXiv:2410.20182 [cs.LG]

## Datasets
See the `ChEBI-20` and `PubChem` dataset subdirectories for details and licenses.
- The ChEBI-20 dataset is from [MolT5](https://doi.org/10.18653/v1/2022.emnlp-main.26)
- PubChem dataset is from [Mol-Instructions](https://openreview.net/forum?id=Tlsdsb6l9n) and [PubChem](https://pubchem.ncbi.nlm.nih.gov/). The unfiltered version is hosted on [Zenodo](https://zenodo.org/doi/10.5281/zenodo.11661517).

## Third-party code and models
The result evaluation code `fingerprint_metrics.py` is from the [MolT5 repository](https://github.com/blender-nlp/MolT5), available under the BSD 3-Clause License Copyright (c) 2023, blender-nlp.

ChemLML uses the following models from Hugging Face:
- MolT5: https://huggingface.co/laituan245/molt5-base-smiles2caption
- Text+Chem T5: https://huggingface.co/GT4SD/multitask-text-and-chemistry-t5-base-standard
- MolGen: https://huggingface.co/zjunlp/MolGen-large
- MolGen-7B: https://huggingface.co/zjunlp/MolGen-7b
- Fine-tuned LLaMA2-7B: https://huggingface.co/zjunlp/llama2-molinst-molecule-7b In order to use LLaMA, please refer to the access request at https://huggingface.co/meta-llama/Llama-2-7b-hf. Then copy and paste the huggingface token to line 37 in `model.py`
- SCIBERT: https://huggingface.co/allenai/scibert_scivocab_uncased
- Galactica series: https://huggingface.co/facebook/galactica-125m
- MolXPT: https://huggingface.co/zequnl/molxpt

See the Hugging Face model cards for licenses, limitations, and citations.
