import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel

from esm.modules import FeedForwardNetwork, NormalizedResidualBlock
from esm.multihead_attention import MultiheadAttention

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_padding_mask(seq, padding_token=0):
    # Create mask with 0s at padding tokens and 1s elsewhere
    mask = (seq == padding_token).transpose(0, 1)
    return mask

class Text2Mol(nn.Module):
    def __init__(self, TextModel, freezeEncoder, MoleculeModel) -> None:
        super().__init__()
        if "scibert" in TextModel:
            self.encoder = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
            self.text_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        if TextModel == "ChemT5":
            self.encoder = AutoModelForSeq2SeqLM.from_pretrained("GT4SD/multitask-text-and-chemistry-t5-base-standard").encoder
            self.text_tokenizer = AutoTokenizer.from_pretrained("GT4SD/multitask-text-and-chemistry-t5-base-standard")
        if "galactica" in TextModel:
            TextModel = "facebook/" + TextModel
            self.text_tokenizer = AutoTokenizer.from_pretrained(TextModel)
            self.text_tokenizer.pad_token= "<pad>"
            self.encoder = AutoModelForCausalLM.from_pretrained(TextModel)
        if TextModel == "Mol-Instruction":
            model =  LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", load_in_8bit=True, torch_dtype=torch.float16, device_map={"": 0}, 
            token="") # Your huggingface token
            self.encoder = PeftModel.from_pretrained(
                model,
                "adapter/",
                torch_dtype=torch.float16,
                device_map={"": 0},
            )
            self.text_tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token="")
            self.text_tokenizer.pad_token= "<pad>"
        
        if freezeEncoder:
            self.encoder.requires_grad = False
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False
            self.encoder.eval()
        else:
            self.encoder.train()
            self.encoder.requires_grad = True
            for name, param in self.encoder.named_parameters():
                param.requires_grad = True
        
        if MoleculeModel == "MolGen":
            model = AutoModelForSeq2SeqLM.from_pretrained("zjunlp/MolGen-Large")
            self.MolGen = model.model.decoder
            self.MolGen.lm_head = model.lm_head
        elif MoleculeModel == "MolGen-7B":
            self.MolGen = AutoModelForCausalLM.from_pretrained("zjunlp/MolGen-7b")
        else:
            self.MolGen = AutoModelForCausalLM.from_pretrained(MoleculeModel)

        self.chemical_adapter_attn = NormalizedResidualBlock(
            layer=MultiheadAttention(
                self.MolGen.config.hidden_size,
                num_heads=16,
                kdim=self.encoder.config.hidden_size,
                vdim=self.encoder.config.hidden_size,
                add_bias_kv=True,
                add_zero_attn=False,
                use_rotary_embeddings=False,
            ),
            embedding_dim=self.MolGen.config.hidden_size,
            dropout=0.1
        )
        self.chemical_adapter_ffn = NormalizedResidualBlock(
            layer=FeedForwardNetwork(
                self.MolGen.config.hidden_size,
                self.MolGen.config.hidden_size // 2,
                activation_dropout=0.1
            ),
            embedding_dim=self.MolGen.config.hidden_size,
            dropout=0.1
        )
        for name, param in self.MolGen.named_parameters():
            param.requires_grad = False
        self.MolGen.eval()

    
    def forward(self, batch):
        text = batch['description']
        selfies_token = batch['selfies']
        selfies_token = batch['prev_tokens']
        text_tokens = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.encoder.device)
        text_output = self.encoder(text_tokens.input_ids, output_hidden_states=True)
        text_embedding = text_output.hidden_states[-1].squeeze().transpose(0, 1)
        smiles_embedding = self.MolGen(selfies_token, output_hidden_states=True).hidden_states[-1].transpose(0, 1)
        dec_output = self.chemical_adapter_attn(
            smiles_embedding,
            key=text_embedding,
            value=text_embedding,
            key_padding_mask=text_tokens['attention_mask'].float(),
            need_weights=False
        )[0]
        dec_output = self.chemical_adapter_ffn(dec_output).transpose(0, 1)
        decode_smiles_logits = self.MolGen.lm_head(dec_output)
        return decode_smiles_logits
    
    def sample_ar(self, batch, temp, cls_idx, greedy):
        text = batch['description']
        text_tokens = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.encoder.device)
        text_output = self.encoder(text_tokens.input_ids, output_hidden_states=True)
        text_embedding = text_output.hidden_states[-1].squeeze().transpose(0, 1)
        smiles_token = cls_idx * torch.ones((len(text), 1)).int().to(text_embedding.device)
        smiles_embedding = self.MolGen(smiles_token, output_hidden_states=True).hidden_states[-1].transpose(0, 1) # , use_cache=True
        smiles_sequence = cls_idx * torch.ones(len(text), 1).int().to(text_embedding.device)
        all_logits = []
        for step in range(350):
            dec_output = self.chemical_adapter_attn(
                smiles_embedding,
                key=text_embedding,
                value=text_embedding,
                key_padding_mask=text_tokens['attention_mask'].float(),
                need_weights=False
            )[0]
            dec_output = self.chemical_adapter_ffn(dec_output).transpose(0, 1)
            logits = self.MolGen.lm_head(dec_output[:, -1, :])
            logits = logits / temp
            probs = F.softmax(logits, dim=-1)
            if greedy == True:
                next_token = torch.argmax(F.softmax(logits, dim=-1), dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
            all_logits.append(logits)
            smiles_sequence = torch.cat([smiles_sequence, next_token], dim=1)
            smiles_embedding = self.MolGen(smiles_sequence, output_hidden_states=True).hidden_states[-1].transpose(0, 1)
        return torch.stack(all_logits).transpose(0, 1), smiles_sequence[:, 1:]
    
    def sample_nar(self, batch):
        text = batch['description']
        text_tokens = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.encoder.device)
        text_output = self.encoder(**text_tokens)
        text_embedding = text_output.last_hidden_state.squeeze().transpose(0, 1)
        selfies_token = batch['prev_tokens']
        smiles_key_padding_mask = create_padding_mask(selfies_token, padding_token=1).transpose(0, 1)
        smiles_embedding = self.MolGen(selfies_token, output_hidden_states=True).hidden_states[0].transpose(0, 1)
        dec_output = self.chemical_adapter_attn(
            smiles_embedding,
            key=text_embedding,
            value=text_embedding,
            key_padding_mask=text_tokens['attention_mask'].float(),
            need_weights=False
        )[0]
        dec_output = self.chemical_adapter_ffn(dec_output).transpose(0, 1)
        decode_smiles_output = self.MolGen.lm_head(dec_output)
        return decode_smiles_output