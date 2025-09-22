#### IMPORTS ####

import pandas as pd
from Bio import SeqIO, Seq
import numpy as np
import esm
import math

from transformers import EsmForMaskedLM, T5EncoderModel,T5Tokenizer,DataCollatorForLanguageModeling
from transformers.modeling_outputs import MaskedLMOutput
import time
import pickle
import scipy


import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from typing import List, Optional, Tuple, Union





#### protT5MLM REQUIREMENTS ####

class T5LMHead(nn.Module):
    """Head for masked language modeling. Linear -> Gelu -> Norm -> Linear + Bias
    Outputs logits the size of the vocabulary (128)
    Adapted from ESMForMaskedLM"""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.decoder = nn.Linear(config.d_model, 128, bias=False)
        self.bias = nn.Parameter(torch.zeros(128))

    @staticmethod
    def gelu(x):
        """
        This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results.
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x) + self.bias
        return x


class T5EncoderMLM(T5EncoderModel):
    def __init__(self, config):
        super().__init__(config)
        self.custom_lm_head = T5LMHead(
            config
        ) 
        self.init_weights()
        print(config)

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = (
            self.config.initializer_factor
        )  # Used for testing weights initialization
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_model) ** -0.5)
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, T5LMHead):
            module.dense.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_model) ** -0.5)
            )
            module.dense.bias.data.zero_()
            module.layer_norm.weight.data.fill_(1.0)
            module.layer_norm.bias.data.zero_()
            module.decoder.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_model) ** -0.5)
            )
            module.bias.data.zero_()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.FloatTensor], MaskedLMOutput]:
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.custom_lm_head(encoder_outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(
                lm_logits.device
            )  # ensure logits and labels are on same device
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=lm_logits,
            attentions=encoder_outputs.attentions,
            hidden_states=encoder_outputs.hidden_states,
        )
		
		
		
#### EXTRAS ####

tokenizer = tokenizer = T5Tokenizer.from_pretrained(
    "Rostlab/prot_t5_xl_uniref50", do_lower_case=False
)

# Add masking token to the tokenizer for the datacollator to use:
tokenizer.add_special_tokens({"mask_token": "<mask>"})

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, return_tensors="pt", mlm_probability=0.15
)  # provide random masking and return tensors during training per-batch

model, alphabet = esm.pretrained.load_model_and_alphabet('esm2_t6_8M_UR50D')
batch_converter = alphabet.get_batch_converter()




#### FUNCTIONS ####



def embed_entropy_esm2(modnam, inputf, outf, save_logit = False, save_pickle = False, torch_device="cuda:0" ):
    
    #make df with all probabilities
    allentropies = []

    #store the hidden states
    allstates = {}

    #store full logit dfs 
    alllogitdfs = []

    print('Starting to load model...\n')
    
    #load model to GPU
    mod = EsmForMaskedLM.from_pretrained(modnam)
    device = torch.device(torch_device)
    if torch.cuda.is_available():
        mod =  mod.to(device)
        print("%s transferred model to GPU\n"%modnam)  


    #read input file based on extension
    if inputf.split('.')[-1] in ['fas', 'fasta', 'fa']:
        
        seqdic = SeqIO.to_dict(SeqIO.parse(inputf, 'fasta')) 

        indf = pd.DataFrame([[x, str(seqdic[x].seq)] for x in list(seqdic)])
        indf.columns = ['node', 'seq']

    elif inputf.split('.')[-1] == 'csv':

        indf = pd.read_csv(inputf)

    else:

        print("Wrong input file! Please input a fasta file with extensions 'fas', 'fasta', 'fa', or a csv file with columns 'node' and 'seq'")
    
    print('Read %i protein sequences\n\n'%len(indf))
    
    maxslen = max([len(x) for x in list(indf.seq)])
    
    c=0
    stt = time.time()

    #for each sequence
    for seqid in list(indf.node):
        #time
        c=c+1
        if c%50==0:
            print('%i sequences done in %.2fmins'%(c, (time.time() - stt)/60))

        #prepare sequence for embedding
        seq = str( list(indf[indf.node == seqid].seq)[0] ).replace('J', 'X')
        batch_labels, batch_strs, batch_tokens = batch_converter([('base', seq)])
        batch_tokens = batch_tokens.to(device=device, non_blocking=True)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        #embed
        if torch.cuda.is_available():
            m_results = mod(batch_tokens, output_hidden_states=True)

        #softmax to normalise the logit values for this sequence
        m_logits = torch.nn.LogSoftmax(dim=1)((m_results["logits"][0]).to(device="cpu")).detach()

        #average out position states for each dimension/layer -> one number per model dimension per layer
        m_hstates = [np.mean(m_results['hidden_states'][layernum][0].to(device="cpu").detach().numpy(), axis = 0) for layernum in range(0, len(mod.esm.encoder.layer)+1 )]
        
        #make dataframe with logits
        df = pd.DataFrame(m_logits)
        #add all model tokens as columns
        df.columns = alphabet.all_toks
        df.drop(".",inplace=True,axis=1)
        #make positions column
        df["pos"] = df.index.values
        df = df.melt(id_vars="pos").sort_values(["pos","variable"])
        #get probabilities by exp the softmaxed values
        df["probability"] = np.exp(df.value)
        real_amino_acids = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
        #only keep amino acid tokens
        df = df[df.variable.isin(real_amino_acids)]
        max_probs = [sum(df[df.pos == pos].probability) for pos in df.pos.sort_values().unique()]
        #normalise by maximum prob in each position to get adjusted probability (token probs within one position sum up to 1)
        df["token_adjusted_probability"] = [max_probs[pos] for pos in df.pos]
        df["token_adjusted_probability"] = df["probability"]/df["token_adjusted_probability"]
        #remove positions that match to start/end special tokens
        df = df[(df.pos>=1) & (df.pos<=np.max(df.pos-1))]

        #list of base 2 entropies of token adjusted probs across aa tokens for each position
        embentropies = [scipy.stats.entropy(list(df[(df.pos == i+1)]['token_adjusted_probability']), base=2) for i in range(len(seq))]

        #add padding if the sequence is shorter than the longest of the alignment
        if len(embentropies) < maxslen:
            embentropies = embentropies + ['' for x in range(maxslen - len(embentropies))]

        allentropies.append([seqid] + embentropies)

        allstates.update({seqid:m_hstates})

        df['seqid'] = [seqid for i in range(len(df))]

        alllogitdfs.append(df)

    
    print('\nFinished embedding all sequences!\n\nWriting output...')
    
    #turn entropy lists into df
    entrodf = pd.DataFrame(allentropies)
    entrodf.columns = ['name'] + [i+1 for i in range(maxslen)]

    entrodf.to_csv(outf + '-site_entropy.csv', index=False)
    print('\n\nWrote %s-site_entropy.csv containing per site pLM entropy for each sequence'%outf)

    if save_pickle==True:
        #export hidden states as pickle
        with open(outf + '.pickle', 'wb') as out:
            pickle.dump(allstates, out, pickle.HIGHEST_PROTOCOL)
        print('\nWrote %s.pickle containing all compressed embedding hidden states'%outf)
        
    if save_logit==True:
        #export df with all logit probs
        fnlogitdf = pd.concat(alllogitdfs)
        fnlogitdf.to_csv(outf + '-all_logitprobs.csv', index=False)
        print('\nWrote %s-all_logitprobs.csv containing all model logit probabilities for every amino acid being in every sequence site'%outf)
    
	



def embed_entropy_protT5(modnam, inputf, outf, save_logit = False, save_pickle = False, torch_device="cuda:0"):
    
    #make df with all probabilities
    allentropies = []

    #store the hidden states
    allstates = {}

    #store full logit dfs 
    alllogitdfs = []

    print('Starting to load model...\n')
    
    #load model to GPU
    mod = T5EncoderMLM.from_pretrained(f"{modnam}", ignore_mismatched_sizes=True)
    device = torch.device(torch_device)
    if torch.cuda.is_available():
        mod =  mod.to(device)
        print("%s transferred model to GPU\n"%modnam)  


    #read input file based on extension
    if inputf.split('.')[-1] in ['fas', 'fasta', 'fa']:
        
        seqdic = SeqIO.to_dict(SeqIO.parse(inputf, 'fasta')) 

        indf = pd.DataFrame([[x, str(seqdic[x].seq)] for x in list(seqdic)])
        indf.columns = ['node', 'seq']

    elif inputf.split('.')[-1] == 'csv':

        indf = pd.read_csv(inputf)

    else:

        print("Wrong input file! Please input a fasta file with extensions 'fas', 'fasta', 'fa', or a csv file with columns 'node' and 'seq'")
    
    print('Read %i protein sequences\n\n'%len(indf))
    
    maxslen = max([len(x) for x in list(indf.seq)])
    
    c=0
    stt = time.time()

    #for each sequence
    for seqid in list(indf.node):
        #time
        c=c+1
        if c%50==0:
            print('%i sequences done in %.2fmins'%(c, (time.time() - stt)/60))

        #prepare sequence for embedding
        seq = str( list(indf[indf.node == seqid].seq)[0] ).replace('J', 'X')

        formseq = [" ".join(list(seq))]

        token_encoding = tokenizer(formseq, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(token_encoding['input_ids']).to(device)
        attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

        #embed
        if torch.cuda.is_available():
            m_results = mod(input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True, output_attentions=True)

        #softmax to normalise the logit values for this sequence
        m_logits = torch.nn.LogSoftmax(dim=1)((m_results["logits"][0]).to(device="cpu")).detach()

        #average out position states for each dimension/layer (for 24 layer model) -> one number per model dimension per layer
        m_hstates = [np.mean(m_results['hidden_states'][layernum][0].to(device="cpu").detach().numpy(), axis = 0) for layernum in range(0, len(mod.encoder.block)+1 )]
        
        #make dataframe with logits
        df = pd.DataFrame(m_logits)
        #add all model tokens as columns
        df.columns = [x.replace("▁", "") for x in list(tokenizer.get_vocab())[:-1]]
        
        #make positions column
        df["pos"] = df.index.values
        df = df.melt(id_vars="pos").sort_values(["pos","variable"])
        #get probabilities by exp the softmaxed values
        df["probability"] = np.exp(df.value)
        real_amino_acids = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
        #only keep amino acid tokens
        df = df[df.variable.isin(real_amino_acids)]
        max_probs = [sum(df[df.pos == pos].probability) for pos in df.pos.sort_values().unique()]
        #normalise by maximum prob in each position to get adjusted probability (token probs within one position sum up to 1)
        df["token_adjusted_probability"] = [max_probs[pos] for pos in df.pos]
        df["token_adjusted_probability"] = df["probability"]/df["token_adjusted_probability"]
        #remove positions that match to start/end special tokens
        df = df[df.pos<max(list(df.pos))]

        #list of base 2 entropies of token adjusted probs across aa tokens for each position
        embentropies = [scipy.stats.entropy(list(df[(df.pos == i)]['token_adjusted_probability']), base=2) for i in range(len(seq))]

        #add padding if the sequence is shorter than the longest of the alignment
        if len(embentropies) < maxslen:
            embentropies = embentropies + ['' for x in range(maxslen - len(embentropies))]

        allentropies.append([seqid] + embentropies)

        allstates.update({seqid:m_hstates})

        df['seqid'] = [seqid for i in range(len(df))]

        alllogitdfs.append(df)

    
    print('\nFinished embedding all sequences!\n\nWriting output...')
    
    #turn entropy lists into df
    entrodf = pd.DataFrame(allentropies)
    entrodf.columns = ['name'] + [i+1 for i in range(maxslen)]

    entrodf.to_csv(outf + '-site_entropy.csv', index=False)
    print('\n\nWrote %s-site_entropy.csv containing per site pLM entropy for each sequence'%outf)

    if save_pickle==True:
        #export hidden states as pickle
        with open(outf + '.pickle', 'wb') as out:
            pickle.dump(allstates, out, pickle.HIGHEST_PROTOCOL)
        print('\nWrote %s.pickle containing all compressed embedding hidden states'%outf)
    
    if save_logit==True:
        #export df with all logit probs
        fnlogitdf = pd.concat(alllogitdfs)
        fnlogitdf.to_csv(outf + '-all_logitprobs.csv', index=False)
        print('\nWrote %s-all_logitprobs.csv containing all model logit probabilities for every amino acid being in every sequence site'%outf)






def aln_plm_entropy(entrdf_f, asralf):
    
    if '.csv' in entrdf_f:
        outf = entrdf_f.replace('.csv', '.aln.csv')
    else:
        outf = entrdf_f + '.aln.csv'

    entrdf = pd.read_csv(entrdf_f, index_col=0)
    
    aldic = SeqIO.to_dict(SeqIO.parse(asralf, 'fasta'))
    
    newentrdic = {}

    c = 0
    
    for nod in list(entrdf.index):

        c = c+1
        if c%100==0:
            print('Aligned %i pLM entropy results...'%c)
            
        newentr = []
        alseq = str(aldic[nod].seq)
    
        tabpoz = 1
        for i in range(len(alseq)):
            # if (i%100==0) and c%100==0:
            #     print('\t\t', i)            
            if alseq[i] != '-':
                newentr.append(entrdf.loc[nod,str(tabpoz)])
                tabpoz = tabpoz + 1
            else:
                newentr.append('')
    
        newentrdic.update({nod:newentr})

    alndf = pd.DataFrame(newentrdic)
    alndf.index = alndf.index +1
    alndf = alndf.T

    alndf.to_csv(outf)

    print('\nDone! Wrote %s'%outf)

    return alndf
    



def calc_aln_entropy(alfile, exclude_internal_nodes=False):

    seqdic = SeqIO.to_dict(SeqIO.parse(alfile, 'fasta'))
    
    #remove internal node seqs from per site entropy calculations
    if exclude_internal_nodes==False:
        aldf = pd.DataFrame({x:[a for a in seqdic[x]] for x in list(seqdic)})
    elif exclude_internal_nodes==True:
        aldf = pd.DataFrame({x:[a for a in seqdic[x]] for x in list(seqdic) if 'NODE_' not in x})
    
    alnpersite = {}
    
    alentropy = []
    
    for i in range(len(aldf)):
        reslist = aldf.iloc[i].tolist()
    #     print(reslist)
        if '-' in reslist:
            reslist.remove('-')
        proplist = [reslist.count(x)/len(reslist) for x in set(reslist)]
        #calculate alignment site entropy
        alentropy.append(scipy.stats.entropy(proplist, base=2))
        
        #also store the per-residue proportions for each site
        alnpersite.update({i+1:{x:reslist.count(x) for x in set(reslist)}})
    
    alentropy = alentropy

    return alentropy
    
    


def al_mod_entr_correl(alentropy, modentr_f, exclude_internal_nodes=False):

    model_entr_df = pd.read_csv(modentr_f, index_col=0)

    if exclude_internal_nodes==True:
        #remove internal node seqs from per site entropy calculations
        model_entr_df = model_entr_df[~model_entr_df.index.str.contains('NODE_')]
        
    model_entr_avg = [sum(model_entr_df.get(i).dropna().tolist())/len(model_entr_df) for i in list(model_entr_df.columns)]
    
    return scipy.stats.spearmanr(alentropy, model_entr_avg)