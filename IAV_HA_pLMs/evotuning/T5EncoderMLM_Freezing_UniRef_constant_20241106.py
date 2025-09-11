# %%
import math
from typing import Optional, Tuple, Union

import torch
from Bio import SeqIO
from datasets import Dataset
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (
    # T5ForConditionalGeneration,
    DataCollatorForLanguageModeling,
    T5EncoderModel,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import MaskedLMOutput

# %%
# Parameters
version = "T5EncoderMLM_UniRef_2e_constant_20241118_test"
model_type = "Rostlab/prot_t5_xl_uniref50"

fasta_file = "uniref50_from_ESM_full.fasta"


# %%
class T5LMHead(nn.Module):
    """Head for masked language modeling. Linear -> Gelu -> Norm -> Linear + Bias
    Outputs logits the size of the vocabulary (128)
    Adapted from ESMForMaskedLM"""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.decoder = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

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


# %%
class T5EncoderMLM(T5EncoderModel):
    def __init__(self, config):
        super().__init__(config)
        self.custom_lm_head = T5LMHead(config)
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
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=lm_logits,
            hidden_states=encoder_outputs.hidden_states,
        )


# %%
model = T5EncoderMLM.from_pretrained(model_type)
tokenizer = T5Tokenizer.from_pretrained(model_type, do_lower_case=False)

device = torch.device("cuda")

# %%
# Freeze the model parameters that aren't part of the MLM head
for name, param in model.named_parameters():
    if "custom_lm_head" not in name:
        param.requires_grad = False

for name, param in model.named_parameters():
    print(f"{name}: {param.requires_grad}")

model = model.to(device)

# %%
# Pre-process Input
seqs = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))

all_seqs = [
    " ".join(list(str(seqs[x].seq))) for x in list(seqs) if len(seqs[x].seq) < 1001
]

train_seqs, rem_seqs = train_test_split(all_seqs, test_size=0.3, random_state=42)
train_tokenized = tokenizer(train_seqs, max_length=1000, padding=True, truncation=True)
train_set = Dataset.from_dict(train_tokenized)

eval_seqs, test_seqs = train_test_split(rem_seqs, test_size=0.3333, random_state=42)
eval_tokenized = tokenizer(eval_seqs, max_length=1000, padding=True, truncation=True)
eval_set = Dataset.from_dict(eval_tokenized)

test_tokenized = tokenizer(test_seqs, max_length=1000, padding=True, truncation=True)
test_set = Dataset.from_dict(test_tokenized)

# %%
# Set mask token name to "<extra_id_0>" for use with data collator
tokenizer.mask_token = "<extra_id_0>"

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, return_tensors="pt", mlm_probability=0.15
)  # provide random masking and return tensors during training per-batch

# %%
# Set training arguments and initialize the trainer
training_args = TrainingArguments(
    output_dir=f"{version}/trainer",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    seed=13,
    tf32=True,
    dataloader_num_workers=12,
    disable_tqdm=False,
    remove_unused_columns=False,
    lr_scheduler_type="constant",
    learning_rate=0.00005,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=eval_set,
    data_collator=data_collator,
)

# %%
trainer.train()

# %%
trainer.save_model(f"{version}/model")

# %%
# Evaluate model performance via perplexity score
evaluator = Trainer(
    model=model,
    data_collator=data_collator,
    eval_dataset=test_set,
)
results = evaluator.evaluate()

print(
    f"Adapted Model's Perplexity: {math.exp(results['eval_loss']):.4f}"
)  # Result: 8.1700
