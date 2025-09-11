# pLM entropy

Code and data for: [Lytras, S., Strange, A., Ito, J., and Sato K. (2025).
Inferring context-specific site variation with evotuned protein language models. *BioRxiv*](https://www.biorxiv.org/content/10.1101/2025.02.19.639211v2)


**pLM entropy** is protein language model (pLM)-based metric to assess protein site conservation and variability.
We test this metric using versions of two popular pLMs (ESM-2 and protT5) fine-tuned on the diversity of different
Influenza A virus serotype hemagglutinin (HA) proteins.

<br>


## The `plm_entropy` Python package


### Installation

The python package for embedding sequences though ESM-2 and protT5 based models and calculating pLM entropy,
along with a number of helper functions is available to install through pip.
We further provide a [conda environment](plm_entropy_env.yml) `yml` file
for easily installing all compatibilities.

To install the conda environment with the python package:


```
git clone

cd plm_entropy

conda env create -f pLM_entropy.yml

pip install update plm_entropy

```

Alternatively, the `plm_entropy` package can also be installed via pip from [PyPI](https://pypi.org/project/plm_entropy/).

<br>

### Usage example

Find a detailed usage example in the following [Jupyter notebook](IAV_HA_pLMs/inference/pLMentropy_inference_example.ipynb).


<br>


### Functions


#### `embed_entropy_esm2(modnam, inputf, outf)`

*options:*

- `modnam`: path to the pLM weights (ESM-2 or fine-tuned version only)
- `inputf`:
	- `.csv` file with at least one `node` column containing sequence identifiers and one `seq` column containing the amino acid sequences, or
	- `fasta` file with amino acid sequences (accepted extensions: `.fa`, `.fas`, `.fasta`)
- `outf`: output files name (without file extension)
- *(optional)* `save_pickle`: (default = `False`)
- *(optional)* `save_logit`: (default = `False`)
- *(optional)* `torch_device`: (default = `cuda:0`)



<br>

#### `embed_entropy_protT5(modnam, inputf, outf)`

*options:*

- `modnam`: path to the pLM weights (protT5MLM or fine-tuned version only)
- `inputf`:
	- `.csv` file with at least one `node` column containing sequence identifiers and one `seq` column containing the amino acid sequences, or
	- `fasta` file with amino acid sequences (accepted extensions: `.fa`, `.fas`, `.fasta`)
- `outf`: output files name (without file extension)
- *(optional)* `save_pickle`: (default = `False`)
- *(optional)* `save_logit`: (default = `False`)
- *(optional)* `torch_device`: (default = `cuda:0`)

<br>

#### `aln_plm_entropy(entrdf_f, asralf)`

Transforms the csv file with pLM entropy values outputted from the `embed_entropy_esm2()` or `embed_entropy_protT5()` functions
so that values correspond to the alignment positions of the aligned input amino acid sequences.

*options:*

- `entrdf_f`: `-site_entropy.csv` output file containing per site pLM entropy values, calculated with the `embed_entropy_esm2()` or `embed_entropy_protT5()` functions
- `asralf`: protein sequence alignment of the sequences for which pLM entropy values were calculated in `fasta` format

<br>


#### `calc_aln_entropy(alfile)`

*options:*

- `alfile`: 
- *(optional)* `exclude_internal_nodes`:

<br>


#### `al_mod_entr_correl(alentropy, modentr_f)`

*options:*

- `alfile`: 
- `modentr_f`:
- *(optional)* `exclude_internal_nodes`:

<br>



## pLM entropy applied to Influenza A virus HA evotuned pLMs

See [Lytras, S., Strange, A., Ito, J., and Sato K. (2025).
Inferring context-specific site variation with evotuned protein language models. *BioRxiv*](https://www.biorxiv.org/content/10.1101/2025.02.19.639211v2) for details.


### Evotuned pLMs

The weights for the [ESM-2](https://huggingface.co/facebook/esm2_t33_650M_UR50D) and [protT5](https://huggingface.co/Rostlab/prot_t5_xl_uniref50) -based evotuned pLMs presented in the manuscript above can be downloaded from the `zenodo` repository below:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14891551.svg)](https://doi.org/10.5281/zenodo.14891551)


<br>

### Data & code available in this repository

- [Evotuning](IAV_HA_pLMs/evotuning)
	
	- IAV HA [sequence datasets](IAV_HA_pLMs/evotuning/training_data) for evotuning
	
	- Jupyter [notebook](IAV_HA_pLMs/evotuning/evotune_ESM2.ipynb) for evotuning ESM-2
	
	- Python [script](IAV_HA_pLMs/evotuning/evotune_T5EncoderMLM.py) for evotuning protT5 (and [adding the MLM head](IAV_HA_pLMs/evotuning/T5EncoderMLM_Freezing_UniRef_constant_20241106.py))
	
- [Inference](IAV_HA_pLMs/inference) 

	- Jupyter [notebook](IAV_HA_pLMs/inference/pLMentropy_inference_example.ipynb) for inferring pLM entropy values using the evotuned models with the `plm_entropy` Python package.
	
	- Example data for the inference.
	
- [Testing data](IAV_HA_pLMs/testing_data) 

	- IAV HA sequences and phylogenies used in the manuscript.
	- *Note that protein sequences retrieved from the GISAID database have been omitted from the repository.*


- [Additional tests](IAV_HA_pLMs/additional)

	- Jupyter [notebook](IAV_HA_pLMs/additional/tree_prediction.ipynb) for predicting which protein sites will mutate in a given sequence context using pLM entropy.
	
	- [Code](IAV_HA_pLMs/additional/reset_weights) for testing influence of pretrained ESM-2 by resetting weights before evotuning
	
	- [Data and code](IAV_HA_pLMs/additional/dnds_comparison) for H3N2 tree backbone aBSREL branch-specific selection analysis
