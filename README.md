When using this code, please cite:
Van Hilten, N.; Verwei, N.; Methorst, J.; Nase, C.; Bernatavicius, A.; Risselada, H.J., Bioinformatics, 2024, 40(2) DOI: 10.1093/bioinformatics/btae069 

This fork is modified to do bulk processing of peptide or protein sequences and heavily minimise outputs.

The original repo contains:
- data/training_set.txt			sequence and MD-calculated ddF for 53,940 sequences in the training set
- data/validation_set.txt		sequence, predicted ddF, MD-calculated ddF, and squared error (SE) for 494 sequences in the validation set
- data/test_set.txt			sequence, predicted ddF, MD-calculated ddF, and squared error (SE) for 493 sequences in the test set
- data/full_range_test_set.txt		sequence, predicted ddF, MD-calculated ddF, and squared error (SE) for 96 sequences in the full range test set

- model/hyperparamter_optimization.py	code used for hyperparameter optimization
- model/transformer.py			code used for final training

- scripts/final_model/			final model and tokenizer (you should download and untar variables.tar.gz)
- scripts/PMIpred_peptide.py		offline version of the peptide module in PMIpred
- scripts/PMIpred_protein.py		offline version of the protein module in PMIpred

- benchmark/				DREAMM, PPM, MODA, and PMIpred data for 27 benchmark proteins (+3 TM proteins)
