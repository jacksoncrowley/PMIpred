#######
# PMIpred was developed by Niek van Hilten, Nino Verwei, Jeroen Methorst, and Andrius Bernatavicius at Leiden University, The Netherlands (29 March 2023)
#
# This script is an offline version of the peptide module in PMIpred at https://pmipred.fkt.physik.tu-dortmund.de/curvature-sensing-peptide/
#
# When using this code, please cite:
# Van Hilten, N.; Verwei, N.; Methorst, J.; Nase, C.; Bernatavicius, A.; Risselada, H.J., Bioinformatics, 2024, 40(2) DOI: 0.1093/bioinformatics/btae069 
#######

import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import sys
import general_functions as gf
import numpy as np
import math
from tqdm import tqdm

EVOMD_HEADER = 82 # number of lines to skip in Evo-MD output

flags = argparse.ArgumentParser()
flags.add_argument("-s", "--seq", help="Input amino acid sequence (one-letter abbreviations).", nargs="+")
flags.add_argument("-evo", "--evomd", help="EvoMD output file", nargs="+")
flags.add_argument("-o", "--output", help="Output file path.", type=str, default="output.csv")

args = flags.parse_args()

def check_sequence(seq):
    seq = seq.strip()
    seq = "".join(seq.split())
    seq = seq.upper()

    ALPHABET = sorted(['A', 'R', 'N', 'D', 'C', 'F', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'P', 'S', 'T', 'W', 'Y', 'V'])
    if len(seq) > 24:
        return False, "Sequence too long (should be <=24 residues)."
    elif len(seq) < 7:
        return False, "Sequence too short (should be >=7 residues)."

    for AA in seq:
        if AA not in ALPHABET:
            return False, "Character \"" + AA + "\" not allowed. Please use single-letter abbreviations for natural amino acids only."

    return True, seq

def parse_evomd(evomd):
    assert os.path.isfile(evomd)

    sequences = []
    with open(evomd, "r") as f:
        lines = f.readlines()

    for line in lines[EVOMD_HEADER:]:
        sequences.append(line.strip().split()[0])

    return sequences

# logical XOR for args.seq or args.evomd but not both
assert (bool(args.seq) ^ bool(args.evomd))

if args.seq:
    sequences = args.seq
elif args.evomd:
    sequences = []
    for evomd in args.evomd:
        sequences.extend(parse_evomd(evomd))

# get script path
path = os.path.abspath(os.path.dirname(__file__))

 
# load transformer model
model, tokenizer = gf.load_model(os.path.join(path, "final_model"), os.path.join(path, "final_model/tokenizer.pickle"))
    

output = ""

for seq in tqdm(sequences):
    # check sequence
    status, message = check_sequence(seq)
    if not status:
        print("ERROR: " + message)
        sys.exit()
   
    # predict ddF
    ddF = gf.predict_ddF(model, tokenizer, seq)
    
    # printing output
    output += f"{seq}, {ddF}\n"

with open(args.output, "w") as f:
    f.write(output)
