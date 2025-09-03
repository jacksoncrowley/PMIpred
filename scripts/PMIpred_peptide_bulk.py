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

def calc_dF_sm(ddF): # Calculate dF_sm_R50: the membrane-binding free energy to a typical liposome (R=50)
    a = 3.83 # calibration ddF_ vs dF_sm_infty
    b = 12.27 # calibration ddF vs dF_sm_infty
    e = 0.165 # relative strain used in ddF calculation
    R = 50 # typical liposome radius
    dF_sm_R50 = a*ddF+b + (ddF/e)*( (1/R**2) + (2/R) )
    return dF_sm_R50

def calc_Pm(ddF, R):
    N_A = 6.022E23 # avogadro constant in mol-1
    V = 1E24 # volume in nm3 (1 liter)
    A_lip = 0.64 # area per lipid in nm2
    conc = 0.001 # [0.0001, 0.005] # concentrations in M
    Vp = 5*1*1 # peptide volume in nm3
    Ap = 5*1 # peptide area in nm2
    kT = 2.479 # kJ/mol
    A = 1/2 * conc*N_A*A_lip
    Ns = V/Vp
    Nm = A/Ap

    a = 3.83
    b = 12.27
    e = 0.165

    Pm = 1/(1+(Ns/Nm)*np.exp((a*ddF+b+(ddF/e)*( (1/R**2) + (2/R) ))/kT))
    return Pm

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

    dF_sm_R50 = calc_dF_sm(ddF)
    Pm_R50 = calc_Pm(ddF,50)
    
    # printing output
    output += f"{seq}, {ddF}, {dF_sm_R50}, {Pm_R50}\n"

with open(args.output, "w") as f:
    f.write(output)
