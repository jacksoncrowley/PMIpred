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

import time

flags = argparse.ArgumentParser()
flags.add_argument("-s", "--seq", help="Input amino acid sequence (one-letter abbreviations).", nargs="+", required=True)
flags.add_argument("-c", "--charge", help="Negative target membrane; apply charge correction.", action="store_true")
flags.add_argument("-o", "--output", help="Output directory.", type=str)
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

def calc_descriptors(seq):
    hydro_dict = {'I': [1.80], 'F': [1.79], 'V': [1.22], 'L': [1.70], 'W': [2.25], 'M': [1.23], 'A': [0.31],
                'G': [0.00], 'C': [1.54], 'Y': [0.96], 'P': [0.72], 'T': [0.26], 'S': [-0.04], 'H': [0.13],
                'E': [-0.64], 'N': [-0.60], 'Q': [-0.22], 'D': [-0.77], 'K': [-0.99], 'R': [-1.01]} # Fauchere & Pliska 1983

    z = seq.count("R")+seq.count("K")-seq.count("D")-seq.count("E")
    
    H = np.mean([hydro_dict[AA][0] for AA in seq])
    
    sum_cos, sum_sin = 0.0, 0.0
    angle = 100.0
    for i, AA in enumerate(seq):
        h = hydro_dict[AA][0]
        rad_inc = ((i*angle)*math.pi)/180.0
        sum_cos += h * math.cos(rad_inc)
        sum_sin += h * math.sin(rad_inc)
    uH = math.sqrt(sum_cos**2 + sum_sin**2) / len(seq)

    return z, H, uH

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

# get script path
path = os.path.abspath(os.path.dirname(__file__))

for seq in args.seq:
    start = time.time()
    # check sequence
    print(seq)
    print(len(seq))
    print(seq[0])
    status, message = check_sequence(seq)
    if not status:
        print("ERROR: " + message)
        sys.exit()
    
    # calculate physicochemical descriptors
    z, H, uH = calc_descriptors(seq)
    
    # load transformer model
    model, tokenizer = gf.load_model(os.path.join(path, "final_model"), os.path.join(path, "final_model/tokenizer.pickle"))
    
    # predict ddF
    ddF = gf.predict_ddF(model, tokenizer, seq)
    
    # corrections
    ddF_L24 = gf.length_correction(seq, ddF)
    if args.charge:
        ddF_adj = gf.charge_correction(seq, ddF_L24)
        dF_sm = calc_dF_sm(ddF_adj)
    else:
        dF_sm = calc_dF_sm(ddF_L24)
    
    # printing output
    output = ""
    output += seq + "\n\n"
    output += "ΔΔF =\t\t\t" + str(round(ddF, 3)) + " kJ/mol\n"
    output += "ΔΔF_L24 =\t\t" + str(round(ddF_L24, 3)) + " kJ/mol\n"
    if args.charge:
        output += "ΔΔF_adj =\t\t" + str(round(ddF_adj, 3)) + " kJ/mol\n"
    output += "----------\n"
    if args.charge:
        output += "Negatively charged membrane\nCalculated from ΔΔF_adj:\nΔF_sm(R=50) =\t\t" + str(round(dF_sm, 3)) + " kJ/mol\n"
    else:
        output += "Neutral membrane\nCalculated from ΔΔF_L24:\nΔF_sm(R=50) =\t\t" + str(round(dF_sm, 3)) + " kJ/mol\n"
    output += "----------\n"
    output += "Length =\t\t\t" + str(len(seq)) + "\n"
    output += "Charge =\t\t\t" + str(z) + "\n"
    output += "Hydrophobicity =\t\t" + str(round(H, 3)) + "\n"
    output += "Hydrophobic moment =\t" + str(round(uH, 3)) + "\n"
    print(output)
    end = time.time()
    print(f"Time elapsed: {end-start}")
