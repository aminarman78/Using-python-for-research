# # -*- coding: utf-8 -*-
# """
# Created on Wed Aug  5 18:44:46 2020

# @author: Amin
# """



#####applicatoins of python in data science
##translation of dna
# def read_sequence(input_file):
#     """Reads and returns the input sequence with special characters removed"""
    
#     with open(input_file,'r') as f:
#         seq = f.read()
#         seq = seq.replace('\n','')
#         seq = seq.replace('\r','')
#     return seq

# # f = open(input_file,'r')
# # seq = f.read()
# # seq = seq.replace('\n','')
# # seq = seq.replace('\r','')

# def translate(seq):
#     """translate a string containing a nucleotide sequence into a string 
#     containing the corresponding sequence of amin oacids,Nucleotides are
#     translated in triplets using the table dictionary ;each amino acid is
#     encoded with  a string of length 1"""
    

#     table = {'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
#     'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
#     'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
#     'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
#     'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
#     'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
#     'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
#     'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
#     'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
#     'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
#     'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
#     'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
#     'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
#     'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
#     'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
#     'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',}
    
#     protein = ''
#     if len(seq) % 3 == 0:
#         for i in range(0, len(seq), 3):
#             codon = seq[i: i+3]
#             protein += table[codon]
#     return protein

# prt = read_sequence('protein.txt')
# dna = read_sequence('dna.txt')
# translate_dna = translate(dna[20:935]) #According to NCBI
# checking = (translate_dna == prt)
# print(checking)

# =============================================================================
# 
# =============================================================================
## HomeWork

##1
import string
alphabet = " " + string.ascii_lowercase

##2
keys = [i for i in alphabet[1:]]
values = range(1,27)
positions = dict(zip(keys,values))
positions[' '] = 0

##3
# message = "hi my name is caesar"
# encoded_message = ''
# for c in message:
#   for key, values in positions.items():
#     if values == (positions[c] + 1) % 27:
#       encoded_message += key

def encoding(message, shift):
    encoded_message = ''
    for c in message:
        for key,values in positions.items():
            if values == (positions[c] + shift) % 27:
                encoded_message += key
    return encoded_message


message = "hi my name is caesar"
s = encoding(message, 3)
print(s)

def decoding(encoded_message, shift):
    decoded_mess = ''
    for c in encoded_message:
        for key,values in positions.items():
            if values == (positions[c] - shift)%27:
                decoded_mess += key
    return decoded_mess

print(decoding(s,3))