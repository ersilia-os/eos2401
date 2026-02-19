import mol_gen
import sys
import csv

input_file = sys.argv[1]
output_file = sys.argv[2]

designer = mol_gen.MoleculeModel()

with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader) 
    smiles_list = [r[0] for r in reader]

safe_list = [designer.smiles_to_safe(smi) for smi in smiles_list]

outputs = designer.run_model(safe_list)
print(f'output length:{len(outputs)}')
print(outputs)
input_len = len(smiles_list)
output_len = len(outputs)
assert input_len == output_len


N_COLS = 100
HEADER = ["smiles_{0}".format(str(x).zfill(3)) for x in range(N_COLS)]

with open(output_file, "w", newline="") as fp:
    csv_writer = csv.writer(fp)
    csv_writer.writerows([HEADER])
    for o in outputs:
        if len(o) == 0:
            o = [None]*100
        csv_writer.writerow(o)