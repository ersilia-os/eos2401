# import libraries
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import transformers.generation as generation
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint

# Make SAFE’s old import path work
generation.DisjunctiveConstraint = DisjunctiveConstraint
generation.PhrasalConstraint = PhrasalConstraint

import safe as sf
from rdkit import Chem
from rdkit.Chem.Scaffolds import rdScaffoldNetwork
from rdkit.Chem import Descriptors
from rdkit import Chem


class MoleculeModel:
    def __init__(self, n_trials=10, n_samples_per_trial=100, lower_molecular_weight=60, upper_molecular_weight=100):
        self.designer = sf.SAFEDesign.load_default(verbose=True)
        self.n_trials = n_trials
        self.n_samples_per_trial = n_samples_per_trial
        self.lower_molecular_weight = lower_molecular_weight
        self.upper_molecular_weight = upper_molecular_weight

    def smiles_to_safe(self, smiles):
        try:
            return sf.encode(smiles)
        except Exception as e:
            print(f" Error in SMILES conversion: {e}")
            return None

    def _extract_core_structure(self, safe):
        # Define scaffold parameter network
        params = rdScaffoldNetwork.ScaffoldNetworkParams()
        # customize parameter attributes
        params.includeScaffoldsWithoutAttachments=False
        if safe is not None:
            mol = Chem.MolFromSmiles(safe)
            net = rdScaffoldNetwork.CreateScaffoldNetwork([mol],params)
            nodemols = [Chem.MolFromSmiles(x) for x in net.nodes]

            filtered_list = []
            for mol in nodemols:
                # Check for the presence of attachment points and molecular weight range
                if "*" in Chem.MolToSmiles(mol) and self.lower_molecular_weight < Descriptors.MolWt(mol) < self.upper_molecular_weight:
                    filtered_list.append(mol)
            
            # If there are no scaffolds within the range, select the closest one
            if not filtered_list:
                closest_mol = min(nodemols, key=lambda x: abs(Descriptors.MolWt(x) - (self.lower_molecular_weight + self.upper_molecular_weight) / 2))
                filtered_list.append(closest_mol)
            
            # Sort the filtered list based on the number of heteroatoms (fewer carbons)
            filtered_list.sort(key=lambda x: x.GetNumHeavyAtoms())

            return filtered_list
        else:
            return None

    def _generate_smiles(self, scaffold):
        generated_smiles = self.designer.scaffold_decoration(
        scaffold=scaffold,
        n_samples_per_trial=self.n_samples_per_trial,
        n_trials=self.n_trials,
        sanitize=True,
        do_not_fragment_further=True,
                )
        return generated_smiles

    def run_model(self, safe):
        generated_smiles = []
        for i in safe:
            row = []
            if i is not None:
                core_structures = self._extract_core_structure(i)
                modified_structures = [Chem.MolToSmiles(core).replace('*', '[*]') for core in core_structures]
                for core in modified_structures:
                    output = self._generate_smiles(core)
                    row += output
                generated_smiles += [row]
            else:
                generated_smiles += [row]
        return generated_smiles
