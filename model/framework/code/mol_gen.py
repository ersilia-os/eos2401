import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import transformers.generation as generation
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint

generation.DisjunctiveConstraint = DisjunctiveConstraint
generation.PhrasalConstraint = PhrasalConstraint

import safe as sf
from rdkit import Chem
from rdkit.Chem.Scaffolds import rdScaffoldNetwork
from rdkit.Chem import Descriptors


class MoleculeModel:
    def __init__(self, n_trials=1, n_samples_per_trial=100, lower_molecular_weight=60, upper_molecular_weight=100):
        self.designer = sf.SAFEDesign.load_default(verbose=True)
        self.n_trials = n_trials
        self.n_samples_per_trial = n_samples_per_trial
        self.lower_molecular_weight = lower_molecular_weight
        self.upper_molecular_weight = upper_molecular_weight

    def smiles_to_safe(self, smiles):
        try:
            return sf.encode(smiles)
        except Exception:
            return None

    def _extract_core_structure(self, safe_str):
        try:
            if safe_str is None:
                return None
            mol = Chem.MolFromSmiles(safe_str)
            if mol is None:
                return None
            params = rdScaffoldNetwork.ScaffoldNetworkParams()
            params.includeScaffoldsWithoutAttachments = False
            net = rdScaffoldNetwork.CreateScaffoldNetwork([mol], params)
            nodemols = [Chem.MolFromSmiles(x) for x in net.nodes]
            nodemols = [m for m in nodemols if m is not None]
            if not nodemols:
                return None

            filtered_list = [
                m for m in nodemols
                if "*" in Chem.MolToSmiles(m) and self.lower_molecular_weight < Descriptors.MolWt(m) < self.upper_molecular_weight
            ]

            if not filtered_list:
                target = (self.lower_molecular_weight + self.upper_molecular_weight) / 2
                filtered_list = [min(nodemols, key=lambda x: abs(Descriptors.MolWt(x) - target))]

            filtered_list.sort(key=lambda x: x.GetNumHeavyAtoms())
            return filtered_list
        except Exception:
            return None

    def _generate_smiles(self, scaffold):
        try:
            return self.designer.scaffold_decoration(
                scaffold=scaffold,
                n_samples_per_trial=self.n_samples_per_trial,
                n_trials=self.n_trials,
                sanitize=True,
                do_not_fragment_further=True,
            )
        except Exception:
            return None

    def run_model(self, safe_list):
        results = []
        for s in safe_list:
            row = []
            if s is not None:
                cores = self._extract_core_structure(s)
                if cores:
                    modified = [Chem.MolToSmiles(c).replace("*", "[*]") for c in cores if c is not None]
                    for core in modified:
                        out = self._generate_smiles(core)
                        if out:
                            row += out
            if not row:
                row = [None] * 100
            results.append(row[:100] if len(row) >= 100 else row + [None] * (100 - len(row)))
        return results
