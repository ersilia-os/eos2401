import os
import math
import random
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

            starred = [m for m in nodemols if "*" in Chem.MolToSmiles(m) and m.GetNumHeavyAtoms() > 0]

            filtered_list = [
                m for m in starred
                if self.lower_molecular_weight < Descriptors.MolWt(m) < self.upper_molecular_weight
            ]

            if not filtered_list:
                # Fall back to the closest-MW candidate, but only among nodes that still carry
                # an attachment point. A scaffold with no "*" has nowhere for scaffold_decoration
                # to grow from, and was previously selected here anyway: confirmed empirically
                # to make the model return chemically disconnected garbage (e.g. an unattached
                # benzene ring tacked onto an unrelated generated fragment via "."), not an error,
                # so it was never caught by any validity check. If nothing with an attachment
                # point exists at all, there's honestly no usable core, and we return None so the
                # caller pads with an empty row instead of returning that garbage as data.
                if not starred:
                    return None
                target = (self.lower_molecular_weight + self.upper_molecular_weight) / 2
                filtered_list = [min(starred, key=lambda x: abs(Descriptors.MolWt(x) - target))]

            # secondary sort key keeps tie-breaking (same heavy-atom count) deterministic instead
            # of depending on scaffold-network node enumeration order
            filtered_list.sort(key=lambda x: (x.GetNumHeavyAtoms(), Chem.MolToSmiles(x)))
            return filtered_list
        except Exception:
            return None

    def _generate_smiles(self, scaffold, n_samples=None):
        # n_samples lets callers request a specific budget for this one core (see run_model),
        # instead of always generating self.n_samples_per_trial regardless of how many cores
        # will share the final output. A small buffer is requested to offset sanitization loss
        # (observed ~1-11% of samples fail sanitization and get dropped), then the caller trims
        # back down to the exact allocated share.
        target = n_samples if n_samples is not None else self.n_samples_per_trial
        if target <= 0:
            return []
        n_request = max(target, math.ceil(target * 1.15))
        try:
            return self.designer.scaffold_decoration(
                scaffold=scaffold,
                n_samples_per_trial=n_request,
                n_trials=self.n_trials,
                sanitize=True,
                do_not_fragment_further=True,
            )
        except Exception:
            return None

    def run_model(self, safe_list, total_budget=100, seed=42):
        # one RNG for the whole call (not reset per input) so different inputs/cores don't
        # end up drawing the exact same shuffle pattern, while the overall call stays fully
        # deterministic given the same seed and input order.
        rng = random.Random(seed)
        results = []
        for s in safe_list:
            row = []
            seen = set()  # canonical SMILES already emitted for this input, across ALL cores
            if s is not None:
                cores = self._extract_core_structure(s)
                if cores:
                    modified = [Chem.MolToSmiles(c).replace("*", "[*]") for c in cores if c is not None]
                    n_cores = len(modified)
                    # split the output budget evenly across every qualifying core instead of
                    # generating a full batch per core and truncating from the front, which
                    # silently discarded every core but the first (see #1810).
                    base_share = total_budget // n_cores
                    remainder = total_budget % n_cores
                    for idx, core in enumerate(modified):
                        share = base_share + (1 if idx < remainder else 0)
                        out = self._generate_smiles(core, n_samples=share)
                        if not out:
                            continue
                        # dedupe (canonical SMILES, both within this core's own batch and
                        # against everything already collected for this input from other
                        # cores), then randomly draw this core's exact share from what's left
                        # instead of just taking the first `share` in generation order
                        candidates = {}
                        for o in out:
                            if not o:
                                continue
                            m = Chem.MolFromSmiles(o)
                            if m is None:
                                continue
                            # scaffold_decoration's raw output is not guaranteed to be a single
                            # connected molecule: it can return salt/ion pairs or two unrelated
                            # fragments glued together via "." (e.g. "NC=S.S=C1SN2CC=C1CC2"),
                            # which RDKit parses without complaint. Reject anything but a single
                            # fragment rather than passing it through as a generated compound.
                            if len(Chem.GetMolFrags(m)) > 1:
                                continue
                            key = Chem.MolToSmiles(m)
                            if key in seen or key in candidates:
                                continue
                            candidates[key] = o
                        pool = list(candidates.items())
                        rng.shuffle(pool)
                        for key, o in pool[:share]:
                            seen.add(key)
                            row.append(o)
            if not row:
                row = [None] * total_budget
            results.append(row[:total_budget] if len(row) >= total_budget else row + [None] * (total_budget - len(row)))
        return results
