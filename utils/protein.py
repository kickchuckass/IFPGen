from rdkit import Chem
import numpy as  np
class PDBProtein(object):
    AA_NAME_SYM = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
                   'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
                   'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V', 'HET': 'X'}

    AA_NAME_NUMBER = {
        k: i for i, (k, _) in enumerate(AA_NAME_SYM.items())
    }

    BACKBONE_NAMES = ["CA", "C", "N", "O"]

    def __init__(self, data, mode='auto', allow_het=[], if_H=False):
        super().__init__()
        if (data[-4:].lower() == '.pdb' and mode == 'auto') or mode == 'path':
            with open(data, 'r') as f:
                self.block = f.read()
        else:
            self.block = data

        self.ptable = Chem.GetPeriodicTable()
        self.allow_het = allow_het
        self.if_H = if_H
        if self.if_H:
            self.L_element = ['D', 'X']
        else:
            self.L_element = ['H', 'D', 'X']
        # self.include_het = include_het

        # Molecule properties
        self.title = None
        # Atom properties
        self.atoms = []
        self.element = []
        self.atomic_weight = []
        self.pos = []
        self.atom_name = []
        self.is_backbone = []
        self.atom_to_aa_type = []
        self.atom2residue = []
        self.atom_lines = []
            # extra properties
        self.atom2residue_idx = []
        self.atom2chain = []
        
        # Residue properties
        self.residues = []
        self.amino_acid = []
        self.amino_idx = []
        self.center_of_mass = []
        self.pos_CA = []
        self.pos_C = []
        self.pos_N = []
        self.pos_O = []
        self.residue_natoms = []
        self.seq = []

        self._parse()

    def _enum_formatted_atom_lines(self):
        for line in self.block.splitlines():
            header = line[0:6].strip()
            res_name = line[17:20].strip()
            res_flag = (isinstance(self.allow_het, bool) and self.allow_het) \
                or (isinstance(self.allow_het, list) and res_name in self.allow_het)
            if header == 'ATOM' or (header == 'HETATM' and res_flag):
                element_symb = line[76:78].strip().capitalize()
                if len(element_symb) == 0:
                    element_symb = line[13:14] if header == 'ATOM' else line[12:16].strip().capitalize()
                yield {
                    'line': line,
                    'type': header,
                    'atom_id': int(line[6:11]),
                    'atom_name': line[12:16].strip(),
                    'alt_loc': line[16],
                    'res_name': line[17:20].strip() if header == 'ATOM' else 'HET', # HETATM
                    'chain': line[21:22].strip(),
                    'res_id': int(line[22:26]),
                    'res_insert_id': line[26:27].strip(),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'occupancy': float(line[54:60]),
                    'segment': line[72:76].strip(),
                    'element_symb': element_symb,
                    'charge': line[78:80].strip(),
                }
            elif line[0:6].strip() == 'HEADER':
                yield {
                    'type': 'HEADER',
                    'value': line[10:].strip()
                }
            elif line[0:6].strip() == 'ENDMDL':
                break  # Some PDBs have more than 1 model.
            else:
                yield {
                    'type': 'others'
                }

    def _parse(self):
        # Process atoms
        residues_tmp = {}
        num_residue = -1
        for atom in self._enum_formatted_atom_lines():
            # print(atom)
            if atom['type'] == 'HEADER':
                self.title = atom['value'].lower()
                continue
            if atom['type'] == 'others':
                continue
            if atom['element_symb'][0] in self.L_element or atom['atom_name'] == 'OXT':
            # if atom['atom_name'][0] == 'H' or atom['atom_name'] == 'OXT':
                continue
            next_ptr = len(self.element)


            chain_res_id = '%s_%s_%d_%s' % (atom['chain'], atom['segment'], atom['res_id'], atom['res_insert_id'])
            if chain_res_id not in residues_tmp:
                num_residue += 1
                residues_tmp[chain_res_id] = {
                    'name': atom['res_name'],
                    'atoms': [next_ptr],
                    'chain': atom['chain'],
                    'segment': atom['segment'],
                    'res_id': atom['res_id'],
                    'full_seq_idx': num_residue,
                    'alt_loc': atom['alt_loc']
                }
            else:
                assert residues_tmp[chain_res_id]['name'] == atom['res_name']
                assert residues_tmp[chain_res_id]['chain'] == atom['chain']
                if atom['alt_loc'] != residues_tmp[chain_res_id]['alt_loc']:
                    continue
                residues_tmp[chain_res_id]['atoms'].append(next_ptr)

            atomic_number = self.ptable.GetAtomicNumber(atom['element_symb'])
            self.atoms.append(atom)
            self.element.append(atomic_number)
            self.atomic_weight.append(self.ptable.GetAtomicWeight(atomic_number))
            self.pos.append(np.array([atom['x'], atom['y'], atom['z']], dtype=np.float32))
            self.atom_name.append(atom['atom_name'])
            self.is_backbone.append(atom['atom_name'] in self.BACKBONE_NAMES and atom['res_name'] != 'HET')
            self.atom_to_aa_type.append(self.AA_NAME_NUMBER.get(atom['res_name'], 0))
            self.atom2residue.append(num_residue)
            self.atom_lines.append(atom['line'])

            # extra info
            self.atom2residue_idx.append(atom['res_id'])
            self.atom2chain.append(atom['chain'])

        # Process residues
        self.residues = [r for _, r in residues_tmp.items()]

        for residue in self.residues:
            sum_pos = np.zeros([3], dtype=np.float32)
            sum_mass = 0.0
            for atom_idx in residue['atoms']:
                sum_pos += self.pos[atom_idx] * self.atomic_weight[atom_idx]
                sum_mass += self.atomic_weight[atom_idx]
                if self.atom_name[atom_idx] in self.BACKBONE_NAMES:
                    residue['pos_%s' % self.atom_name[atom_idx]] = self.pos[atom_idx]
            residue['center_of_mass'] = sum_pos / sum_mass
            self.residue_natoms.append(len(residue['atoms']))
            # print(residue, residue['name'], len(residue['atoms']))
            # assert len(residue['atoms']) <= NUM_ATOMS[self.AA_NAME_NUMBER[residue['name']]]

            # Process backbone atoms of residues
            self.amino_acid.append(self.AA_NAME_NUMBER.get(residue['name'], 0))
            self.center_of_mass.append(residue['center_of_mass'])
            self.amino_idx.append(residue['res_id'])
            self.seq.append(self.AA_NAME_SYM.get(residue['name'], 'X'))
            for name in self.BACKBONE_NAMES:
                pos_key = 'pos_%s' % name  # pos_CA, pos_C, pos_N, pos_O
                if pos_key in residue:
                    getattr(self, pos_key).append(residue[pos_key])
                else:
                    getattr(self, pos_key).append(residue['center_of_mass'])

        # convert atom_name to number
        # self.atom_name = np.array([ATOM_TYPES.index(atom) for atom in self.atom_name])
        self.pos = np.array(self.pos, dtype=np.float32)

    def to_dict_atom(self, include_lines=False):
        atom_dict = {
            'element': np.array(self.element, dtype=np.longlong),
            'molecule_name': self.title,
            'pos': self.pos,
            'is_backbone': np.array(self.is_backbone, dtype=bool),
            'atom_name': self.atom_name,
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.longlong),
            'atom2residue': np.array(self.atom2residue, dtype=np.longlong)
        }
        if include_lines:
            atom_dict['lines'] = np.array(self.atom_lines)

        return atom_dict

    def to_dict_residue(self):
        return {
            'seq': self.seq,
            'res_idx': np.array(self.amino_idx, dtype=np.longlong),
            'amino_acid': np.array(self.amino_acid, dtype=np.longlong),
            'center_of_mass': np.array(self.center_of_mass, dtype=np.float32),
            'pos_CA': np.array(self.pos_CA, dtype=np.float32),
            'pos_C': np.array(self.pos_C, dtype=np.float32),
            'pos_N': np.array(self.pos_N, dtype=np.float32),
            'pos_O': np.array(self.pos_O, dtype=np.float32),
            'residue_natoms': np.array(self.residue_natoms, dtype=np.longlong),
        }

    def to_dict_atom_2(self, include_lines=False):
        atom_dict = {
            'element': np.array(self.element, dtype=np.longlong),
            'atom_name': self.atom_name,
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.longlong),
            'atom2residue_idx': self.atom2residue_idx,
            'atom2chain': self.atom2chain,
        }
        if include_lines:
            atom_dict['lines'] = np.array(self.atom_lines)

        return atom_dict
    
    def query_residues_radius(self, center, radius, criterion='center_of_mass'):
        center = np.array(center).reshape(3)
        selected = []
        for residue in self.residues:
            distance = np.linalg.norm(residue[criterion] - center, ord=2)
            if distance < radius:
                selected.append(residue)
        return selected
    
    def query_residues_ligand_v0(self, ligand, radius, criterion='center_of_mass'):
        selected = []
        sel_idx = set()
        # The time-complexity is O(mn).
        for center in ligand['pos']:
            for i, residue in enumerate(self.residues):
                distance = np.linalg.norm(residue[criterion] - center, ord=2)
                if distance < radius and i not in sel_idx:
                    selected.append(residue)
                    sel_idx.add(i)
        return selected
    
    def query_residues_ligand(self, ligand, radius=3.5, selected_residue=None, return_mask=True):
        selected = []
        sel_idx = set()
        selected_mask = np.zeros(len(self.residues), dtype=bool)
        full_seq_idx = set()
        if selected_residue is None:
            selected_residue = self.residues
        # The time-complexity is O(mn).
        for i, residue in enumerate(selected_residue):
            for center in ligand['pos']:
                distance = np.min(np.linalg.norm(self.pos[residue['atoms']] - center, ord=2, axis=1))
                if distance <= radius and i not in sel_idx:
                    selected.append(residue)
                    sel_idx.add(i)
                    full_seq_idx.add(residue['full_seq_idx'])
                    break
        selected_mask[list(sel_idx)] = 1
        if return_mask:
            return list(full_seq_idx), selected_mask
        return list(full_seq_idx), selected

    # can be used for select pocket residues
    def residues_to_pdb_block(self, residues, name='POCKET'):
        block = "HEADER    %s\n" % name
        block += "COMPND    %s\n" % name
        for residue in residues:
            for atom_idx in residue['atoms']:
                block += self.atoms[atom_idx]['line'] + "\n"
        block += "END\n"
        return block

    def return_residues(self):
        return self.residues, self.atoms
