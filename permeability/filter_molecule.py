import sys
import pandas as pd
import numpy as np
import re
from rdkit import Chem
from rdkit.Chem import SaltRemover, Descriptors
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from collections import Counter
import time

# Import necessary functions from the existing code
def alt_frag_remover(m):
    # split m into mol fragments, keep fragment with highest num atoms
    mols = list(Chem.GetMolFrags(m, asMols=True))
    if (mols):
        mols.sort(reverse=True, key=lambda x: x.GetNumAtoms())
        mol = mols[0]
    else:
        mol = None
    return mol

def smiles_to_rdkit(smi):
    """Convert a SMILES string to an RDKit mol object."""
    if smi is None or smi == '':
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
    except Exception as e:
        mol = None
        print(f"Error converting SMILES '{smi}' to RDKit mol: {e}", file=sys.stderr)
    return mol

def calculate_similarity(mol1, mol2):
    if mol1 is None or mol2 is None:
        return 0.0
    
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def contains_natural_language(smiles):
    pattern = r'^[A-Za-z0-9@+\-\[\]=#%()\\/\.]*$'
    return not bool(re.match(pattern, smiles))

def has_significant_size_reduction(original_smiles, cleaned_smiles):
    if original_smiles is None or cleaned_smiles is None:
        return False
        
    original_length = len(original_smiles)
    cleaned_length = len(cleaned_smiles)
    
    if original_length > 20 and cleaned_length / original_length < 0.5:
        return True
    
    return False

def is_cleaned_subset_of_original(original_smiles, cleaned_smiles):
    if original_smiles is None or cleaned_smiles is None:
        return False
    
    clean_pattern = re.sub(r'[\[\]\(\)=#@\-\+\.:]', '', cleaned_smiles)
    
    if clean_pattern in original_smiles and len(original_smiles) > 2 * len(cleaned_smiles):
        return True
    
    return False

def has_single_element(mol):
    """Check if a molecule contains only one type of element."""
    if mol is None:
        return False
    
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atom_type = [atom for atom in atoms]
    
    unique_atoms = set(atom_type)
    return len(unique_atoms) <= 1

def process_molecule(smiles, salt_remover, similarity_threshold=1.0):
    """
    Process a single molecule through the cleaning pipeline and return its status.
    Returns: (status, cleaned_smiles)
    Status is one of: 'parse_failure', 'natural_language', 'single_element', 
                      'cleaning_failure', 'low_similarity', 'success'
    """
    # First check for natural language
    if contains_natural_language(smiles):
        return 'natural_language', None
    
    # Try parsing with RDKit
    mol = smiles_to_rdkit(smiles)
    if mol is None:
        return 'parse_failure', None
    
    # Check for single element molecules
    if has_single_element(mol):
        return 'single_element', None
    
    # Salt removal
    desalted_mol = salt_remover.StripMol(mol, dontRemoveEverything=True)
    
    # Fragment removal
    cleaned_mol = alt_frag_remover(desalted_mol)
    if cleaned_mol is None:
        return 'cleaning_failure', None
    
    # Check if the cleaned molecule is a single element
    if has_single_element(cleaned_mol):
        return 'single_element', None
    
    # Generate cleaned SMILES
    cleaned_smiles = Chem.MolToSmiles(cleaned_mol)
    
    # Check for natural language via size reduction or subset analysis
    # if has_significant_size_reduction(smiles, cleaned_smiles) or is_cleaned_subset_of_original(smiles, cleaned_smiles):
    #     return 'natural_language', None
    
    # Calculate similarity
    similarity = calculate_similarity(mol, cleaned_mol)
    if similarity < similarity_threshold:
        return 'low_similarity', cleaned_smiles
    
    return 'success', cleaned_smiles

def get_unique_preserving_order(seq):
    return list(dict.fromkeys(seq))

def main():
    print("Starting sequential molecule filtering...")
    start_time = time.time()
    
    # Determine which file to load
    try:
        file_path = "evaluation/permeability_molecule_ChemT5_MolGen.pkl"
        with open(file_path, "rb") as f:
            generated_molecules = pickle.load(f)
            molecules = get_unique_preserving_order(generated_molecules[0])
        print(f"Loaded {len(molecules)} molecules from pickle file")
    except:
        try:
            # Fall back to CSV file if pickle doesn't exist
            file_path = "evaluation/permeability_ChemT5.csv"
            df = pd.read_csv(file_path)
            molecules = df['smiles'].tolist()
            print(f"Loaded {len(molecules)} molecules from CSV file")
        except:
            print("Error: Could not load molecule data. Make sure the file exists.")
            return
    
    # Initialize salt remover
    salt_remover = SaltRemover.SaltRemover()
    
    # Counters for tracking molecule statuses
    status_counts = Counter()
    successful_molecules = []
    processed_count = 0
    target_success_count = 100
    
    # Lists to store molecules for each status for examples
    example_molecules = {
        'parse_failure': [],
        'natural_language': [],
        'single_element': [],
        'cleaning_failure': [],
        'low_similarity': [],
        'success': []
    }
    
    # Process molecules sequentially
    for smiles in molecules:
        processed_count += 1
        status, cleaned_smiles = process_molecule(smiles, salt_remover)
        status_counts[status] += 1
        
        # Save example molecules (up to 5 per category)
        if len(example_molecules[status]) < 5:
            example_molecules[status].append((smiles, cleaned_smiles))
        
        # Save successful molecules
        if status == 'success':
            successful_molecules.append(cleaned_smiles)
            
        # Stop when target number of successful molecules is reached
        if len(successful_molecules) >= target_success_count:
            break
    
    # Calculate time taken
    elapsed_time = time.time() - start_time
    
    # Results
    print(f"\nSequential Processing Complete!")
    print(f"Processed {processed_count} molecules to get {len(successful_molecules)} successful ones")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print("\nStatus Counts:")
    
    for status, count in status_counts.items():
        print(f"  {status}: {count} molecules ({count/processed_count:.1%})")
    
    # Save results to file
    with open("evaluation/sequential_filter_report.md", "w") as f:
        f.write("# Sequential Molecule Filtering Analysis\n\n")
        f.write("## Overview\n\n")
        f.write(f"This analysis processed molecules one by one until {target_success_count} successful molecules were found.\n\n")
        f.write(f"- **Total molecules processed**: {processed_count}\n")
        f.write(f"- **Success rate**: {len(successful_molecules)/processed_count:.2%}\n")
        f.write(f"- **Time taken**: {elapsed_time:.2f} seconds\n\n")
        
        f.write("## Detailed Breakdown\n\n")
        f.write("| Status | Count | Percentage |\n")
        f.write("|--------|------:|-----------:|\n")
        
        for status, count in status_counts.most_common():
            f.write(f"| {status.replace('_', ' ').title()} | {count} | {count/processed_count:.1%} |\n")
        
        f.write("\n## Example Molecules\n\n")
        
        for status, examples in example_molecules.items():
            if examples:
                f.write(f"### {status.replace('_', ' ').title()} Examples\n\n")
                f.write("```\n")
                for i, (original, cleaned) in enumerate(examples, 1):
                    f.write(f"{i}. Original: {original}\n")
                    if cleaned:
                        f.write(f"   Cleaned:  {cleaned}\n")
                    f.write("\n")
                f.write("```\n\n")
    
    # Save successful molecules in the format needed
    with open("evaluation/cleaned_permeability_molecule_ChemT5_MolXPT.csv", "w") as f:
        f.write("smiles\n")  # Header with single column
        for smiles in successful_molecules:
            f.write(f"{smiles}\n")

if __name__ == "__main__":
    main()