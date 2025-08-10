

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

# need an alternative FRAGMENT REMOVER since some
# non-standard counterions are not RDKit's default
# salt fragment list: $RDBASE/Data/Salts.txt
def alt_frag_remover( m ):
    # split m into mol fragments, keep fragment with highest num atoms
    mols = list(Chem.GetMolFrags( m, asMols=True ))
    if (mols):
        mols.sort(reverse=True, key=lambda x: x.GetNumAtoms() )
        mol = mols[0]
    else:
        mol = None
    return mol

# load isolated/edited dock rows for mol gen model
df_dock = pd.read_csv('dock_rows_molt5.csv')

# load original SMILES from mol gen model 
df_smi = pd.read_csv('original_molt5_smi.csv')

# combine, keep unscored mols
df = df_smi.merge( df_dock, on='molid', how='left')

# sort cpds for each target by score, failed molecules rank last
df = df.sort_values( by=['target','ECR_score','molid'], ascending=[True,False,True] )

# remove lines with missing smiles strings (2 out of 100482)
df.dropna( subset=['SMILES'], inplace=True )

# make column to store mol objects
#PandasTools.AddMoleculeColumnToFrame( df, 'smiles', 'rdkit_mol', includeFingerprints=False )
df['rdkit_mol'] = df['SMILES'].apply( lambda smi: Chem.MolFromSmiles( smi ) if smi is not None else None )

# remove lines where smiles string could not generate mol object (2 out of 100480)
df.dropna( subset=['rdkit_mol'], inplace=True )

_saltRemover = SaltRemover.SaltRemover()
df['rdkit_mol'] = df['rdkit_mol'].apply( lambda x: _saltRemover.StripMol(x, dontRemoveEverything=True ) if x is not None else None)

# apply secondary salt remover
df['rdkit_mol'] = df['rdkit_mol'].apply( lambda x: alt_frag_remover(x) if x is not None else None )

# replace smiles with clean smiles
df['rdkit_smiles_cln'] = df['rdkit_mol'].apply( lambda x: Chem.MolToSmiles(x) if x is not None else None )

# for each targ, dump a grid mol image
for t in df['target'].unique(): 
    df_temp = df.loc[ df['target'] == t ]
    ms = df_temp['rdkit_mol'].tolist()
    molids = df_temp['molid'].tolist()
    scores = [ "F A I L" if np.isnan(x) else str(round(x,8)) for x in df_temp['ECR_score'] ]
    molid_score_list = [ i[0] + "\n" + i[1] for i in zip(molids,scores) ]
    img = Draw.MolsToGridImage( ms, molsPerRow=10, subImgSize=(600,600), legends=molid_score_list )
    img.save( "molt5_"+t+"_cpds.png" )
