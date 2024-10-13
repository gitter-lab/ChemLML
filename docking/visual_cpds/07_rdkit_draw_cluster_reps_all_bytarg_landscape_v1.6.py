
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools
from rdkit.Chem.Draw import rdMolDraw2D

try:
    import Image
except ImportError:
    from PIL import Image
from io import BytesIO

def DrawMolsZoomed( mols, molsPerRow=3, subImgSize=(200, 200), legends=None ):
    nRows = len(mols) // molsPerRow
    if len(mols) % molsPerRow: nRows += 1
    fullSize = (molsPerRow * subImgSize[0], nRows * subImgSize[1])
    full_image = Image.new('RGBA', fullSize )
    for ii, mol in enumerate(mols):
        if mol.GetNumConformers() == 0:
            AllChem.Compute2DCoords(mol)
        column = ii % molsPerRow
        row = ii // molsPerRow
        offset = ( column * subImgSize[0], row * subImgSize[1] )
        d2d = rdMolDraw2D.MolDraw2DCairo(subImgSize[0], subImgSize[1] )
        d2d.drawOptions().legendFontSize=20
        d2d.DrawMolecule(mol, legend=legends[ii] )
        d2d.FinishDrawing()
        sub = Image.open(BytesIO(d2d.GetDrawingText()))
        full_image.paste(sub,box=offset)
    return full_image

# read in 'inhibitor_molecule_canonical_2024_06_13_rdkitcln_P_tidy_clusts_pop.csv'
targ = sys.argv[1]
incsv = sys.argv[2]

df1 = pd.read_csv( incsv )
df2 = df1.loc[ df1['target'] == targ ]
df2 = df2.drop_duplicates( subset='cid_0.750', keep='first' )
#df2 = df2.loc[ df1['medoid_0.750'] == 1 ]
df2 = df2[['molid','rdkit_smiles_cln_protonated','cid_0.750','cpop_0.750']]
df2['cid_0.750'] = df2['cid_0.750'].astype('int')
df2['cpop_0.750'] = df2['cpop_0.750'].astype('int')
df2 = df2.sort_values( by=['cpop_0.750','cid_0.750'], ascending=[False, True] )

clst_id_list = df2['cid_0.750'].tolist()
#clst_id_list = df2['cid_0.750'].sort_values().tolist()
ms = []
ms_titles = []

for clst_id in clst_id_list:
    clst_pop = df2.loc[ df2['cid_0.750'] == clst_id, 'cpop_0.750' ].iloc[0]
    cid = df2.loc[ df2['cid_0.750'] == clst_id, 'molid' ].iloc[0]
    smiles = df2.loc[ df2['cid_0.750'] == clst_id, 'rdkit_smiles_cln_protonated' ].iloc[0]
    ms.append( Chem.MolFromSmiles( smiles ) )
    wrapped_string = "clustID: "+str(clst_id)+"  size: "+str(clst_pop)+"  CID: "+str(cid)
    ms_titles.append( wrapped_string )
    print( "clust_id:{}, clust_pop:{}, molid:{}, smiles:{}".format( clst_id, clst_pop, cid, smiles) )

#img = Draw.MolsToGridImage( ms, molsPerRow=7, subImgSize=(600,600), legends=ms_titles )
img = DrawMolsZoomed( ms, molsPerRow=6, subImgSize=(500,500), legends=ms_titles )
img.save( "cluster_medoids_"+targ+"_0.750_landscape.png")

