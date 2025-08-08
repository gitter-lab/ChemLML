from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
import numpy as np
import sys

targ_dict = {
        'high0':'HSP90AA1',
        'high1':'ACE',
        'high2':'Mpro',
        'high3':'MAPKK1',
        'med0':'IMPDH',
        'med1':'LSD1',
        'low0':'AChE',
        'low1':'TOPIIB'
        }

targ = sys.argv[1]
targ_sim = targ[0:-1]
targ_idx = int(targ[-1])
targ_name = targ_dict[targ]

# get raw scores
df1 = pd.read_pickle('../docking_results_8targs_chemlml_smi.pkl')
df1 = df1.loc[ df1['target'] == targ ]

if targ_idx == 0:
    df1['compound_set'] = df1['compound_set'].replace('t5molgen_gen_ens','llm_gen_ens')
    df1['compound_set'] = df1['compound_set'].replace('textchemt5_gen_ens','chemt5_gen_ens')
    df1 = df1.loc[ ~( (df1['compound_set'] == 'llm_proto') & ~(df1['molid'].str.contains(targ_sim)) )  ]
    df1 = df1.loc[ ~( (df1['compound_set'] == 'llm_ref')   & ~(df1['molid'].str.contains(targ_sim)) )  ]

df1['compound_set'] = df1['compound_set'].replace('llm_gen_ens','ChemLML(T5+MolGen)')
df1['compound_set'] = df1['compound_set'].replace('t5molxpt_gen_ens','ChemLML(T5+MolXPT)')
df1['compound_set'] = df1['compound_set'].replace('chemt5_gen_ens','Text+ChemT5')
df1['compound_set'] = df1['compound_set'].replace('molt5_gen_ens','MolT5')
df1['compound_set'] = df1['compound_set'].replace('llm_background','ChemLML_bg')

# add mean_qt score
df1['mean_qt'] = df1[['fred_qt','gnina_qt','plants_qt','rdock_qt']].mean(axis=1)

# plot the docking score distributions for each program
fig, axes = plt.subplots( 2, 3, figsize=(24,16), sharey=False, sharex=False )

progs = ['fred_qt','gnina_qt','plants_qt','rdock_qt','mean_qt','ECR_score']

hue_list = ['FDA', 'ChemLML_bg', 'MolT5', 'Text+ChemT5', 'ChemLML(T5+MolGen)', 'ChemLML(T5+MolXPT)' ]

palette = { 'FDA':'tab:grey', 'ChemLML_bg':'tab:blue', 
            'MolT5':'tab:brown', 'Text+ChemT5':'tab:pink', 
            'ChemLML(T5+MolGen)':'tab:orange', 'ChemLML(T5+MolXPT)':'tab:cyan'}

for i, ax1 in enumerate(axes.flat):

    df_temp = df1.loc[ df1['compound_set'].isin( hue_list ) ]
    if progs[i] != 'ECR_score':
        # set reasonable bounds to avoid inclusion of outliers
        stats = df1[ progs[i] ].describe()
        upper_bound = 3.75
        lower_bound = -3.75
        df_temp = df_temp.loc[ (df_temp[progs[i]] <= upper_bound) & (df_temp[progs[i]] >= lower_bound) ]
        ax1.set_xlim( (upper_bound, lower_bound) )

    # plot background cpds (FDA and LLM bg)
    df_temp1 = df_temp.loc[ df_temp['compound_set'].isin( hue_list[0:2] )  ]
    sns.kdeplot( 
            data=df_temp, x=progs[i], hue='compound_set', 
            hue_order=hue_list[0:2], common_norm=True, 
            multiple='layer', cumulative=False, fill=False, linewidth=1,
            alpha=1.0, palette=palette, ax=ax1
            )
    sns.move_legend( ax1, loc='upper left', title='compound sets background' )

    ax1.set_title( progs[i] + " docking scores, targ="+targ_name )


    # plot active cpds (ref and ensemble)
    df_temp2 = df_temp.loc[ df_temp['compound_set'].isin( hue_list[2:] ) ]

    ax2 = ax1.twinx()
    
    ref_score = df1.loc[ df1['compound_set'] == 'llm_ref', progs[i] ].iloc[0]
    proto_score = df1.loc[ df1['compound_set'] == 'llm_proto', progs[i] ].iloc[0]
    
    sns.kdeplot(
            data=df_temp2, x=progs[i], hue='compound_set',
            hue_order=['MolT5','Text+ChemT5','ChemLML(T5+MolGen)','ChemLML(T5+MolXPT)'],
            common_norm=True, multiple='layer', cumulative=False, fill=True, linewidth=0,
            alpha=0.25, palette=palette, ax=ax2
            )

    # Get Legend handles
    handles = ax2.legend_.legend_handles
    labels = [text.get_text() for text in ax2.legend_.texts ]

    if progs[i] == 'ECR_score':
        ax2.set_ylim((0,500))
    else:
        ax2.set_ylim((0,0.33))

    line1 = ax2.axvline( ref_score, linestyle='--', color='black' )
    line2 = ax2.axvline( proto_score, linestyle='--', color='red' )
    
    labels.append('ref')
    labels.append('proto')
    handles.append( Line2D( [0],[0], color='black', linestyle='--', lw=1 ) )
    handles.append( Line2D( [0],[0], color='red', linestyle='--', lw=1 ) )

    lgd = ax2.legend( handles=handles, labels=labels, loc='upper right', title='compound sets active' )


#dump multiplot image

fig.tight_layout( pad=5.0 )
fig.savefig("docking_all_ecr_qt_qtmean_scores_"+targ+"_"+targ_name+".png", dpi=800 )
plt.close()

