from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys

targ = sys.argv[1]

# get raw scores
df1 = pd.read_pickle('scores_'+targ+'_ranking_ECR.pkl')
df2 = df1.reset_index()

# get pickle with comopund subset labels
df = pd.read_pickle('docking_results_llm_high_medium_low_smi.pkl')
df = df[['molid','compound_set','smiles']]

# add the compound subset labels to raw scores
df3 = df2.merge(df, on='molid', how='left' )

# remove cpds with imputed (mean) scores
if targ == "low":
    df3 = df3.loc[ ~((df3['fred'] <= -9.457) & (df3['fred'] >= -9.459)) ]
    df3 = df3.loc[ ~((df3['gnina'] <= -6.100895) & (df3['gnina'] >= -6.100897)) ]

# add mean_qt score
df3['mean_qt'] = df3[['fred_qt','gnina_qt','plants_qt','rdock_qt']].mean(axis=1)

# plot the docking score distributions for each program
fig, axes = plt.subplots( 2, 3, figsize=(24,16), sharey=False, sharex=False )

progs = ['fred_qt','gnina_qt','plants_qt','rdock_qt','mean_qt','ECR_score']

hue_list = ['FDA', 'llm_background', 'llm_gen_ref_'+targ, 'llm_gen_ens_'+targ ]
#hue_list = ['FDA', 'llm_background', 'llm_gen_ens_'+targ, 'llm_gen_ref_'+targ ]

palette = { 'FDA':'tab:blue', 'llm_background':'tab:green', 'llm_gen_ref_'+targ:'tab:red', 'llm_gen_ens_'+targ:'tab:orange' }

for i, ax1 in enumerate(axes.flat):

    df_temp = df3.loc[ df['compound_set'].isin( hue_list ) ]

    if progs[i] != 'ECR_score':
        # set reasonable bounds to avoid inclusion of outliers
        stats = df3[ progs[i] ].describe()
        upper_bound = 3.75
        lower_bound = -3.75
        df_temp = df_temp.loc[ (df_temp[progs[i]] <= upper_bound) & (df_temp[progs[i]] >= lower_bound) ]
        ax1.set_xlim( (upper_bound, lower_bound) )
        ax1.set_ylim( (0,90) )
    elif progs[i] == 'ECR_score':
        ax1.set_ylim( (0,100) )

    # compute bin intervals based on all scores
    bins = np.histogram( df_temp[ progs[i] ].values, bins=100 )[1].tolist()


    # plot background cpds (FDA and LLM bg)
    df_temp1 = df_temp.loc[ df_temp['compound_set'].isin( hue_list[0:2] )  ]
    sns.histplot( 
            data=df_temp1, x=progs[i], hue='compound_set', 
            hue_order=hue_list[0:2], bins=bins, multiple='layer',
            alpha=0.5, palette=palette, ax=ax1
            )
    sns.move_legend( ax1, loc='upper left', title='compound sets background' )

    ax1.set_title( progs[i] + " docking scores, targ="+targ )


    # plot active cpds (ref and ensemble)
    df_temp2 = df_temp.loc[ df_temp['compound_set'].isin( hue_list[2:4] ) ]

    ax2 = ax1.twinx()

    sns.histplot(
            data=df_temp2, x=progs[i], hue='compound_set',
            hue_order=hue_list[2:4], bins=bins, multiple='layer',
            alpha=1.00, palette=palette, ax=ax2
            )
    sns.move_legend(ax2, loc='upper right', title='compound sets active' )
    ax2.set_ylim((0,10))


#dump multiplot image
fig.tight_layout( pad=5.0 )
fig.savefig("docking_all_ecr_qt_qtmean_scores_"+targ+"_layer.png", dpi=800 )
plt.close()

