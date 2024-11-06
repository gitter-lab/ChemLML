


from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_pickle('docking_results_llm_high_medium_low_smi.pkl')

fig, axes = plt.subplots( 3, 1, figsize=(10,15) )

targets = [ 'low','medium','high' ]

#kws = ["layer", "dodge", "stack", "fill"]

for i, ax in enumerate(axes):
    df_temp = df.dropna( subset='ECR_score_'+targets[i] )
    hue_list = ['FDA', 'llm_background', 'llm_gen_ens_'+targets[i], 'llm_gen_ref_'+targets[i] ]
    df_temp = df.loc[ df['compound_set'].isin( hue_list ) ]
    #sns.histplot( data=df_temp, x='ECR_score_'+targets[i], hue='compound_set', multiple='dodge', ax=ax )
    #sns.histplot( data=df_temp, x='ECR_score_'+targets[i], kde=True, common_norm=False, hue='compound_set', hue_order=hue_list, multiple='layer', ax=ax )
    sns.histplot( 
        data=df_temp, x='ECR_score_'+targets[i], bins=100, stat='density', 
        log_scale=True, common_norm=False, hue='compound_set', element='step',
        fill=True, alpha=0.5, palette='bright', hue_order=hue_list, 
        multiple='layer', linewidth=0, ax=ax 
    )
    sns.move_legend(ax, "upper left")
    ax.set_title( targets[i]+' target ECR scores by compound subset')
    #ax.set_ylim( (0,100) )

fig.tight_layout( pad=5.0 )
fig.savefig("all_targets_ecr_score_pdfs_logscale.png", dpi=800 )
plt.close()

