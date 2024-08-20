def visual_loss(loss_file,fontsize=18,linewidth=3):
    data = pd.read_csv(loss_file)
    fig,axes = plt.subplots(1,1,figsize=(16,7))

    sns.lineplot(data,x='epoch',y='train_loss',color='#1f77b4',alpha=0.75,label='Training loss',linewidth=linewidth,ax=axes)
    sns.lineplot(data,x='epoch',y='val_loss',color='#ff7f0e',alpha=0.75,label='Validation loss',linewidth=linewidth,ax=axes)

    axes_right = axes.twinx()
    sns.lineplot(data,x='epoch',y='accuracy',color='#2ca02c',alpha=0.85,label='Accuracy',linestyle='--',linewidth=linewidth,ax=axes_right)
    sns.lineplot(data,x='epoch',y='f1',color='#d62728',alpha=0.85,label='F1 Score',linestyle='--',linewidth=linewidth,ax=axes_right)
    
    # axes.text(0,0.4,f'No TIR: Pearson $r$: {corr_no_TIR:.2f}',color='#ff7f0e',alpha=0.7,fontsize=fontsize)

    axes.spines['top'].set_visible(False)
    # axes.spines['right'].set_visible(False)
    axes_right.spines['top'].set_visible(False)
    # axes_right.spines['right'].set_visible(False)
    axes.set_xlabel('Epoch', fontsize=fontsize)
    axes.set_ylabel('', fontsize=fontsize)
    axes_right.set_ylabel('', fontsize=fontsize)
    axes.tick_params(axis='x', labelsize=fontsize) 
    axes.tick_params(axis='y', labelsize=fontsize) 
    axes_right.tick_params(axis='x', labelsize=fontsize) 
    axes_right.tick_params(axis='y', labelsize=fontsize) 
    axes.legend(bbox_to_anchor=(0.1, 1.15), loc='upper center', ncol=2, fontsize=fontsize, frameon=False)
    axes_right.legend(bbox_to_anchor=(0.9, 1.15), loc='upper center', ncol=2, fontsize=fontsize, frameon=False)

    plt.savefig('temp.pdf')
    plt.show()
