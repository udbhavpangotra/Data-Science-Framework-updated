from dataset import dataset_reading

train_df,test_df = dataset_reading(train_data_location,test_data_location)

def visualizations():

    plt.rcParams['figure.dpi'] = 300
    fig = plt.figure(figsize=(5, 5), facecolor='#f6f5f5')
    gs = fig.add_gridspec(3, 2)
    gs.update(wspace=0.4, hspace=0.8)

    background_color = "#f6f5f5"

    # background_color = "#f6f5f5"
    column = 'Survived'
    color_map = ['#eeb977', 'lightgray']
    sns.set_palette(sns.color_palette(color_map))
    temp_train = pd.DataFrame(train_df[column].value_counts()).reset_index(drop=False)
    ax0 = fig.add_subplot(gs[0, 0])
    for s in ["right", "top"]:
        ax0.spines[s].set_visible(False)
    ax0.set_facecolor(background_color)
    ax0.tick_params(axis = "y", which = "both", left = False)
    ax0.text(-1, 83, 'Survival Rate on the training data', color='black', fontsize=7, ha='left', va='bottom', weight='bold')
    # ax0.text(-1, 82, 'Survival Rate ', color='#292929', fontsize=5, ha='left', va='top')
    # ax0.text(1.18, 73.3, 'for age and fare', color='#292929', fontsize=4, ha='left', va='bottom')
    ax0_sns = sns.barplot(ax=ax0, x=temp_train['index'], y=temp_train[column]/1000, zorder=2)
    ax0_sns.set_xlabel("Survived",fontsize=5, weight='bold')
    ax0_sns.set_ylabel('')
    ax0.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax0_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE')
    ax0_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE')
    ax0_sns.tick_params(labelsize=5)
    ax0_sns.legend(['Survived', 'Not Survived'], ncol=2, facecolor=background_color, edgecolor=background_color, fontsize=4, bbox_to_anchor=(-0.26, 1.3), loc='upper left')
    leg = ax0_sns.get_legend()
    leg.legendHandles[0].set_color('#eeb977')
    leg.legendHandles[1].set_color('lightgray')




    nan_data = (train_df.isna().sum().sort_values(ascending=False) / len(train_df) * 100)[:6]
    nan_data_1 = pd.DataFrame(data = nan_data,columns=["Missing % "]).reset_index()
    a4_dims = (11.7, 8.27)
    plt.rcParams['figure.dpi'] = 300
    fig = plt.figure(figsize=(2, 2), facecolor='#f6f5f5')
    gs = fig.add_gridspec(1, 1)
    gs.update(wspace=0.4, hspace=0.8)

    background_color = "#f6f5f5"

    column = 'Missing % '
    color_map = ['#eeb977', 'lightgray']
    sns.set_palette(sns.color_palette(color_map))
    temp_train = nan_data_1
    ax0 = fig.add_subplot(gs[0, 0])
    for s in ["right", "top"]:
        ax0.spines[s].set_visible(False)
    ax0.set_facecolor(background_color)

    ax0.tick_params(axis = "y", which = "both", left = False)
    ax0.text(-1, 5, '% of Missing values for Training Data', color='black', fontsize=7, ha='left', va='bottom', weight='bold')
    # ax0.text(-1, 5, 'Survival Rate ', color='#292929', fontsize=5, ha='left', va='top')
    ax0_sns = sns.barplot(ax=ax0, x=temp_train['index'], y=temp_train[column], zorder=2 )
    ax0_sns.set_xlabel("Column Names",fontsize=4, weight='bold')
    ax0_sns.set_ylabel("Percentage",fontsize=4, weight='bold')
    ax0.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax0_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE')
    ax0_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE')
    ax0_sns.tick_params(labelsize=2)




    plt.rcParams['figure.dpi'] = 300
    fig = plt.figure(figsize=(5, 5), facecolor='#f6f5f5')
    gs = fig.add_gridspec(4, 3)
    gs.update(wspace=0.4, hspace=0.8)

    background_color = "#f6f5f5"
    color_map = ['#eeb977', 'lightgray']
    sns.set_palette(sns.color_palette(color_map))

    column = 'Pclass'
    temp_train = pd.DataFrame(train_df[column].value_counts()).reset_index(drop=False)
    temp_train['source'] = 'train'
    temp_test = pd.DataFrame(test_df[column].value_counts()).reset_index(drop=False)
    temp_test['source'] = 'test'
    temp_combine = pd.concat([temp_train, temp_test], axis=0)
    ax0 = fig.add_subplot(gs[0, 0])
    for s in ["right", "top"]:
        ax0.spines[s].set_visible(False)
    ax0.set_facecolor(background_color)
    ax0.tick_params(axis = "y", which = "both", left = False)
    ax0.text(-1.2, 88, 'Features comparison', color='black', fontsize=7, ha='left', va='bottom', weight='bold')
    ax0.text(-1.2, 87, 'Comparing features distribution between train and test dataset', color='#292929', fontsize=5, ha='left', va='top')
    ax0_sns = sns.barplot(ax=ax0, x=temp_combine['index'], y=temp_combine[column]/1000, zorder=2, hue=temp_combine['source'])
    ax0_sns.set_xlabel("Ticket Class",fontsize=5, weight='bold')
    ax0_sns.set_ylabel('')
    ax0.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax0_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE')
    ax0_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE')
    ax0_sns.tick_params(labelsize=5)
    ax0_sns.legend(ncol=2, facecolor=background_color, edgecolor=background_color, fontsize=4, bbox_to_anchor=(0.46, 1.22))

    column = 'Sex'
    temp_train = pd.DataFrame(train_df[column].value_counts()).reset_index(drop=False)
    temp_train['source'] = 'train'
    temp_test = pd.DataFrame(test_df[column].value_counts()).reset_index(drop=False)
    temp_test['source'] = 'test'
    temp_combine = pd.concat([temp_train, temp_test], axis=0)
    ax1 = fig.add_subplot(gs[0, 1])
    for s in ["right", "top"]:
        ax1.spines[s].set_visible(False)
    ax1.set_facecolor(background_color)
    ax1.legend(prop={'size': 3})
    ax1.tick_params(axis = "y", which = "both", left = False)
    ax1_sns = sns.barplot(ax=ax1, x=temp_combine['index'], y=temp_combine[column]/1000, zorder=2, hue=temp_combine['source'])
    ax1_sns.set_xlabel('Sex', fontsize=5, weight='bold')
    ax1_sns.set_ylabel('')
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax1_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE')
    ax1_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE')
    ax1_sns.tick_params(labelsize=5)
    ax1_sns.get_legend().remove()

    column = 'Age'
    ax3 = fig.add_subplot(gs[0, 2])
    for s in ["right", "top"]:
        ax3.spines[s].set_visible(False)
    ax3.set_facecolor(background_color)
    ax3.legend(prop={'size': 3})
    ax3.tick_params(axis = "y", which = "both", left = False)
    ax3_sns = sns.kdeplot(ax=ax3, x=train_df['Age'], zorder=2, shade=True)
    ax3_sns = sns.kdeplot(ax=ax3, x=test_df['Age'], zorder=2, shade=True)
    ax3_sns.set_xlabel('Age', fontsize=5, weight='bold')
    ax3_sns.set_ylabel('')
    ax3_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE')
    ax3_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE')
    ax3_sns.tick_params(labelsize=5)
    ax3_sns.get_legend().remove()

    column = 'SibSp'
    temp_train = pd.DataFrame(train_df[column].value_counts()).reset_index(drop=False)
    temp_train['source'] = 'train'
    temp_test = pd.DataFrame(test_df[column].value_counts()).reset_index(drop=False)
    temp_test['source'] = 'test'
    temp_combine = pd.concat([temp_train, temp_test], axis=0)
    ax4 = fig.add_subplot(gs[1, 0])
    for s in ["right", "top"]:
        ax4.spines[s].set_visible(False)
    ax4.set_facecolor(background_color)
    ax4.legend(prop={'size': 3})
    ax4.tick_params(axis = "y", which = "both", left = False)
    ax4_sns = sns.barplot(ax=ax4, x=temp_combine['index'], y=temp_combine[column]/1000, zorder=2, hue=temp_combine['source'])
    ax4_sns.set_xlabel('Siblings / spouse', fontsize=5, weight='bold')
    ax4_sns.set_ylabel('')
    ax4.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax4_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE')
    ax4_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE')
    ax4_sns.tick_params(labelsize=5)
    ax4_sns.get_legend().remove()

    column = 'Parch'
    temp_train = pd.DataFrame(train_df[column].value_counts()).reset_index(drop=False)
    temp_train['source'] = 'train'
    temp_test = pd.DataFrame(test_df[column].value_counts()).reset_index(drop=False)
    temp_test['source'] = 'test'
    temp_combine = pd.concat([temp_train, temp_test], axis=0)
    ax5 = fig.add_subplot(gs[1, 1])
    for s in ["right", "top"]:
        ax5.spines[s].set_visible(False)
    ax5.set_facecolor(background_color)
    ax5.legend(prop={'size': 3})
    ax5.tick_params(axis = "y", which = "both", left = False)
    ax5_sns = sns.barplot(ax=ax5, x=temp_combine['index'], y=temp_combine[column]/1000, zorder=2, hue=temp_combine['source'])
    ax5_sns.set_xlabel('Parents / children', fontsize=5, weight='bold')
    ax5_sns.set_ylabel('')
    ax5.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax5_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE')
    ax5_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE')
    ax5_sns.tick_params(labelsize=5)
    ax5_sns.get_legend().remove()

    column = 'Fare'
    ax6 = fig.add_subplot(gs[1, 2])
    for s in ["right", "top"]:
        ax6.spines[s].set_visible(False)
    ax6.set_facecolor(background_color)
    ax6.legend(prop={'size': 3})
    ax6.tick_params(axis = "y", which = "both", left = False)
    ax6_sns = sns.kdeplot(ax=ax6, x=train_df['Fare'], zorder=2, shade=True)
    ax6_sns = sns.kdeplot(ax=ax6, x=test_df['Fare'], zorder=2, shade=True)
    ax6_sns.set_xlabel('Fare', fontsize=5, weight='bold')
    ax6_sns.set_ylabel('')
    ax6_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE')
    ax6_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE')
    ax6_sns.tick_params(labelsize=5)
    ax6_sns.get_legend().remove()

    train_df["Cabin"] = train_df["Cabin"].fillna("No")
    train_df["Cabin_code"] = train_df["Cabin"].str[0]
    test_df["Cabin"] = test_df["Cabin"].fillna("No")
    test_df["Cabin_code"] = test_df["Cabin"].str[0]

    column = 'Cabin_code'
    temp_train = pd.DataFrame(train_df[column].value_counts()).reset_index(drop=False)
    temp_train['source'] = 'train'
    temp_test = pd.DataFrame(test_df[column].value_counts()).reset_index(drop=False)
    temp_test['source'] = 'test'
    temp_combine = pd.concat([temp_train, temp_test], axis=0)
    ax7 = fig.add_subplot(gs[2, 0])
    for s in ["right", "top"]:
        ax7.spines[s].set_visible(False)
    ax7.set_facecolor(background_color)
    ax7.legend(prop={'size': 3})
    ax7.tick_params(axis = "y", which = "both", left = False)
    ax7_sns = sns.barplot(ax=ax7, x=temp_combine['index'], y=temp_combine[column]/1000, zorder=2, hue=temp_combine['source'])
    ax7_sns.set_xlabel('Cabin', fontsize=5, weight='bold')
    ax7_sns.set_ylabel('')
    ax7.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax7_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE')
    ax7_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE')
    ax7_sns.tick_params(labelsize=5)
    ax7_sns.get_legend().remove()

    train_df["Embarked"] = train_df["Embarked"].fillna("N")
    test_df["Embarked"] = test_df["Embarked"].fillna("N")

    column = 'Embarked'
    temp_train = pd.DataFrame(train_df[column].value_counts()).reset_index(drop=False)
    temp_train['source'] = 'train'
    temp_test = pd.DataFrame(test_df[column].value_counts()).reset_index(drop=False)
    temp_test['source'] = 'test'
    temp_combine = pd.concat([temp_train, temp_test], axis=0)
    ax8 = fig.add_subplot(gs[2, 1])
    for s in ["right", "top"]:
        ax8.spines[s].set_visible(False)
    ax8.set_facecolor(background_color)
    ax8.legend(prop={'size': 3})
    ax8.tick_params(axis = "y", which = "both", left = False)
    ax8_sns = sns.barplot(ax=ax8, x=temp_combine['index'], y=temp_combine[column]/1000, zorder=2, hue=temp_combine['source'])
    ax8_sns.set_xlabel('Port', fontsize=5, weight='bold')
    ax8_sns.set_ylabel('')
    ax8.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax8_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE')
    ax8_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE')
    ax8_sns.tick_params(labelsize=5)
    ax8_sns.get_legend().remove()

    plt.show()





    plt.rcParams['figure.dpi'] = 300
    fig = plt.figure(figsize=(5, 6), facecolor='#f6f5f5')
    gs = fig.add_gridspec(3, 4)
    gs.update(wspace=0.4, hspace=0.8)

    background_color = "#f6f5f5"
    color_map = ['#eeb977', 'lightgray']
    sns.set_palette(sns.color_palette(color_map))
    column = 'Sex'
    color_map = ['#eeb977', 'lightgray']
    sns.set_palette(sns.color_palette(color_map))
    temp_train = train_df.groupby(column)['Survived'].sum()
    ax2 = fig.add_subplot(gs[0, 2])
    for s in ["right", "top"]:
        ax2.spines[s].set_visible(False)
    ax2.set_facecolor(background_color)
    ax2.tick_params(axis = "y", which = "both", left = False)
    ax2.text(-1, 35, 'Survival Rate for Males and Females', color='black', fontsize=4, ha='left', va='bottom', weight='bold')
    ax2_sns = sns.barplot(ax=ax2, x=temp_train.index, y=temp_train/1000, zorder=2)
    ax2_sns.set_xlabel('')
    ax2_sns.set_ylabel('')
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax2_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE')
    ax2_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE')
    ax2_sns.tick_params(labelsize=5)
    plt.show()





    plt.rcParams['figure.dpi'] = 300
    fig = plt.figure(figsize=(5, 6), facecolor='#f6f5f5')
    gs = fig.add_gridspec(3, 4)
    gs.update(wspace=0.4, hspace=0.8)

    column = 'Pclass'
    color_map = ['#eeb977', 'lightgray', 'lightgray']
    sns.set_palette(sns.color_palette(color_map))
    temp_train = train_df.groupby(column)['Survived'].sum()
    ax1 = fig.add_subplot(gs[0, 1])
    for s in ["right", "top"]:
        ax1.spines[s].set_visible(False)
    ax1.set_facecolor(background_color)
    ax1.tick_params(axis = "y", which = "both", left = False)
    ax1.text(-1, 20, 'Survival Rate for Different Ticket Classes', color='black', fontsize=4, ha='left', va='bottom', weight='bold')
    ax1_sns = sns.barplot(ax=ax1, x=temp_train.index, y=temp_train/1000, zorder=2)
    ax1_sns.set_xlabel("Ticket Class",fontsize=5, weight='bold')
    ax1_sns.set_ylabel('')
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax1_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE')
    ax1_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE')
    ax1_sns.tick_params(labelsize=5)





    plt.rcParams['figure.dpi'] = 300
    fig = plt.figure(figsize=(5, 6), facecolor='#f6f5f5')
    gs = fig.add_gridspec(3, 3)
    gs.update(wspace=0.4, hspace=0.8)


    column = 'Embarked'
    color_map = ['lightgray' for _ in range(4)]
    color_map[3] = '#eeb977'
    sns.set_palette(sns.color_palette(color_map))
    temp_train = train_df.groupby(column)['Survived'].sum()
    ax8 = fig.add_subplot(gs[2, 2])
    for s in ["right", "top"]:
        ax8.spines[s].set_visible(False)
    ax8.set_facecolor(background_color)
    ax8.tick_params(axis = "y", which = "both", left = False)
    ax8.text(-1, 25, 'Survival Rate for Different Embarkmet Ports', color='black', fontsize=4, ha='left', va='bottom', weight='bold')
    ax8_sns = sns.barplot(ax=ax8, x=temp_train.index, y=temp_train/1000, zorder=2)
    ax8_sns.set_xlabel("Port",fontsize=5, weight='bold')
    ax8_sns.set_ylabel('')
    ax8.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax8_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE')
    ax8_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE')
    ax8_sns.tick_params(labelsize=5)




    plt.rcParams['figure.dpi'] = 300
    fig = plt.figure(figsize=(5, 6), facecolor='#f6f5f5')
    gs = fig.add_gridspec(3, 2)
    gs.update(wspace=0.4, hspace=0.8)

    column = 'Cabin_code'
    color_map = ['lightgray' for _ in range(9)]
    color_map[7] = '#eeb977'
    sns.set_palette(sns.color_palette(color_map))
    temp_train = train_df.groupby(column)['Survived'].sum()
    ax7 = fig.add_subplot(gs[1, 1])
    for s in ["right", "top"]:
        ax7.spines[s].set_visible(False)
    ax7.set_facecolor(background_color)
    ax7.tick_params(axis = "y", which = "both", left = False)
    ax7.text(0, 25, 'Survival Rate for Different Cabins', color='black', fontsize=5, ha='left', va='bottom', weight='bold')
    ax7_sns = sns.barplot(ax=ax7, x=temp_train.index, y=temp_train/1000, zorder=2)
    ax7_sns.set_xlabel("Cabin",fontsize=5, weight='bold')
    ax7_sns.set_ylabel('')
    ax7.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax7_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE')
    ax7_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE')
    ax7_sns.tick_params(labelsize=5)





    plt.rcParams['figure.dpi'] = 300
    fig = plt.figure(figsize=(5, 6), facecolor='#f6f5f5')
    gs = fig.add_gridspec(3, 2)
    gs.update(wspace=0.4, hspace=0.8)

    column = 'Fare'
    color_map = ['#eeb977', 'lightgray']
    sns.set_palette(sns.color_palette(color_map))
    temp_train = train_df.groupby(column)['Survived'].sum()
    ax6 = fig.add_subplot(gs[2, 0])
    for s in ["right", "top"]:
        ax6.spines[s].set_visible(False)
    ax6.set_facecolor(background_color)
    ax6.tick_params(axis = "y", which = "both", left = False)
    ax6.text(-2, .037, 'Survival Rate for Fares', color='black', fontsize=6, ha='left', va='bottom', weight='bold')
    ax6_sns = sns.kdeplot(ax=ax6, x=train_df[train_df['Survived']==1]['Fare'], zorder=2, shade=True)
    ax6_sns = sns.kdeplot(ax=ax6, x=train_df[train_df['Survived']==0]['Fare'], zorder=2, shade=True)
    ax6_sns.set_xlabel("Fare",fontsize=5, weight='bold')
    ax6_sns.set_ylabel('')
    ax6_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE')
    ax6_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE')
    ax6_sns.tick_params(labelsize=5)





    plt.rcParams['figure.dpi'] = 300
    fig = plt.figure(figsize=(5, 6), facecolor='#f6f5f5')
    gs = fig.add_gridspec(3, 2)
    gs.update(wspace=0.4, hspace=0.8)

    column = 'Parch'
    color_map = ['lightgray' for _ in range(8)]
    color_map[0] = '#eeb977'
    sns.set_palette(sns.color_palette(color_map))
    temp_train = train_df.groupby(column)['Survived'].sum()
    ax5 = fig.add_subplot(gs[1, 1])
    for s in ["right", "top"]:
        ax5.spines[s].set_visible(False)
    ax5.set_facecolor(background_color)
    ax5.tick_params(axis = "y", which = "both", left = False)
    ax5.text(0, 33, 'Survival Rate for Parch', color='black', fontsize=6, ha='left', va='bottom', weight='bold')
    ax5_sns = sns.barplot(ax=ax5, x=temp_train.index, y=temp_train/1000, zorder=2)
    ax5_sns.set_xlabel("Parents / children",fontsize=5, weight='bold')
    ax5_sns.set_ylabel('')
    ax5.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax5_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE')
    ax5_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE')
    ax5_sns.tick_params(labelsize=5)





    plt.rcParams['figure.dpi'] = 300
    fig = plt.figure(figsize=(5, 6), facecolor='#f6f5f5')
    gs = fig.add_gridspec(3, 2)
    gs.update(wspace=0.4, hspace=0.8)
    column = 'SibSp'
    color_map = ['lightgray' for _ in range(7)]
    color_map[0] = '#eeb977'
    sns.set_palette(sns.color_palette(color_map))
    temp_train = train_df.groupby(column)['Survived'].sum()
    ax4 = fig.add_subplot(gs[1, 1])
    for s in ["right", "top"]:
        ax4.spines[s].set_visible(False)
    ax4.set_facecolor(background_color)
    ax4.tick_params(axis = "y", which = "both", left = False)
    ax4.text(0, 33, 'Survival Rate for SibSp', color='black', fontsize=6, ha='left', va='bottom', weight='bold')
    ax4_sns = sns.barplot(ax=ax4, x=temp_train.index, y=temp_train/1000, zorder=2)
    ax4_sns.set_xlabel("Siblings / spouses",fontsize=5, weight='bold')
    ax4_sns.set_ylabel('')
    ax4.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax4_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE')
    ax4_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE')
    ax4_sns.tick_params(labelsize=5)




    plt.rcParams['figure.dpi'] = 300
    fig = plt.figure(figsize=(5, 6), facecolor='#f6f5f5')
    gs = fig.add_gridspec(3, 2)
    gs.update(wspace=0.4, hspace=0.8)

    column = 'Age'
    color_map = ['#eeb977', 'lightgray']
    sns.set_palette(sns.color_palette(color_map))
    temp_train = train_df.groupby(column)['Survived'].sum()
    ax3 = fig.add_subplot(gs[1, 0])
    for s in ["right", "top"]:
        ax3.spines[s].set_visible(False)
    ax3.set_facecolor(background_color)
    ax3.tick_params(axis = "y", which = "both", left = False)
    ax3.text(-2, .037, 'Survival Rate for Age', color='black', fontsize=6, ha='left', va='bottom', weight='bold')

    ax3_sns = sns.kdeplot(ax=ax3, x=train_df[train_df['Survived']==1]['Age'], zorder=2, shade=True)
    ax3_sns = sns.kdeplot(ax=ax3, x=train_df[train_df['Survived']==0]['Age'], zorder=2, shade=True)
    ax3_sns.set_xlabel("Age",fontsize=5, weight='bold')
    ax3_sns.set_ylabel('')
    ax3_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE')
    ax3_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE')
    ax3_sns.tick_params(labelsize=5)







    plt.rcParams['figure.dpi'] = 300
    fig = plt.figure(figsize=(5, 5), facecolor='#f6f5f5')
    gs = fig.add_gridspec(4, 3)
    gs.update(wspace=0.4, hspace=0.8)

    background_color = "#f6f5f5"

    column = 'Survived'
    color_map = ['#eeb977', 'lightgray']
    sns.set_palette(sns.color_palette(color_map))
    temp_train = pd.DataFrame(train_df[column].value_counts()).reset_index(drop=False)
    ax0 = fig.add_subplot(gs[0, 0])
    for s in ["right", "top"]:
        ax0.spines[s].set_visible(False)
    ax0.set_facecolor(background_color)
    ax0.tick_params(axis = "y", which = "both", left = False)
    ax0.text(-1, 83, 'Survival Rate', color='black', fontsize=7, ha='left', va='bottom', weight='bold')
    ax0.text(-1, 82, 'Survival rate on each individual feature', color='#292929', fontsize=5, ha='left', va='top')
    ax0.text(1.18, 73.3, 'for age and fare', color='#292929', fontsize=4, ha='left', va='top')
    ax0_sns = sns.barplot(ax=ax0, x=temp_train['index'], y=temp_train[column]/1000, zorder=2)
    ax0_sns.set_xlabel("Survived",fontsize=5, weight='bold')
    ax0_sns.set_ylabel('')
    ax0.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax0_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE')
    ax0_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE')
    ax0_sns.tick_params(labelsize=5)
    ax0_sns.legend(['Survived', 'Not Survived'], ncol=2, facecolor=background_color, edgecolor=background_color, fontsize=4, bbox_to_anchor=(-0.26, 1.3), loc='upper left')
    leg = ax0_sns.get_legend()
    leg.legendHandles[0].set_color('#eeb977')
    leg.legendHandles[1].set_color('lightgray')

    column = 'Pclass'
    color_map = ['#eeb977', 'lightgray', 'lightgray']
    sns.set_palette(sns.color_palette(color_map))
    temp_train = train_df.groupby(column)['Survived'].sum()
    ax1 = fig.add_subplot(gs[0, 1])
    for s in ["right", "top"]:
        ax1.spines[s].set_visible(False)
    ax1.set_facecolor(background_color)
    ax1.tick_params(axis = "y", which = "both", left = False)
    ax1_sns = sns.barplot(ax=ax1, x=temp_train.index, y=temp_train/1000, zorder=2)
    ax1_sns.set_xlabel("Ticket Class",fontsize=5, weight='bold')
    ax1_sns.set_ylabel('')
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax1_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE')
    ax1_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE')
    ax1_sns.tick_params(labelsize=5)

    column = 'Sex'
    color_map = ['#eeb977', 'lightgray']
    sns.set_palette(sns.color_palette(color_map))
    temp_train = train_df.groupby(column)['Survived'].sum()
    ax2 = fig.add_subplot(gs[0, 2])
    for s in ["right", "top"]:
        ax2.spines[s].set_visible(False)
    ax2.set_facecolor(background_color)
    ax2.tick_params(axis = "y", which = "both", left = False)
    ax2_sns = sns.barplot(ax=ax2, x=temp_train.index, y=temp_train/1000, zorder=2)
    ax2_sns.set_xlabel("Sex",fontsize=5, weight='bold')
    ax2_sns.set_ylabel('')
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax2_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE')
    ax2_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE')
    ax2_sns.tick_params(labelsize=5)

    column = 'Age'
    color_map = ['#eeb977', 'lightgray']
    sns.set_palette(sns.color_palette(color_map))
    temp_train = train_df.groupby(column)['Survived'].sum()
    ax3 = fig.add_subplot(gs[1, 0])
    for s in ["right", "top"]:
        ax3.spines[s].set_visible(False)
    ax3.set_facecolor(background_color)
    ax3.tick_params(axis = "y", which = "both", left = False)
    ax3_sns = sns.kdeplot(ax=ax3, x=train_df[train_df['Survived']==1]['Age'], zorder=2, shade=True)
    ax3_sns = sns.kdeplot(ax=ax3, x=train_df[train_df['Survived']==0]['Age'], zorder=2, shade=True)
    ax3_sns.set_xlabel("Age",fontsize=5, weight='bold')
    ax3_sns.set_ylabel('')
    ax3_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE')
    ax3_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE')
    ax3_sns.tick_params(labelsize=5)

    column = 'SibSp'
    color_map = ['lightgray' for _ in range(7)]
    color_map[0] = '#eeb977'
    sns.set_palette(sns.color_palette(color_map))
    temp_train = train_df.groupby(column)['Survived'].sum()
    ax4 = fig.add_subplot(gs[1, 1])
    for s in ["right", "top"]:
        ax4.spines[s].set_visible(False)
    ax4.set_facecolor(background_color)
    ax4.tick_params(axis = "y", which = "both", left = False)
    ax4_sns = sns.barplot(ax=ax4, x=temp_train.index, y=temp_train/1000, zorder=2)
    ax4_sns.set_xlabel("Siblings / spouses",fontsize=5, weight='bold')
    ax4_sns.set_ylabel('')
    ax4.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax4_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE')
    ax4_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE')
    ax4_sns.tick_params(labelsize=5)

    column = 'Parch'
    color_map = ['lightgray' for _ in range(8)]
    color_map[0] = '#eeb977'
    sns.set_palette(sns.color_palette(color_map))
    temp_train = train_df.groupby(column)['Survived'].sum()
    ax5 = fig.add_subplot(gs[1, 2])
    for s in ["right", "top"]:
        ax5.spines[s].set_visible(False)
    ax5.set_facecolor(background_color)
    ax5.tick_params(axis = "y", which = "both", left = False)
    ax5_sns = sns.barplot(ax=ax5, x=temp_train.index, y=temp_train/1000, zorder=2)
    ax5_sns.set_xlabel("Parents / children",fontsize=5, weight='bold')
    ax5_sns.set_ylabel('')
    ax5.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax5_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE')
    ax5_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE')
    ax5_sns.tick_params(labelsize=5)

    column = 'Fare'
    color_map = ['#eeb977', 'lightgray']
    sns.set_palette(sns.color_palette(color_map))
    temp_train = train_df.groupby(column)['Survived'].sum()
    ax6 = fig.add_subplot(gs[2, 0])
    for s in ["right", "top"]:
        ax6.spines[s].set_visible(False)
    ax6.set_facecolor(background_color)
    ax6.tick_params(axis = "y", which = "both", left = False)
    ax6_sns = sns.kdeplot(ax=ax6, x=train_df[train_df['Survived']==1]['Fare'], zorder=2, shade=True)
    ax6_sns = sns.kdeplot(ax=ax6, x=train_df[train_df['Survived']==0]['Fare'], zorder=2, shade=True)
    ax6_sns.set_xlabel("Fare",fontsize=5, weight='bold')
    ax6_sns.set_ylabel('')
    ax6_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE')
    ax6_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE')
    ax6_sns.tick_params(labelsize=5)

    column = 'Cabin_code'
    color_map = ['lightgray' for _ in range(9)]
    color_map[7] = '#eeb977'
    sns.set_palette(sns.color_palette(color_map))
    temp_train = train_df.groupby(column)['Survived'].sum()
    ax7 = fig.add_subplot(gs[2, 1])
    for s in ["right", "top"]:
        ax7.spines[s].set_visible(False)
    ax7.set_facecolor(background_color)
    ax7.tick_params(axis = "y", which = "both", left = False)
    ax7_sns = sns.barplot(ax=ax7, x=temp_train.index, y=temp_train/1000, zorder=2)
    ax7_sns.set_xlabel("Cabin",fontsize=5, weight='bold')
    ax7_sns.set_ylabel('')
    ax7.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax7_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE')
    ax7_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE')
    ax7_sns.tick_params(labelsize=5)

    column = 'Embarked'
    color_map = ['lightgray' for _ in range(4)]
    color_map[3] = '#eeb977'
    sns.set_palette(sns.color_palette(color_map))
    temp_train = train_df.groupby(column)['Survived'].sum()
    ax8 = fig.add_subplot(gs[2, 2])
    for s in ["right", "top"]:
        ax8.spines[s].set_visible(False)
    ax8.set_facecolor(background_color)
    ax8.tick_params(axis = "y", which = "both", left = False)
    ax8_sns = sns.barplot(ax=ax8, x=temp_train.index, y=temp_train/1000, zorder=2)
    ax8_sns.set_xlabel("Port",fontsize=5, weight='bold')
    ax8_sns.set_ylabel('')
    ax8.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax8_sns.grid(which='major', axis='x', zorder=0, color='#EEEEEE')
    ax8_sns.grid(which='major', axis='y', zorder=0, color='#EEEEEE')
    ax8_sns.tick_params(labelsize=5)
    
    
    