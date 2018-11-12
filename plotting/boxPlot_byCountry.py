import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
'''
boxPlot of probabilities by class label!!!
'''
results_dir_dict = ['logisticRegression/', 'MLP-1Branch-1layer/', 'MLP-1Branch/', 'MLP-2Branch/']
model_folder = results_dir_dict[3]
test_data = pd.read_csv('C:/behrouz/projects/data/O365_Business_Premium_solo_2017-06-28/' + 'tlc_test_data.csv')[['OMSTenantId', 'CountryCode']]
res = pd.read_csv('C:/behrouz/projects/businessPremiumSolo/'+model_folder+'/results/test_imbalanced_batch.csv')
res = pd.merge(res, test_data, on='OMSTenantId', how='inner')
print(res.shape)
country = sorted(list(set(res['CountryCode'].values)))
print(country)
plot_titles = country
prob_list_pos = []
prob_list_neg = []
print(res.info())
for cntry in country:
    res1 = res.loc[res['CountryCode'] == cntry]
    print(res1.shape)
    pos = res1['Probs'].loc[res1['outHasActiveO365'] == 1.]
    print(res1.shape[0], len(pos) / res1.shape[0])
    neg = res1['Probs'].loc[res1['outHasActiveO365'] == 0.]
    prob_list_pos.append(pos.values)
    prob_list_neg.append(neg.values)

fig = plt.figure(1, figsize=(18, 14))
prob_list = [prob_list_pos, prob_list_neg]
box_fill_colors = ['lightcoral', 'cadetblue']
plot_titles = ['Business Premium Solo-P(y=active), Active class',
               'Business Premium Solo-P(y=active)- Not Active class']
for i in range(2):
    ax = fig.add_subplot(2, 1, i+1)
    bp_pos = ax.boxplot(prob_list[i], whis=[5, 95], patch_artist=True, meanline=True, showmeans=True)
    ax.set_xticklabels(country, rotation=45)

    # change outline color, fill color and linewidth of the boxes
    for box in bp_pos['boxes']:
        box.set(color='#7570b3', linewidth=2)
        # change fill color
        box.set(facecolor=box_fill_colors[i])
    # change color and linewidth of the whiskers
    for whisker in bp_pos['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)

    # change color and linewidth of the caps
    for cap in bp_pos['caps']:
        cap.set(color='#7570b3', linewidth=2)

    # change color and linewidth of the medians
    for median in bp_pos['medians']:
        median.set(color='#b2df8a', linewidth=2)

    # change color and linewidth of the means
    for median in bp_pos['means']:
        median.set(color='black', linewidth=2)

    # change the style of fliers and their fill
    for flier in bp_pos['fliers']:
        flier.set(marker='.', color='red', alpha=0.05)

    # Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.title(plot_titles[i])
    plt.ylabel('Probability')

    for cl in [('black', 'Mean', ':'), ('#7570b3', 'Whiskers-5&95 Quantiles', '-'), ('#b2df8a', 'Median', '-')]:
        plt.plot([], [], cl[2], color=cl[0], label=cl[1], lw=3)

    ax.legend(loc='lower left')
    # plt.subplots_adjust(left=None, bottom=0.3, right=None, top=0.9,
    #                     wspace=None, hspace=None)

plt.show()
path = 'C:/behrouz/projects/businessPremiumSolo/'

# fig.savefig(path + '/BoxPlot-PredByCountry-BusinessPremiumSolo-06-28-2017'+model_folder[:-1]+'.png', bbox_inches='tight')
plt.clf()
plt.close()



