fig = plt.figure(1, figsize=(20, 20))

            sharpes = ['ALL', 'Long', 'Short']

            plot_num = 1
            for i in range(8):
                ax = fig.add_subplot(4, 2, plot_num)
                for j in range(len(markets)):
                    ax.plot(train_indiv_sharpes[j, :, i], '-', label=markets[j], c=MYcolors[j])
                    ax.set_title('Train %s sharpe by market' % sharpes[i])
                    plt.axhline(y=0, linestyle='--')
                    plt.axhline(y=0.5, linestyle='--')
                    plt.axhline(y=1, linestyle='--')
                    plt.ylabel('Sharpe')
                    plt.ylim(ymin, ymax)
                if plot_num == 3:
                    plt.legend(loc='upper left', ncol=3)
                plot_num += 1
            # plt.show()
            fig.savefig(resultPath + str(l2Reg) + '/Learning-dynamics-plot/l2Reg-%.3f-run%d-dropout-%.3f.png' %
                        (l2Reg, R, drop_rate), bbox_inches='tight')
            plt.clf()
            plt.close()
###################################################
fig, axes = plt.subplots(2, 10, figsize=(10, 2))
	for i in range(examples_to_show):
		axes[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
		axes[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
	fig.show()
	plt.draw()
	plt.waitforbuttonpress()
###########################################################