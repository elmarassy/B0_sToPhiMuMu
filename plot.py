import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText

import taggedTimeDependent
import generation
import tools


def project(saveProjectionsValue, massless, pulls, values):
    saveIndices = np.where((np.abs(pulls) > saveProjectionsValue).any(axis=1))[0]
    projectionList = [taggedTimeDependent.projectCosThetaL, taggedTimeDependent.projectCosThetaK,
                      taggedTimeDependent.projectPhi, taggedTimeDependent.projectT]
    projectionRanges = [(-1.0, 1.0), (-1.0, 1.0), (-np.pi, np.pi), (0.0, 10.0)]
    projectionNames = [r"$\cos{\theta_l})$", r"$\cos{\theta_k}$", r"$\phi$", "$t$"]

    def plot(params, index):
        figB = plt.figure(figsize=(8, 6))
        figBBar = plt.figure(figsize=(8, 6))
        for i in range(4):
            axB = figB.add_subplot(2, 2, i+1)
            axBBar = figBBar.add_subplot(2, 2, i+1)
            space = np.linspace(*projectionRanges[i], 1000)
            axB.plot(space, projectionList[i](1, params, space))
            axBBar.plot(space, projectionList[i](-1, params, space))
            axB.set_title(projectionNames[i])
            axBBar.set_title(projectionNames[i])
            axB.set_ylim(bottom=0)
            axBBar.set_ylim(bottom=0)
        figB.suptitle("$B_0$ projections")
        figBBar.suptitle(r"$\bar{B}_0$ projections")
        figB.tight_layout()
        figBBar.tight_layout()

        figB.savefig(f'plots/projections/B_{index}.pdf')
        figBBar.savefig(f'plots/projections/BBar_{index}.pdf')
        plt.close(figB)
        plt.close(figBBar)

    for index in saveIndices:
        params = generation.getAllParamsFromMassless(*values[index])
        plot(params, index)


def plotSummary(names, data, massless, values, trueValues=None):
    if values:
        description = "values"
    else:
        description = "pulls"
    summaryFig, summaryAxs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    summaryAxs = summaryAxs.flatten()
    nbins = 100
    cols = 4
    if massless:
        cols = 3
    for letterIndex, letter in enumerate(('K', 'W', 'H', 'Z')):
        means = []
        sigmas = []
        letterNames = []
        fig, axs = plt.subplots(nrows=3, ncols=cols, figsize=(12, 8))
        fig.suptitle(f"{letter} {description}")
        fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, wspace=0.3, hspace=0.3)
        axs = axs.flatten()
        i = 0
        for index, name in enumerate(names):
            if name[0] == letter:
                ax = axs[i]
                results, errors = tools.fitGaussian(data[:, index])
                if values:
                    space = np.linspace(results[0] - 5*results[1], results[0] + 5*results[1], nbins)
                else:
                    space = np.linspace(-5, 5, nbins)
                ax.hist(data[:, index], bins=space, color="cyan")
                ax.plot(space, tools.gaussDistribution(space, *results) * len(data[:, index]) * (space[1] - space[0]))
                if values:
                    ax.axvline(trueValues[index], color='red', linewidth=1)
                ax.set_title(names[index])
                text = (
                    fr"$\mu$: ${float(results[0]):.3f}\pm{float(errors[0]):.3f}$" "\n"
                    fr"$\sigma$: ${float(results[1]):.3f}\pm{float(errors[1]):.3f}$"
                )
                at = AnchoredText(
                    text,
                    loc='upper right',
                    prop=dict(size=9),
                    frameon=True
                )
                at.patch.set_boxstyle("round,pad=0.3")
                at.patch.set_facecolor("white")
                at.patch.set_edgecolor("gray")
                at.patch.set_alpha(0.6)
                ax.add_artist(at)
                i += 1
                means.append((results[0], errors[0]))
                sigmas.append((results[1], errors[1]))
                letterNames.append(name)
        while i < len(axs):
            fig.delaxes(axs[i])
            i += 1
        ax = summaryAxs[letterIndex]
        x = [i for i in range(len(letterNames))]
        ax.set_xticks(x)
        ax.set_xticklabels(letterNames)
        means = np.array(means).T
        sigmas = np.array(sigmas).T
        if values:
            ax.errorbar(x, [generation.trueValues[name] for name in letterNames], label='True values', xerr=0.5, yerr=0, color='red', fmt='.')
            ax.errorbar(x, means[0], yerr=sigmas[0], label=r'Fit values', fmt='.', ecolor='black', capsize=3, color='cyan')
        else:
            ax.axhline(0, color='gray', linestyle='--')
            ax.axhline(1, color='gray', linestyle='--')
            ax.set_ylim((-0.5, 1.5))
            ax.errorbar(x, means[0], yerr=means[1], label=r"$\mu$", fmt='.', ecolor='black', capsize=3, color='cyan')
            ax.errorbar(x, sigmas[0], yerr=sigmas[1], label=r"$\sigma$", fmt='.', ecolor='black', capsize=3, color='red')
        ax.legend()
        letterIndex += 1

        fig.savefig(f'plots/{description}_{letter}.pdf')
        plt.close(fig)
        summaryFig.suptitle(f"{description.capitalize()} summary")
        summaryFig.tight_layout()

        summaryFig.savefig(f'plots/{description}Summary.pdf')
        plt.close(summaryFig)





