import matplotlib.pyplot as plt
import numpy as np
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


def plotSummary(names, data, massless, values):
    nbins = 100
    cols = 4
    if massless:
        cols = 3
    for letter in ('K', 'W', 'H', 'Z'):
        fig = plt.figure(figsize=(12, 8))
        i = 1
        for index, name in enumerate(names):
            if name[0] == letter:
                ax = fig.add_subplot(3, cols, i)
                results, errors = tools.fitGaussian(data[:, index])
                if values:
                    space = np.linspace(results[0] - 5*results[1], results[0] + 5*results[1], nbins)
                    # ax.set_xlim(results[0] - 5*results[1], results[0] + 5*results[1])
                else:
                    space = np.linspace(-5, 5, nbins)
                    # ax.set_xlim(-5, 5)
                ax.hist(data[:, index], bins=space, color="cyan")
                ax.plot(space, tools.gaussDistribution(space, *results) * len(data[:, index]) * (space[1] - space[0]))
                if values:
                    ax.axvline(generation.trueValues[name], color='red', linewidth=1)
                ax.set_title(names[index])
                i += 1
        if values:
            description = "values"
        else:
            description = "pulls"
        fig.suptitle(f"{letter} {description} summary")
        fig.tight_layout()

        fig.savefig(f'plots/{description}_{letter}.pdf')
        plt.close(fig)

