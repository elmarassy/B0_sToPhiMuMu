import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
import os
import generation as generation
import jax
import tools
from matplotlib.gridspec import GridSpec
import matplotlib.patches
import run

from run import cosThetaLRange, cosThetaKRange, phiRange, timeRange, massRange
from run import projectSignalCosThetaL, projectSignalCosThetaK, projectSignalPhi, projectSignalT, projectSignalMass, projectBackgroundAngles, projectBackgroundTime, projectBackgroundMass

projectionRanges = [cosThetaLRange, cosThetaKRange, phiRange, timeRange, massRange]
signalProjectionFunctions = [projectSignalCosThetaL, projectSignalCosThetaK, projectSignalPhi, projectSignalT, projectSignalMass]
backgroundProjectionFunctions = [projectBackgroundAngles, projectBackgroundAngles, projectBackgroundAngles, projectBackgroundTime, projectBackgroundMass]

projectionNames = [r"$\cos{\theta_l}$", r"$\cos{\theta_k}$", r"$\phi$", "$t$", "$m_{B_s^0}$"]
colors = {'Both': 'green', 'SS': 'yellow', 'OS': 'orange', 'Untagged': 'red', 'Fit': 'blue'}
#
# wSS = generation.trueValues['wSS']
# wOS = generation.trueValues['wOS']
# effSS = generation.trueValues['effSS']
# effOS = generation.trueValues['effOS']
#
# bothWeight = (1. - wSS) * (1. - wOS)
# ssWeight = (1. - wSS) / 2
# osWeight = (1. - wOS) / 2
# untaggedWeight = 1. / 4
#
# cmap = plt.get_cmap('viridis')
#
# colors = [cmap(i) for i in np.linspace(0, 1, 4)]


def plotGeneration(axis, signalParams, backgroundParams, f, sign, projectionIndex, stack=False):

    projectionAxis = np.linspace(*projectionRanges[projectionIndex], 1000)
    width = projectionAxis[1] - projectionAxis[0]
    signal = f*signalProjectionFunctions[projectionIndex](projectionAxis, sign, *signalParams)
    oppositeSignal = f*signalProjectionFunctions[projectionIndex](projectionAxis, -sign, *signalParams)

    both = signal*effSS*effOS*bothWeight
    ss = signal*effSS*ssWeight
    os = signal*effOS*osWeight
    untagged = signal*untaggedWeight
    scale = np.sum(both + ss + os + untagged) * width

    background = (1-f)*backgroundProjectionFunctions[projectionIndex](projectionAxis, *backgroundParams)*scale
    base = 0
    if stack:
        base = background
        ax.bar(projectionAxis, background, width=width, color='tomato')

    ax.bar(projectionAxis, untagged, bottom=base, width=width, color=colors[0])
    ax.bar(projectionAxis, ss, bottom=base + untagged, width=width, color=colors[1])
    ax.bar(projectionAxis, os, bottom=base + untagged + ss, width=width, color=colors[2])
    ax.bar(projectionAxis, both, bottom=base + untagged + ss + os, width=width, color=colors[3])

    signal *= scale/f
    if not stack:
        axis.plot(projectionAxis, signal, color='red')
    axis.plot(projectionAxis, background, linestyle='--', color='orange')
    axis.plot(projectionAxis, signal + background, color='black', linestyle='--')
    axis.set_ylim(bottom=0)
#
#
# def plotProjection(axis, signalParams, backgroundParams, f, sign, projectionIndex, data, both, ss, os, untagged):
#     dataBins = 100
#     # both = data[both]
#     # ss = data[ss]
#     # os = data[os]
#     # untagged = data[untagged]
#     # total = axis.hist([both, ss, os, untagged], bins=dataBins, range=projectionRanges[projectionIndex],
#     #                       weights=[np.full_like(both, bothWeight), np.full_like(ss, ssWeight), np.full_like(os, osWeight), np.full_like(untagged, untaggedWeight)])[0]
#     # both = np.histogram(data[both], bins=dataBins, range=projectionRanges[projectionIndex])[0]
#     # ss = np.histogram(data[ss], bins=dataBins, range=projectionRanges[projectionIndex])[0]
#     # os = np.histogram(data[os], bins=dataBins, range=projectionRanges[projectionIndex])[0]
#     # untagged = np.histogram(data[untagged], bins=dataBins, range=projectionRanges[projectionIndex])[0]
#     # total = both*bothWeight + ss*ssWeight + os*osWeight + untagged*untaggedWeight
#
#
#     projectionAxis = np.linspace(*projectionRanges[projectionIndex], 1000)
#     dataAxis = np.linspace(*projectionRanges[projectionIndex], dataBins)
#     scale = np.sum(total) * (dataAxis[1] - dataAxis[0])
#
#     axis.errorbar(dataAxis, total, yerr=np.sqrt(total), fmt='.', ecolor='black', capsize=3)
#     signal = f*signalProjectionFunctions[projectionIndex](projectionAxis, sign, *signalParams) * scale
#     background = (1-f)*backgroundProjectionFunctions[projectionIndex](projectionAxis, *backgroundParams) * scale
#     axis.plot(projectionAxis, signal)
#     axis.plot(projectionAxis, background, linestyle='--')
#     axis.plot(projectionAxis, signal + background)
#     axis.set_ylim(bottom=0)


#
# params = generation.getFitParams(True)
# p = generation.getAllSignalParamsFromMassless(*params[0]), params[1], params[2]
#
# for i in range(5):
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     b = True
#     if i == 4:
#         b = False
#     plotGeneration(ax, *p, 1, i, b)
#     ax.set_title(projectionNames[i])
#     plt.show()
#     plt.close(fig)
#
#
# raise Exception





def plotProjection(axB, axBBar, signalParams, backgroundParams, f, index, data):
    ssB = data[5] == 1
    osB = data[6] == 1
    ssBBar = data[5] == -1
    osBBar = data[6] == -1
    ssUntagged = data[5] == 0
    osUntagged = data[6] == 0
    wSS = data[7][0]
    wOS = data[8][0]
    bothBMask = np.logical_and(ssB, osB)
    osBMask = np.logical_and(osB, ssUntagged)
    ssBMask = np.logical_and(ssB, osUntagged)
    bothBBarMask = np.logical_and(ssBBar, osBBar)
    osBBarMask = np.logical_and(osBBar, ssUntagged)
    ssBBarMask = np.logical_and(ssBBar, osUntagged)
    untaggedMask = np.logical_and(ssUntagged, osUntagged)

    space = np.linspace(*projectionRanges[index], 1000)
    dataSpace = np.linspace(*projectionRanges[index], 100)

    currentData = data[index]

    bothB = currentData[bothBMask]
    osB = currentData[osBMask]
    ssB = currentData[ssBMask]

    bothBBar = currentData[bothBBarMask]
    osBBar = currentData[osBBarMask]
    ssBBar = currentData[ssBBarMask]

    bothUntagged = currentData[untaggedMask]

    weightB = np.full_like(bothB, (1 - wSS)*(1-wOS))
    weightSSB = np.full_like(ssB, (1 - wSS)/2)
    weightOSB = np.full_like(osB, (1-wOS)/2)
    weightUntagged = np.full_like(bothUntagged, 0.25)
    weightBBar = np.full_like(bothBBar, (1 - wSS)*(1-wOS))
    weightSSBBar = np.full_like(ssBBar, (1 - wSS)/2)
    weightOSBBar = np.full_like(osBBar, (1-wOS)/2)

    counts, bins, _ = axB.hist([bothB, ssB, osB, bothUntagged], bins=dataSpace, color=['green', 'yellow', 'orange', 'red'], stacked=True,
                               weights=[weightB, weightSSB, weightOSB, weightUntagged],
                               label=['Both', 'SS', 'OS', 'Untagged'])
    scaleB = np.sum(counts[-1]) * (bins[1] - bins[0])
    bSignalProj = signalProjectionFunctions[index](space, 1, *signalParams)
    bBackgroundProj = backgroundProjectionFunctions[index](space, *backgroundParams)
    bProj = f*bSignalProj + (1-f)*bBackgroundProj

    axB.plot(space, bProj * scaleB, label='Distribution', color='blue')
    counts, bins, _ = axBBar.hist([bothBBar, ssBBar, osBBar, bothUntagged], bins=dataSpace, color=['green', 'yellow', 'orange', 'red'], stacked=True,
                                  weights=[weightBBar, weightSSBBar, weightOSBBar, weightUntagged],
                                  label=['Both', 'SS', 'OS', 'Untagged'])
    scaleBBar = np.sum(counts[-1]) * (bins[1] - bins[0])
    bBarSignalProj = signalProjectionFunctions[index](space, -1, *signalParams)
    bBarBackgroundProj = backgroundProjectionFunctions[index](space, *backgroundParams)
    bBarProj = f*bBarSignalProj + (1-f)*bBarBackgroundProj

    axBBar.plot(space, bBarProj * scaleBBar, label='Fit', color='blue')
    axB.set_title(projectionNames[index])
    axBBar.set_title(projectionNames[index])
    axB.set_ylim(bottom=0)
    axBBar.set_ylim(bottom=0)


def plotProjectionSummary(signalParams, backgroundParams, f, data, dir, index=-1):
    projDir = os.path.join(dir, 'projections')
    largePullDir = os.path.join(projDir, 'largePulls')

    figB = plt.figure(figsize=(8, 6))
    gsB = GridSpec(2, 3, figure=figB)
    figBBar = plt.figure(figsize=(8, 6))
    gsBBar = GridSpec(2, 3, figure=figBBar)
    for i in range(5):
        axB = figB.add_subplot(gsB[i // 3, i % 3])
        axBBar = figBBar.add_subplot(gsB[i // 3, i % 3])
        plotProjection(axB, axBBar, signalParams, backgroundParams, f, i, data)

    legend_axB = figB.add_subplot(gsB[1, 2])
    legend_axB.axis('off')
    legend_axBBar = figBBar.add_subplot(gsBBar[1, 2])
    legend_axBBar.axis('off')
    handles = [matplotlib.patches.Patch(color=color, label=label) for label, color in colors.items()]
    legend_axB.legend(handles=handles, loc='center', fontsize=12, title="Legend")
    legend_axBBar.legend(handles=handles, loc='center', fontsize=12, title="Legend")

    figB.suptitle("$B_s^0$ projections")
    figBBar.suptitle(r"$\bar{B}_s^0$ projections")
    figB.tight_layout()
    figBBar.tight_layout()
    if index != -1:
        currentDir = os.path.join(largePullDir, f'{index}')
        os.makedirs(currentDir, exist_ok=True)
        figB.savefig(os.path.join(currentDir, "B.pdf"))
        figBBar.savefig(os.path.join(currentDir, "BBar.pdf"))
        plt.close(figB)
        plt.close(figBBar)
        return currentDir
    else:
        figB.savefig(os.path.join(projDir, f"B.pdf"))
        figBBar.savefig(os.path.join(projDir, f"BBar.pdf"))
        plt.close(figB)
        plt.close(figBBar)
        figB_t = plt.figure(figsize=(8, 6))
        figBBar_t = plt.figure(figsize=(8, 6))
        axT = figB_t.add_subplot(1, 1, 1)
        axTBar = figBBar_t.add_subplot(1, 1, 1)
        plotProjection(axT, axTBar, signalParams, backgroundParams, f, 3, data)
        figB_t.savefig(os.path.join(projDir, f"B_t.pdf"))
        figBBar_t.savefig(os.path.join(projDir, f"BBar_t.pdf"))
        plt.close(figB_t)
        plt.close(figBBar_t)
        figB_m = plt.figure(figsize=(8, 6))
        figBBar_m = plt.figure(figsize=(8, 6))
        axM = figB_m.add_subplot(1, 1, 1)
        axMBar = figBBar_m.add_subplot(1, 1, 1)
        plotProjection(axM, axMBar, signalParams, backgroundParams, f, 4, data)
        figB_m.savefig(os.path.join(projDir, f"B_m.pdf"))
        figBBar_m.savefig(os.path.join(projDir, f"BBar_m.pdf"))
        plt.close(figB_m)
        plt.close(figBBar_m)

def plot(names, pulls, values, savedData, dir, massless=True):
    trueValues = generation.getFitParams(massless)
    plotSummary(names, pulls, massless, False, dir)
    plotSummary(names, values, massless, True, dir, trueValues[0] + trueValues[1] + [trueValues[2]])
    data = run.generate(jax.random.key(0), 100000, generation.getAllSignalParamsFromMassless(*trueValues[0]),
                           trueValues[1], trueValues[2], 0.5, generation.trueValues['effSS'],
                           generation.trueValues['effOS'], generation.trueValues['wSS'], generation.trueValues['wOS'])
    plotProjectionSummary(generation.getAllSignalParamsFromMassless(*trueValues[0]),
                          trueValues[1], trueValues[2], data, dir)

def plotMatrix(params, matrix, title, dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, vmin=-1, vmax=1)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation')

    ax.set_xticks(np.arange(len(params)))
    ax.set_yticks(np.arange(len(params)))
    ax.set_xticklabels(params, rotation=45, fontsize=8)
    ax.set_yticklabels(params, fontsize=8)
    for i in range(len(params)):
        for j in range(i, len(params)):
            text = f"{matrix[i, j]:.2f}"
            ax.text(j, i, text, fontsize=4, ha='center', va='center')

    ax.set_title(title)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(dir, f'{title}.pdf'))
    plt.close(fig)


def plotSummary(names, data, massless, values, dir, trueValues=None):
    if values:
        description = "values"
    else:
        description = "pulls"
        corr = tools.pearsonCorrelation(data)
    newDir = os.path.join(dir, description)
    os.makedirs(newDir, exist_ok=True)

    summaryFig, summaryAxs = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))
    summaryAxs = summaryAxs.flatten()
    nbins = 100
    cols = 4
    if massless:
        cols = 3
    for letterIndex, letter in enumerate(('K', 'W', 'H', 'Z', 'background')):
        means = []
        sigmas = []
        letterNames = []
        fig, axs = plt.subplots(nrows=3, ncols=cols, figsize=(12, 8))
        fig.suptitle(f"{letter} {description}")
        fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, wspace=0.3, hspace=0.3)
        axs = axs.flatten()
        i = 0
        for index, name in enumerate(names):
            if name[1] == letter or (letter == 'W' and name == '$A_{CP}$') or (letter == 'background' and name in ["$f$", "$c_0$", "$c_1$", "$c_2$", "$k_m$", "$m_{B^0_s}$", r"$\sigma_m$"]):
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
        fig.savefig(os.path.join(newDir, f'{letter}.pdf'))
        # fig.savefig(f'plots/{description}_{letter}.pdf')
        plt.close(fig)
        summaryFig.suptitle(f"{description.capitalize()} summary")
        summaryFig.tight_layout()

        # summaryFig.savefig(f'plots/summary.pdf')
        summaryFig.savefig(os.path.join(newDir, f'summary.pdf'))
        plt.close(summaryFig)

        if not values:
            plotMatrix(names, corr, 'pearsonCorrelation', newDir)





