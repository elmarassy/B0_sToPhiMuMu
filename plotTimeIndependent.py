import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
import os

from mpl_toolkits.axes_grid1 import make_axes_locatable

import generation as generation
import jax
import tools
from matplotlib.gridspec import GridSpec
import matplotlib.patches
import run
import matplotlib as mpl

mpl.rc_file('config.rc')
from objdict import ObjDict

from run import cosThetaLRange, cosThetaKRange, phiRange, timeRange, massRange
import timeIndependent

projectionRanges = [cosThetaLRange, cosThetaKRange, phiRange, massRange]
signalProjectionFunctions = [
    timeIndependent.projectSignalCosThetaL,
    timeIndependent.projectSignalCosThetaK,
    timeIndependent.projectSignalPhi,
    timeIndependent.projectSignalMass,
]
backgroundProjectionFunctions = [timeIndependent.projectBackgroundAngles,
                                 timeIndependent.projectBackgroundAngles,
                                 timeIndependent.projectBackgroundAngles,
                                 timeIndependent.projectBackgroundMass]

projectionNames = [
    r"$\cos{\theta_l}$",
    r"$\cos{\theta_k}$",
    r"$\phi$",
    r"$m(K^+K^-\mu^+\mu^-)$ $[\mathrm{GeV}/c^2]$",
]
colors = {
    "Pseudodata": 'black',
    "Total": "darkturquoise",
    "Signal": "orchid",
    "Background": "orange",
}


def plotProjection(axis, signalParams, backgroundParams, f, projectionIndex, data):
    divider = make_axes_locatable(axis)
    pullAxis = divider.append_axes('bottom', 1.5, pad=0.0)

    dataBins = 100 if projectionIndex == 4 else 40
    projectionBins = 1000
    dataAxis = np.linspace(*projectionRanges[projectionIndex], dataBins)
    projectionAxis = np.linspace(*projectionRanges[projectionIndex], projectionBins)

    hist, bins = np.histogram(data[projectionIndex], bins=np.linspace(*projectionRanges[projectionIndex], dataBins+1), range=projectionRanges[projectionIndex])
    scale = np.sum(hist) * (bins[1] - bins[0])
    artist = axis.errorbar(
        bins[:-1] + (bins[1]-bins[0])/2, hist, yerr=np.sqrt(hist), fmt=".", color='black', ecolor="black"
    )
    for cap in artist[1]:
        cap.set_zorder(10)
    for line in artist[2]:
        line.set_zorder(11)
    background = (1-f)*backgroundProjectionFunctions[projectionIndex](projectionAxis, *backgroundParams) * scale
    signal = f*signalProjectionFunctions[projectionIndex](projectionAxis, *signalParams) * scale

    axis.fill_between(projectionAxis, background, color='orange', linewidth=3)
    axis.plot(projectionAxis, signal, color='orchid', linewidth=3)
    axis.plot(
        projectionAxis,
        signal + background,
        color="darkturquoise",
        )
    axis.set_ylim(bottom=0)
    axis.set_xlim(*projectionRanges[projectionIndex])
    axis.set_xlabel(projectionNames[projectionIndex])
    units = ["", "", "", "GeV/$c^2$"]
    axis.set_ylabel(fr"Candidates per {(max(projectionRanges[projectionIndex])-min(projectionRanges[projectionIndex]))/dataBins / (np.pi if projectionIndex == 2 else 1):.3f}{r"$\pi$" if projectionIndex == 2 else ""}$\,${units[projectionIndex]}")

    conversion = int(projectionBins / dataBins)
    total = signal + background
    averages = [sum([total[b*conversion + i] for i in range(conversion)]) / conversion for b in range(dataBins)]

    err = np.array([tools.poissonError(b) for b in hist])
    difference = (hist - np.array(averages))
    err = np.where(difference > 0, err[:, 0], err[:, 1])
    pulls = difference/err
    pullAxis.bar(dataAxis, pulls, width=(dataAxis[1]-dataAxis[0]), color='black')
    axis.set_ylim(bottom=0)
    pullAxis.set_ylim(-5, 5)
    pullAxis.set_yticks([-3, 0, 3])

    pullAxis.get_xaxis().set_ticks([tick for tick in axis.get_xticks()])
    axis.set_xticks([tick for tick in axis.get_xticks()])
    axis.set_xticklabels(['' for _ in axis.get_xticks()])
    axis.set_xlim(*projectionRanges[projectionIndex])
    pullAxis.set_xlim(*projectionRanges[projectionIndex])

    pullAxis.set_xlabel(projectionNames[projectionIndex], loc='center')

    units = ["", "", "", "ps", "GeV/$c^2$"]
    axis.set_ylabel(fr"Candidates per {(max(projectionRanges[projectionIndex])-min(projectionRanges[projectionIndex]))/dataBins / (np.pi if projectionIndex == 2 else 1):.3f}{r"$\pi$" if projectionIndex == 2 else ""}$\,${units[projectionIndex]}", loc='center')
#
# def plotProjection(axis, signalParams, backgroundParams, f, projectionIndex, data):
#     dataBins = 100 if projectionIndex == 4 else 40
#     projectionBins = 1000
#     dataAxis = np.linspace(*projectionRanges[projectionIndex], dataBins)
#     projectionAxis = np.linspace(*projectionRanges[projectionIndex], projectionBins)
#
#     hist, bins = np.histogram(data[projectionIndex], bins=np.linspace(*projectionRanges[projectionIndex], dataBins+1), range=projectionRanges[projectionIndex])
#     scale = np.sum(hist) * (bins[1] - bins[0])
#     artist = axis.errorbar(dataAxis + (dataAxis[1]-dataAxis[0])/2, hist, yerr=np.sqrt(hist), xerr=0, fmt=".", color='black', ecolor="black", capsize=3)
#
#     for cap in artist[1]:
#         cap.set_zorder(10)
#     for line in artist[2]:
#         line.set_zorder(11)
#     background = (1-f)*backgroundProjectionFunctions[projectionIndex](projectionAxis, *backgroundParams) * scale
#     signal = f*signalProjectionFunctions[projectionIndex](projectionAxis, *signalParams) * scale
#
#     axis.fill_between(projectionAxis, background, color='orange', linewidth=3)
#     axis.plot(projectionAxis, signal, color='red', linewidth=3)
#     axis.plot(
#         projectionAxis,
#         signal + background,
#         color="darkturquoise",
#         )
#     axis.set_ylim(bottom=0)
#     axis.set_xlim(*projectionRanges[projectionIndex])
#     axis.set_xlabel(projectionNames[projectionIndex])
#     units = ["", "", "", "GeV/$c^2$"]
#     axis.set_ylabel(fr"Candidates per {(max(projectionRanges[projectionIndex])-min(projectionRanges[projectionIndex]))/dataBins / (np.pi if projectionIndex == 2 else 1):.3f}{r"$\pi$" if projectionIndex == 2 else ""}$\,${units[projectionIndex]}")


def plotProjectionSummary(signalParams, backgroundParams, f, data, dir, index=-1):
    projDir = os.path.join(dir, "projections")
    exampleDir = os.path.join(projDir, "fitProjectionExamples")
    os.makedirs(projDir, exist_ok=True)
    os.makedirs(exampleDir, exist_ok=True)
    figB = plt.figure(figsize=(32, 18))
    gs = figB.add_gridspec(2, 3, hspace=0.20, wspace=0.2)
    # figBBar = plt.figure(figsize=(8, 6))
    # gsBBar = GridSpec(2, 3, figure=figBBar)
    stacked = [True, True, True, True]
    for i in range(4):
        axB = figB.add_subplot(gs[i // 3, i % 3])
        # axBBar = figBBar.add_subplot(gsB[i // 3, i % 3])
        plotProjection(axB, signalParams, backgroundParams, f, i, data)

    legend_axB = figB.add_subplot(gs[1, 2])
    legend_axB.axis("off")
    # legend_axBBar = figBBar.add_subplot(gsBBar[1, 2])
    # legend_axBBar.axis("off")
    # handles = [
    #     matplotlib.patches.Patch(color=color, label=label) if label == 'Background' else (legend_axB.plot([], [], color=color, label=label, linewidth=5)[0] if label != "Pseudodata" else legend_axB.errorbar([], [], xerr=None, yerr=[], color='black', fmt='.', label=label))
    #     for label, color in colors.items()
    # ]
    # legend_axB.legend(handles=handles, loc="center", fontsize=44)
    # legend_axBBar.legend(handles=handles, loc="center", fontsize=12, title="Legend")

    q2 = str(dir.split('_')[1])
    if q2 == 'low':
        q2 = generation.q2_ranges[0].split('_')
        lower, upper = float(q2[0]), float(q2[1])
    elif q2 == 'central':
        q2 = generation.q2_ranges[1].split('_')
        lower, upper = float(q2[0]), float(q2[1])
    elif q2 == 'high':
        q2 = generation.q2_ranges[2].split('_')
        lower, upper = float(q2[0]), float(q2[1])

    lum = dir.split('_')[2]

    handles = [
        matplotlib.patches.Patch(color=color, label=label) if label == 'Background' else (legend_axB.plot([], [], color=color, label=label, linewidth=5)[0] if label != "Pseudodata" else legend_axB.errorbar([], [], xerr=None, yerr=[], color='black', fmt='.', label=label))
        for label, color in colors.items()
    ]
    legend = legend_axB.legend(handles=handles, loc="center", fontsize=44, title=r''f'${lower}'r'\,\mathrm{MeV}/\mathrm{c}^2<q^2<'f'{upper}' r'\,\mathrm{MeV}/\mathrm{c}^2$' '\n'r'$\mathcal{L}_{\mathrm{int}}='f'{lum}' r'\,\mathrm{fb}^{-1}$''\n', title_fontsize=44)
    plt.setp(legend.get_title(), multialignment='center')

    # figB.suptitle("$B_s^0$ projections")
    # figBBar.suptitle(r"$\bar{B}_s^0$ projections")
    figB.tight_layout()
    # figBBar.tight_layout()
    if index != -1:
        figB.savefig(os.path.join(exampleDir, f"{index}.pdf"))
        plt.close(figB)
    else:
        figB.savefig(os.path.join(projDir, f"summary.pdf"))
        plt.close(figB)
        figB_m = plt.figure(figsize=(8, 6))
        axM = figB_m.add_subplot(1, 1, 1)
        plotProjection(axM, signalParams, backgroundParams, f, 3, data)
        figB_m.savefig(os.path.join(projDir, f"m.pdf"))
        plt.close(figB_m)


def plot(names, pulls, values, savedData, nEvents, nToys, dir, q2_range, luminosity, generationName):
    trueValues = generation.getFitParams(False, q2_range, generationName)
    corr = plotSummary(names, pulls, False, dir)
    valSummary = plotSummary(
        names,
        values,
        True,
        dir,
        trueValues[0] + trueValues[1] + [trueValues[2]],
        )
    data = timeIndependent.generate(
        jax.random.key(0),
        nEvents,
        *generation.getFitParams(False, q2_range, generationName)
    )
    plotProjectionSummary(
        *generation.getFitParams(False, q2_range, generationName),
        data,
        dir,
    )
    for i, d in enumerate(savedData):
        plotProjectionSummary(*generation.getFitParams(False, q2_range, generationName), savedData[i], dir, i)
    o = ObjDict()
    o.nToys = nToys
    o.nEvents = nEvents
    o.q2_range = generation.q2_ranges[q2_range]
    o.integratedLuminosity = generation.integratedLuminosities[luminosity]

    o.nameOrder = [name.replace("$", "").replace("{", "").replace("}", "").replace("_", "").replace("\\", "").replace("^", "") for name in names]

    o.trueValues = trueValues[0] + trueValues[1] + [trueValues[2]]
    o.fitValues = [valSummary[name] for name in names]
    o.pearsonCorrelation = corr.tolist()


    with open(os.path.join(dir, f"summary.json"), "w") as f:
        json.dump(o, f, indent=4)


def plotMatrix(params, matrix, title, dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, vmin=-1, vmax=1)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation")

    ax.set_xticks(np.arange(len(params)))
    ax.set_yticks(np.arange(len(params)))
    ax.set_xticklabels(params, rotation=45, fontsize=8)
    ax.set_yticklabels(params, fontsize=8)
    for i in range(len(params)):
        for j in range(i, len(params)):
            text = f"{matrix[i, j]:.2f}"
            ax.text(j, i, text, fontsize=4, ha="center", va="center")

    ax.set_title(title)
    ax.grid(False)
    # fig.tight_layout()
    fig.savefig(os.path.join(dir, f"{title}.pdf"))
    plt.close(fig)


def plotSummary(names, data, values, dir, trueValues=None):
    mpl.rc_file('base.rc')
    if values:
        description = "values"
    else:
        description = "pulls"
        corr = tools.pearsonCorrelation(data)
    newDir = os.path.join(dir, description)
    os.makedirs(newDir, exist_ok=True)
    valSummary = dict()
    summaryFig, summaryAx = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
    nbins = 100

    means = []
    sigmas = []
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(12, 8))
    fig.suptitle(f"{description}")
    fig.subplots_adjust(
        left=0.05, right=0.95, top=0.92, bottom=0.08, wspace=0.3, hspace=0.3
    )
    axs = axs.flatten()
    i = 0
    for index, name in enumerate(names):
            ax = axs[index]
            results, errors = tools.fitGaussian(data[:, index])
            if values:
                space = np.linspace(
                    results[0] - 5 * results[1], results[0] + 5 * results[1], nbins
                )
            else:
                space = np.linspace(-5, 5, nbins)
            ax.hist(data[:, index], bins=space, color="cyan")
            ax.plot(
                space,
                tools.gaussDistribution(space, *results)
                * len(data[:, index])
                * (space[1] - space[0]),
                )
            if values:
                ax.axvline(trueValues[index], color="red", linewidth=1)
            ax.set_title(names[index])
            text = (
                rf"$\mu$: ${float(results[0]):.3f}\pm{float(errors[0]):.3f}$"
                "\n"
                rf"$\sigma$: ${float(results[1]):.3f}\pm{float(errors[1]):.3f}$"
            )
            at = AnchoredText(
                text, loc="upper right", prop=dict(size=9), frameon=True
            )
            at.patch.set_boxstyle("round,pad=0.3")
            at.patch.set_facecolor("white")
            at.patch.set_edgecolor("gray")
            at.patch.set_alpha(0.6)
            ax.add_artist(at)
            i += 1
            means.append((results[0], errors[0]))
            sigmas.append((results[1], errors[1]))
    while i < len(axs):
        fig.delaxes(axs[i])
        i += 1
    ax = summaryAx
    x = [i for i in range(len(names))]
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    means = np.array(means).T
    sigmas = np.array(sigmas).T
    if values:
        ax.errorbar(
            x,
            [trueValues[names.index(name)] for name in names],
            label="True values",
            xerr=0.5,
            yerr=0,
            color="red",
            fmt=".",
        )
        ax.errorbar(
            x,
            means[0],
            yerr=sigmas[0],
            label=r"Fit values",
            fmt=".",
            ecolor="black",
            capsize=3,
            color="cyan",
        )
        for i, name in enumerate(names):
            valSummary[name] = (means[0][i], sigmas[0][i])
    else:
        ax.axhline(0, color="gray", linestyle="--")
        ax.axhline(1, color="gray", linestyle="--")
        ax.set_ylim((-0.5, 1.5))
        ax.errorbar(
            x,
            means[0],
            yerr=means[1],
            label=r"$\mu$",
            fmt=".",
            ecolor="black",
            capsize=3,
            color="cyan",
        )
        ax.errorbar(
            x,
            sigmas[0],
            yerr=sigmas[1],
            label=r"$\sigma$",
            fmt=".",
            ecolor="black",
            capsize=3,
            color="red",
        )
    ax.legend()
    fig.savefig(os.path.join(newDir, f"{description}.pdf"))
    plt.close(fig)
    summaryFig.suptitle(f"{description.capitalize()} summary")
    summaryFig.tight_layout()

    summaryFig.savefig(os.path.join(newDir, f"summary.pdf"))
    plt.close(summaryFig)

    if not values:
        plotMatrix(names, corr, "pearsonCorrelation", newDir)
        mpl.rc_file('config.rc')
        return corr
    mpl.rc_file('config.rc')
    return valSummary
