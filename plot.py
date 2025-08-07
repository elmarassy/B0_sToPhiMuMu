import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
import os

from mpl_toolkits.axes_grid1 import make_axes_locatable

import generation as generation
import jax
import tools
import matplotlib.patches
import run
import matplotlib as mpl

mpl.rc_file('config.rc')
from objdict import ObjDict

from run import cosThetaLRange, cosThetaKRange, phiRange, timeRange, massRange
from run import (
    projectSignalCosThetaK,
    projectSignalCosThetaL,
    projectSignalPhi,
    projectSignalT,
    projectSignalMass,
    projectBackgroundAngles,
    projectBackgroundTime,
    projectBackgroundMass,
)

projectionRanges = [cosThetaLRange, cosThetaKRange, phiRange, timeRange, massRange]
signalProjectionFunctions = [
    projectSignalCosThetaK,
    projectSignalCosThetaL,
    projectSignalPhi,
    projectSignalT,
    projectSignalMass,
]
backgroundProjectionFunctions = [
    projectBackgroundAngles,
    projectBackgroundAngles,
    projectBackgroundAngles,
    projectBackgroundTime,
    projectBackgroundMass,
]

projectionNames = [
    r"$\cos{\theta_k}$",
    r"$\cos{\theta_l}$",
    r"$\phi$",
    "$t$ [ps]",
    r"$m(K^+K^-\mu^+\mu^-)$ $[\mathrm{GeV}/c^2]$",
]
colors = {
    "Pseudodata": 'black',
    "Total": "darkturquoise",
    "Signal (tag $B_s^0$)": "blue",
    r"Signal (tag $\bar{B}_s^0$)": "red",
    "Signal (no tag)": "orchid",
    "Background": "orange",

}

def plotProjection(axis, signalParams, backgroundParams, f, projectionIndex, data, B0proportion=0.5, stacked=False, color=None):
    divider = make_axes_locatable(axis)
    pullAxis = divider.append_axes('bottom', 1.5, pad=0.0)
    untagged = np.where(data[5] + data[6] == 0)
    proportionUntagged = float(len(data[projectionIndex][untagged])) / len(data[projectionIndex])
    proportionB = (1-proportionUntagged) * B0proportion
    proportionBBar = (1-proportionUntagged) * (1 - B0proportion)

    dataBins = 100 if projectionIndex == 4 else 40
    projectionBins = 3000
    dataAxis = np.linspace(*projectionRanges[projectionIndex], dataBins)
    projectionAxis = np.linspace(*projectionRanges[projectionIndex], projectionBins)

    signalB = signalProjectionFunctions[projectionIndex](
        projectionAxis, 1, *signalParams
    )
    signalBBar = signalProjectionFunctions[projectionIndex](
        projectionAxis, -1, *signalParams
    )

    hist, bins = np.histogram(data[projectionIndex], bins=np.linspace(*projectionRanges[projectionIndex], dataBins+1), range=projectionRanges[projectionIndex])

    scale = np.sum(hist) * (bins[1] - bins[0])

    artist = axis.errorbar(
        bins[:-1] + (bins[1]-bins[0])/2, hist, yerr=np.sqrt(hist), fmt=".", color='black', ecolor="black", capsize=3
    )

    for cap in artist[1]:
        cap.set_zorder(10)
    for line in artist[2]:
        line.set_zorder(11)
    background = (1-f)*backgroundProjectionFunctions[projectionIndex](projectionAxis, *backgroundParams) * scale

    un = f*proportionUntagged * scale * (B0proportion*signalB + (1-B0proportion)*signalBBar)
    b = f*signalB * scale * proportionB
    bBar = f*signalBBar * scale * proportionBBar
    if not stacked:
        axis.fill_between(projectionAxis, background, color='orange', linewidth=3)
        axis.plot(projectionAxis, un, color='purple', linewidth=3)
        axis.plot(projectionAxis, b, color="blue", linewidth=3)
        axis.plot(projectionAxis, bBar, color="red", linewidth=3, linestyle=('--' if projectionIndex != 3 else '-'))
        axis.plot(
            projectionAxis,
            un + b + bBar + background,
            color="darkturquoise" if color is None else color,
            )
    else:
        axis.fill_between(projectionAxis, background, color='orange', linewidth=3)
        axis.bar(projectionAxis, background, color='orange', width=(projectionAxis[1] - projectionAxis[0]))
        axis.plot(projectionAxis, un, color='orchid', linewidth=3)
        axis.plot(projectionAxis, b, color="blue", linewidth=3)
        axis.plot(projectionAxis, bBar, color="red", linewidth=3, linestyle=('--' if projectionIndex != 3 else '-'))
        axis.plot(
            projectionAxis,
            un + b + bBar + background,
            color="darkturquoise" if color is None else color,
            )
    conversion = int(projectionBins / dataBins)
    total = un + b + bBar + background
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


def plotProjectionSummary(signalParams, backgroundParams, f, data, dir, index=-1, B0proportion=0.5, tagging='untagged'):
    projDir = os.path.join(dir, "projections")
    exampleDir = os.path.join(projDir, "fitProjectionExamples")
    os.makedirs(projDir, exist_ok=True)
    os.makedirs(exampleDir, exist_ok=True)
    figB = plt.figure(figsize=(32, 18))
    gs = figB.add_gridspec(2, 3, hspace=0.20, wspace=0.2)
    stacked = [True, True, True, True, True]
    pars = generation.getFitParams(True, 2, 'theory')
    for i in range(5):
        axB = figB.add_subplot(gs[i])
        plotProjection(axB, signalParams, backgroundParams, f, i, data, stacked=stacked[i])
        # plotProjection(axB, generation.getAllSignalParamsFromMassless(*pars[0]), pars[1], pars[2], i, data, stacked=stacked[i], color='red')

    legend_axB = figB.add_subplot(gs[1, 2])
    legend_axB.axis("off")
    legend_axB.set_xticks([])
    legend_axB.set_yticks([])

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
        matplotlib.patches.Patch(color=color, label=label) if label == 'Background' else (legend_axB.plot([], [], color=color, label=label, linewidth=5)[0] if label != "Pseudodata"  else legend_axB.errorbar([], [], xerr=None, yerr=[], color='black', fmt='.', label=label))
        for label, color in colors.items() if (tagging != 'untagged' or (label != "Signal (tag $B_s^0$)" and label != r"Signal (tag $\bar{B}_s^0$)"))
    ]
    legend = legend_axB.legend(handles=handles, loc="center", fontsize=44, title=r''f'${lower}'r'\,\mathrm{GeV}^2/\mathrm{c}^4<q^2<'f'{upper}' r'\,\mathrm{GeV}^2/\mathrm{c}^4$' '\n'r'$\mathcal{L}_{\mathrm{int}}='f'{lum}' r'\,\mathrm{fb}^{-1}$''\n', title_fontsize=44)
    plt.setp(legend.get_title(), multialignment='center')

    if index != -1:
        figB.savefig(os.path.join(exampleDir, f"{index}.pdf"))
        plt.close(figB)
    else:
        figB.savefig(os.path.join(projDir, f"summary.pdf"))
        plt.close(figB)
        for ind, name in enumerate(["t", "m"]):
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot()
            plotProjection(ax, signalParams, backgroundParams, f, ind+3, data, stacked=True)
            fig.savefig(os.path.join(projDir, f"{name}.pdf"))
            plt.close(fig)


def plot(names, pulls, values, savedData, nEvents, nToys, dir, q2_range, luminosity, generationName, tagging, errs, deltacov, B0proportion=0.5):
    trueValues = generation.getFitParams(True, q2_range, generationName)
    corr, pullSummary = plotSummary(names, pulls, True, False, dir, q2_range, generationName)
    valSummary = plotSummary(
        names,
        values,
        True,
        True,
        dir,
        q2_range, generationName,
        trueValues[0] + trueValues[1] + [trueValues[2]],
    )
    data = run.generate(
        jax.random.key(0),
        nEvents,
        generation.getAllSignalParamsFromMassless(*trueValues[0]),
        trueValues[1],
        trueValues[2],
        0.5,
        generation.tagging[tagging]["effSS"],
        generation.tagging[tagging]["effOS"],
        generation.tagging[tagging]["wSS"],
        generation.tagging[tagging]["wOS"],
    )
    plotProjectionSummary(
        generation.getAllSignalParamsFromMassless(*trueValues[0]),
        trueValues[1],
        trueValues[2],
        data,
        dir,
        B0proportion=B0proportion,
        tagging=tagging,
    )
    for i, d in enumerate(savedData):
        plotProjectionSummary(generation.getAllSignalParamsFromMassless(*generation.transformBack(*values[i][:-5])), values[i][-5:-1], values[i][-1], savedData[i], dir, index=i, B0proportion=B0proportion, tagging=tagging)
    o = ObjDict()
    o.nToys = nToys
    o.nEvents = nEvents
    o.q2_range = generation.q2_ranges[q2_range]
    o.integratedLuminosity = generation.integratedLuminosities[luminosity]
    o.tagging = {'effSS': generation.tagging[tagging]['effSS'], 'effOS': generation.tagging[tagging]['effOS'],
                 "wSS": generation.tagging[tagging]['wSS'], "wOS": generation.tagging[tagging]['wOS']}
    o.nameOrder = [name.replace("$", "").replace("{", "").replace("}", "").replace("_", "").replace("\\", "").replace("^", "") for name in names]

    o.trueValues = trueValues[0] + trueValues[1] + [trueValues[2]]
    o.fitValues = [valSummary[name] for name in names]
    o.pullValues = [pullSummary[name] for name in names]
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
    fig.tight_layout()
    fig.savefig(os.path.join(dir, f"{title}.pdf"))
    plt.close(fig)


def plotSummary(names, data, massless, values, dir, q2_range, generationName, trueValues=None):
    mpl.rc_file('base.rc')
    if values:
        description = "values"
    else:
        description = "pulls"
        corr = tools.pearsonCorrelation(data)
    newDir = os.path.join(dir, description)
    os.makedirs(newDir, exist_ok=True)
    valSummary = dict()
    pullSummary = dict()
    summaryFig, summaryAxs = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))
    summaryAxs = summaryAxs.flatten()
    nbins = 100
    cols = 4
    if massless:
        cols = 3
    for letterIndex, letter in enumerate(("K", "W", "H", "Z", "background")):
        means = []
        sigmas = []
        letterNames = []
        fig, axs = plt.subplots(nrows=3, ncols=cols, figsize=(12, 8))
        fig.suptitle(f"{letter} {description}")
        fig.subplots_adjust(
            left=0.05, right=0.95, top=0.92, bottom=0.08, wspace=0.3, hspace=0.3
        )
        axs = axs.flatten()
        i = 0
        for index, name in enumerate(names):
            if (
                name[1] == letter
                or (letter == "W" and name == "$A_{CP}$")
                or (
                    letter == "background"
                    and name
                    in [
                        "$f$",
                        "$c_0$",
                        "$c_1$",
                        "$c_2$",
                        "$k_m$",
                        "$m_{B^0_s}$",
                        r"$\sigma_m$",
                    ]
                )
            ):
                ax = axs[i]
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
        generated = generation.getFitParams(True, q2_range, generationName)
        generated = (generation.transform(*generated[0]), generated[1], generated[2])
        if values:
            ax.errorbar(
                x,
                [(generated[0] + generated[1] + [generated[2]])[names.index(name)] for name in letterNames],
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
            for i, name in enumerate(letterNames):
                valSummary[name] = (means[0][i], sigmas[0][i])
            # print(letterNames, means, sigmas)
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
        letterIndex += 1
        fig.savefig(os.path.join(newDir, f"{letter}.pdf"))
        plt.close(fig)
        for i, name in enumerate(letterNames):
            pullSummary[name] = [[means[0][i], means[1][i]], [sigmas[0][i], sigmas[1][i]]]
    summaryFig.suptitle(f"{description.capitalize()} summary")
    summaryFig.tight_layout()

    summaryFig.savefig(os.path.join(newDir, f"summary.pdf"))
    plt.close(summaryFig)

    if not values:
        plotMatrix(names, corr, "pearsonCorrelation", newDir)
        mpl.rc_file('config.rc')
        return corr, pullSummary
    mpl.rc_file('config.rc')
    return valSummary
