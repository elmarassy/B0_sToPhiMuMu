import json
import math

import generation
import matplotlib.pyplot as plt
import numpy as np
#
# if __name__ == "__main__":
#     for generationName in ['theory', 'alternative']:
#         file = f'untagged/theory_low_23/summary.json'
#         with open(file) as f:
#             summary = json.load(f)
#         names = [name for name in summary['nameOrder'] if name[0] == 'H']
#         results = [[[] for _ in range(len(names))],[[] for _ in range(len(names))]]
#
#         for i, tagging in enumerate(['untagged', 'badTagging']):
#             for q2_range in range(1, 3):
#                 for luminosity in range(3):
#                     file = f'{tagging}/{generationName}_{generation.q2_range_names[q2_range]}_{generation.integratedLuminosities[luminosity]}/summary.json'
#                     with open(file) as f:
#                         summary = json.load(f)
#                     for j, param in enumerate(summary['nameOrder']):
#                         if param in names:
#                             results[i][names.index(param)].append(summary['fitValues'][j][1])
#
#         results = np.array(results)
#         summaryFig, summaryAx = plt.subplots(1, 2, figsize=(12, 6))
#
#         configs = [f'{generation.q2_range_names[q2]}_{generation.integratedLuminosities[lum]}' for q2 in range(1, 3) for lum in range(3)]
#         colors = {'23': 'blue', '50': 'red', '300': 'green'}
#
#         untagged = results[0].T
#         tagged = results[1].T
#
#         x = [i for i in range(len(names))]
#         summaryAx[0].set_xticks(x)
#         summaryAx[0].set_xticklabels(names)
#         summaryAx[1].set_xticks(x)
#         summaryAx[1].set_xticklabels(names)
#         for i, config in enumerate(configs):
#             index = 0 if config.split('_')[0] == 'central' else 1
#             summaryAx[index].errorbar([k for k in range(len(names))], untagged[i], fmt='.', markersize=0, xerr=0.4, color=colors[config.split('_')[1]], label=f'untagged_{config}')
#             _, _, artl = summaryAx[index].errorbar([k for k in range(len(names))], tagged[i], fmt='.', markersize=0, xerr=0.4, color=colors[config.split('_')[1]], label=f'tagged_{config}')
#             for line in artl:
#                 line.set_linestyle('--')
#             summaryAx[index].legend()
#
#         # for line in artl:
#         #         line.set_linestyle(styles[config.split('_')[0]])
#         # artp,artC, artl = summaryAx.errorbar([k for k in range(len(names))], tagged[i], fmt='.', xerr=0.4, color=colors[config.split('_')[1]])
#         # for line in artl:
#         #     line.set_linestyle(styles[config.split('_')[0]])
#         summaryAx[0].set_ylim((0, 0.5))
#         summaryAx[0].set_title('Central')
#         summaryAx[1].set_title('High')
#         summaryAx[1].set_ylim((0, 0.5))
#         summaryFig.tight_layout()
#         plt.show()
#
#
#
#         # names = [param for param in results if param[0] == 'H']#[f'{generation.q2_range_names[q2]}_{generation.integratedLuminosities[lum]}' for q2 in range(1, 3) for lum in range(3)]
#         # x = [i for i in range(len(names))]
#         # summaryAx.set_xticks(x)
#         # summaryAx.set_xticklabels(names)
#         #
#         # for i, param in enumerate(names):
#         #     summaryAx.errorbar([i], results[param][0], xerr=0.1, yerr=0, ecolor='black', capsize=0, fmt='.', color='blue')
#         #     summaryAx.plot(x, results[param][1], xerr=0.1, yerr=0, ecolor='black', capsize=0, fmt='.', color='red')
#         #
#         #     plt.show()
#         #     plt.close(summaryFig)
#         # # summaryFig, summaryAxs = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))
#         # # summaryAxs = summaryAxs.flatten()
#         # # letters = ['K', 'W', 'H', 'Z', 'background']
#         # # for index, letter in enumerate(letters):
#         # #     ax = summaryAxs[index]
#         # #     names = []
#         # #     vals = []
#         # #     for i, j in enumerate(res):
#         # #         if j[0] == letter or (j[0] not in letters and letter == 'background'):
#         # #             names.append(j)
#         # #             for b in res[j]:
#         # #                 if b[1] < 1e-4:
#         # #                     b[0] = np.NaN
#         # #             vals.append(np.array(res[j]).T)
#         # #     x = np.array([3*i for i in range(len(names))])
#         #
#         #
#         #     # for folder in ['badTagging', 'goodTagging', 'untagged']:
#         #         #     file = f'{folder}/{generationName}_{generation.q2_range_names[q2_range]}_{generation.integratedLuminosities[luminosity]}/summary.json'
#         #         #     with open(file) as f:
#         #         #         summary = json.load(f)
#         #         #     results = dict(zip(summary['nameOrder'], summary['fitValues']))
#         #         #     for r in results:
#         #         #         if r not in res:
#         #         #             res[r] = [results[r]]
#         #         #         else:
#         #         #             res[r].append(results[r])
#         #
#         #     # summaryFig, summaryAxs = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))
#         #     # summaryAxs = summaryAxs.flatten()
#         #     # letters = ['K', 'W', 'H', 'Z', 'background']
#         #     # for index, letter in enumerate(letters):
#         #     #     ax = summaryAxs[index]
#         #     #     names = []
#         #     #     vals = []
#         #     #     for i, j in enumerate(res):
#         #     #         if j[0] == letter or (j[0] not in letters and letter == 'background'):
#         #     #             names.append(j)
#         #     #             for b in res[j]:
#         #     #                 if b[1] < 1e-4:
#         #     #                     b[0] = np.NaN
#         #     #             vals.append(np.array(res[j]).T)
#         #     #     x = np.array([3*i for i in range(len(names))])
#         #     #
#         #     #     ax.errorbar(x, [vals[i][0][0] for i in range(len(names))], yerr=[vals[i][1][0] for i in range(len(names))], fmt='.', ecolor='black', color='red', label='Bad tagging')
#         #     #     ax.errorbar(x+1, [vals[i][0][1] for i in range(len(names))], yerr=[vals[i][1][1] for i in range(len(names))], fmt='.', ecolor='black', color='blue', label='Good tagging')
#         #     #     ax.errorbar(x+2, [vals[i][0][2] for i in range(len(names))], yerr=[vals[i][1][2] for i in range(len(names))], fmt='.', ecolor='black', color='orchid', label='Untagged')
#         #     #
#         #     #         ax.set_xticks(x)
#         #     #         ax.set_xticklabels(names)
#         #     #         # ax.errorbar(x, [vals[i][0][2] for i in range(len(names))], yerr=[vals[i][1][2] for i in range(len(names))], color='red', label='Bad tagging')
#         #     #     plt.show()
import numpy as np

def getCorrMatrix(matrix, indices):
    return matrix[indices][:,indices]


if __name__ == '__main__':
    import tools
    import run
    import uncertainties
    from uncertainties import umath

    coshIntegral, sinhIntegral, cosIntegral, sinIntegral = tools.getIntegrals(run.acceptance, run.timeRange, run.gamma, run.x, run.y)
    for lum in ['23', '50', '300']:
        file = f'badTagging/theory_high_{lum}'
        with open(f'{file}/summary.json', 'r') as f:
            summary = json.load(f)
            nameOrder = summary['nameOrder']
            pullValues = summary['pullValues']
            fitValues = summary['fitValues']
            trueValues = summary['trueValues']
            corr = np.array(summary['pearsonCorrelation'])
        indexH3 = nameOrder.index('H3')
        indexH9 = nameOrder.index('H9')
        indexK3 = nameOrder.index('K3')
        indexK9 = nameOrder.index('K9')
        corr = getCorrMatrix(corr, np.array([indexH3, indexH9, indexK3, indexK9]))



        H3, dH3 = fitValues[indexH3]
        H9, dH9 = fitValues[indexH9]
        K3, dK3 = fitValues[indexK3]
        K9, dK9 = fitValues[indexK9]
        (h3, h9, k3, k9) = uncertainties.correlated_values_norm([(H3, dH3), (H9, dH9), (K3, dK3), (K9, dK9)], corr)

        A = k9*float(coshIntegral) - h9*float(sinhIntegral)
        B = k3*float(coshIntegral) - h3*float(sinhIntegral)

        delta = umath.atan((B/A))
        # dDelta = ((dK3*coshIntegral/A)**2 + (dH3*sinhIntegral/A)**2 + (dK9*coshIntegral*B/A**2)**2 + (dH9*sinhIntegral*B/A**2)**2)**0.5
        print(f"{A}.6f")
        print(f"{B}.6f")
        print(f"{delta:.6f}")
        print(f"{(A**2 + B**2)**0.5:.6f}")
        H3 = trueValues[indexH3]
        H9 = trueValues[indexH9]
        K3 = trueValues[indexK3]
        K9 = trueValues[indexK9]
        ATrue = K9*float(coshIntegral) - H9*float(sinhIntegral)
        BTrue = K3*float(coshIntegral) - H3*float(sinhIntegral)
        deltaTrue = math.atan((BTrue/ATrue))
        print(deltaTrue)
        print(delta.nominal_value)
        print(f'pull: {(delta.nominal_value - deltaTrue)/delta.std_dev}')

        print("...........")





#
#
# if __name__ == '__main__':
#     letter = 'Z'
#     names = [name for name in generation.getFitParamNames(True) if letter in name]
#     jsonNames = [name.replace("$", "").replace("{", "").replace("}", "").replace("_", "").replace("\\", "").replace("^", "") for name in names]
#     colors = {'badTagging': 'red', 'goodTagging': 'blue', 'untagged': 'orchid'}
#     q2_names = {'low': (0.10, 0.98), 'central': (1.10, 6.00), 'high': (15.00, 19.00)}
#     summaryPlot, summaryAxs = plt.subplots(3, 3, figsize=(14, 9))
#     summaryAxs = summaryAxs.flatten()
#     i = 0
#     for q2_range in ['low', 'central', 'high']:
#         for lum in [23, 50, 300]:
#             ax = summaryAxs[i]
#             x = np.array([3*i for i in range(len(jsonNames))])
#             ax.set_xticks(x+1)
#             ax.set_xticklabels(jsonNames)
#             for j, tagging in enumerate(['untagged', 'badTagging', 'goodTagging']):
#                 file = f'{tagging}/theory_{q2_range}_{lum}/summary.json'
#                 with open(file) as f:
#                     summary = json.load(f)
#                     nameOrder = summary['nameOrder']
#                     pullValues = summary['pullValues']
#                     pulls = np.array([pullValues[nameOrder.index(name)] for name in jsonNames])
#                     # (pulls[:,0][:, 0])[pulls[:,0][:, 0] == 0] = -10
#                     means = pulls[:,0][:, 0]
#                     sigmas = pulls[:,1][:, 0]
#                     for k in range(len(means)):
#                         if means[k] == 0:
#                             means[k] = -10
#                             sigmas[k] = -10
#                 ax.errorbar(x+j, means, yerr=pulls[:,0][:, 1], fmt='.', markersize=4, capsize=2, ecolor='black', color=colors[tagging])
#                 ax.errorbar(x+j, sigmas, yerr=pulls[:,1][:, 1], fmt='.', markersize=4, capsize=2, ecolor='black', color=colors[tagging])
#                 ax.axhline(0, linestyle='--', color='gray', linewidth=1)
#                 ax.axhline(1, linestyle='--', color='gray', linewidth=1)
#                 ax.legend(loc='center', title=r''f'${q2_names[q2_range][0]}'r'\,\mathrm{GeV}^2/\mathrm{c}^4<q^2<'f'{q2_names[q2_range][0]}' r'\,\mathrm{GeV}^2/\mathrm{c}^4$' r',$\quad$'r'$\mathcal{L}_{\mathrm{int}}='f'{lum}' r'\,\mathrm{fb}^{-1}$''\n')
#                 ax.set_ylim(-0.25, 1.25)
#
#             i += 1
#     summaryPlot.tight_layout()
#     plt.savefig(f'pullSummary{letter}.pdf')
#
#
