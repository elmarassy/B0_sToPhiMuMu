import json
import generation
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    for generationName in ['theory', 'alternative']:
        for q2_range in range(3):
            for luminosity in range(3):
                res = {}
                for folder in ['badTagging', 'goodTagging', 'untagged']:
                    file = f'{folder}/{generationName}_{generation.q2_range_names[q2_range]}_{generation.integratedLuminosities[luminosity]}/summary.json'
                    with open(file) as f:
                        summary = json.load(f)
                    results = dict(zip(summary['nameOrder'], summary['fitValues']))
                    for r in results:
                        if r not in res:
                            res[r] = [results[r]]
                        else:
                            res[r].append(results[r])

                summaryFig, summaryAxs = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))
                summaryAxs = summaryAxs.flatten()
                letters = ['K', 'W', 'H', 'Z', 'background']
                for index, letter in enumerate(letters):
                    ax = summaryAxs[index]
                    names = []
                    vals = []
                    for i, j in enumerate(res):
                        if j[0] == letter or (j[0] not in letters and letter == 'background'):
                            names.append(j)
                            for b in res[j]:
                                if b[1] < 1e-4:
                                    b[0] = np.NaN
                            vals.append(np.array(res[j]).T)
                    x = np.array([3*i for i in range(len(names))])

                    ax.errorbar(x, [vals[i][0][0] for i in range(len(names))], yerr=[vals[i][1][0] for i in range(len(names))], fmt='.', ecolor='black', color='red', label='Bad tagging')
                    ax.errorbar(x+1, [vals[i][0][1] for i in range(len(names))], yerr=[vals[i][1][1] for i in range(len(names))], fmt='.', ecolor='black', color='blue', label='Good tagging')
                    ax.errorbar(x+2, [vals[i][0][2] for i in range(len(names))], yerr=[vals[i][1][2] for i in range(len(names))], fmt='.', ecolor='black', color='orchid', label='Untagged')

                    ax.set_xticks(x)
                    ax.set_xticklabels(names)
                    # ax.errorbar(x, [vals[i][0][2] for i in range(len(names))], yerr=[vals[i][1][2] for i in range(len(names))], color='red', label='Bad tagging')
                plt.show()
