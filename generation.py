#
# generationValues = [("x", 26.93), ("y", 0.124),
#     ("K1c", 0.8661557825167852), ("K2s", 0.04906538817313792),
#     ("K2c", -0.8392845161954529), ("K3", -0.014487296919474972),
#     ("K4", -0.1453399803705659), ("K5", -0.1662968116777378),
#     ("K6s", 0.006879902115106718), ("K7", -0.021579262269771426),
#     ("K8", -0.006834401112133996), ("K9", -0.000917567971296945),
#
#     ("W1s", 2.89030033977211e-05), ("W1c", 0.0009074215263274517),
#     ("W2s", 8.53322844894951e-06), ("W2c", -0.0008821425176404872),
#     ("W3", 2.2017298457314245e-05), ("W4", 0.00017483191366255714),
#     ("W5", 0.000262027717737243), ("W6s", -0.0004842899129342597),
#     ("W7", 0.0020984263762797157), ("W8", 0.0009964600073610955),
#     ("W9", 0.00012160250786178678),
#
#     ("H1s", 0.021591681620283635), ("H1c", 0.863927164713325),
#     ("H2s", 0.007229344144145437), ("H2c", -0.8371254802199508),
#     ("H3", -0.09784412774836446), ("H4", -0.14503651133194273),
#     ("H5", 0.002348195427913586), ("H6s", -0.0002222216963881069),
#     ("H7", -0.021465598669130555), ("H8", -0.0060444387388412),
#     ("H9", -0.007129635706042069),
#
#     ("Z1s", 0.0013644744381565237), ("Z1c", 0.06207175897316717),
#     ("Z2s", 0.00045851656243805626), ("Z2c", -0.06013978158780949),
#     ("Z3", -0.007335561747475699), ("Z4", -0.009433821313667429),
#     ("Z5", -0.021431246680527438), ("Z6s", 0.0020160501308048038),
#     ("Z7", -0.002348079363078445), ("Z8", 0.09773598626199576),
#     ("Z9", 0.09496335196443795)]

trueValues = {"$f$": 0.67, #proportion of events which are signal
              "$c_0$": 0.75, "$c_1$": 10, "$c_2$": 0.65, #parameters for double exponential time background
              "$k_m$": 2, #decay constant for combinatorial mass background
              "$m_{B^0_s}$": 5.36691, r"$\sigma_m$": 0.015, #signal mass mean and width
              '$x$': 26.93, '$y$': 0.124, '$gamma$': 0.6598,
              # "$K_{1c}$": 0.8661557825167852, "$K_{2s}$": 0.04906538817313792, "$K_{2c}$": -0.8392845161954529,
              # "$K_3$": -0.014487296919474972, "$K_4$": -0.1453399803705659, '$K_5$': -0.1662968116777378, '$K_{6s}$': 0.006879902115106718,
              # '$K_7$': -0.021579262269771426, '$K_8$': -0.006834401112133996, '$K_9$': -0.000917567971296945,
              # # '$A_{CP}$': -0.0023679169347,
              # '$W_{1s}$': 2.89030033977211e-05, '$W_{1c}$': 0.0009074215263274517, '$W_{2s}$': 8.53322844894951e-06, '$W_{2c}$': -0.0008821425176404872,
              # '$W_3$': 2.2017298457314245e-05, '$W_4$': 0.00017483191366255714, '$W_5$': 0.000262027717737243, '$W_{6s}$': -0.0004842899129342597,
              # '$W_7$': 0.0020984263762797157, '$W_8$': 0.0009964600073610955, '$W_9$': 0.00012160250786178678,
              # '$H_{1s}$': 0.021591681620283635, '$H_{1c}$': 0.863927164713325, '$H_{2s}$': 0.007229344144145437, '$H_{2c}$': -0.8371254802199508,
              # '$H_3$': -0.09784412774836446, '$H_4$': -0.14503651133194273, '$H_5$': 0.002348195427913586, '$H_{6s}$': -0.0002222216963881069,
              # '$H_7$': -0.021465598669130555, '$H_8$': -0.0060444387388412, '$H_9$': -0.007129635706042069,
              # '$Z_{1s}$': 0.0013644744381565237, '$Z_{1c}$': 0.06207175897316717, '$Z_{2s}$': 0.00045851656243805626, '$Z_{2c}$': -0.06013978158780949,
              # '$Z_3$': -0.007335561747475699, '$Z_4$': -0.009433821313667429, '$Z_5$': -0.021431246680527438, '$Z_{6s}$': 0.0020160501308048038,
              # '$Z_7$': -0.002348079363078445, '$Z_8$': 0.09773598626199576, '$Z_9$': 0.09496335196443795,
            # 'effSS': 1, 'effOS': 1, 'wSS': 0.0, 'wOS': 0.0
              # 'effSS': 0.8, 'effOS': 0.4, 'wSS': 0.42, 'wOS': 0.39
              }

tagging = {'badTagging': {'effSS': 0.8, 'effOS': 0.4, 'wSS': 0.42, 'wOS': 0.39},
           'goodTagging': {'effSS': 0.8, 'effOS': 0.4, 'wSS': 0.40, 'wOS': 0.36},
           'untagged': {'effSS': 0.0, 'effOS': 0.0, 'wSS': 0.0, 'wOS': 0.0}}

paramOrder = ["$x$", "$y$", "$gamma$", "$K_{1s}$", "$K_{1c}$", "$K_{2s}$", "$K_{2c}$", "$K_3$", "$K_4$", "$K_5$", "$K_{6s}$", "$K_7$", "$K_8$", "$K_9$",
                 "$W_{1s}$", "$W_{1c}$", "$W_{2s}$", "$W_{2c}$", "$W_3$", "$W_4$", "$W_5$", "$W_{6s}$", "$W_7$", "$W_8$", "$W_9$",
                 "$H_{1s}$", "$H_{1c}$", "$H_{2s}$", "$H_{2c}$", "$H_3$", "$H_4$", "$H_5$", "$H_{6s}$", "$H_7$", "$H_8$", "$H_9$",
                 "$Z_{1s}$", "$Z_{1c}$", "$Z_{2s}$", "$Z_{2c}$", "$Z_3$", "$Z_4$", "$Z_5$", "$Z_{6s}$", "$Z_7$", "$Z_8$", "$Z_9$",
                 "$f$", "$c_0$", "$c_1$", "$c_2$", "$k_m$", "$m_{B^0_s}$", r"$\sigma_m$"]

signalParams = ["$K_{1s}$", "$K_{1c}$", "$K_{2s}$", "$K_{2c}$", "$K_3$", "$K_4$", "$K_5$", "$K_{6s}$", "$K_7$", "$K_8$", "$K_9$",
                "$W_{1s}$", "$W_{1c}$", "$W_{2s}$", "$W_{2c}$", "$W_3$", "$W_4$", "$W_5$", "$W_{6s}$", "$W_7$", "$W_8$", "$W_9$",
                "$H_{1s}$", "$H_{1c}$", "$H_{2s}$", "$H_{2c}$", "$H_3$", "$H_4$", "$H_5$", "$H_{6s}$", "$H_7$", "$H_8$", "$H_9$",
                "$Z_{1s}$", "$Z_{1c}$", "$Z_{2s}$", "$Z_{2c}$", "$Z_3$", "$Z_4$", "$Z_5$", "$Z_{6s}$", "$Z_7$", "$Z_8$", "$Z_9$",
                "$m_{B^0_s}$", r"$\sigma_m$"]

backgroundParams = ["$c_0$", "$c_1$", "$c_2$", "$k_m$"]



# def getAllParams():
#     params = []
#     for param in paramOrder:
#         if param == "$K_{1s}$":
#             y = trueValues["$y$"]
#             K1s = (4*(1-y*y) - 3*trueValues["$K_{1c}$"] + trueValues["$K_{2c}$"] + 2*trueValues["$K_{2s}$"] + y*(3*trueValues["$H_{1c}$"] + 6*trueValues["$H_{1s}$"] - trueValues["$H_{2c}$"] - 2*trueValues["$H_{2s}$"]))/6
#             params.append(K1s)
#         else:
#             params.append(trueValues[param])
#     return params

def getAllSignalParamsFromMassless(K1c, K3, K4, K5, K6s, K7, K8, K9,
                             W1s, W1c, W3, W4, W5, W6s, W7, W8, W9,
                             H1s, H1c, H3, H4, H5, H6s, H7, H8, H9,
                             Z1s, Z1c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM):
    x = trueValues['$x$']
    y = trueValues['$y$']
    gamma = trueValues['$gamma$']

    K1s = (3.0/4.0) * (1 - K1c + y*H1c) + y*H1s
    K2s = (1.0/4.0) * (1 - K1c + y*H1c) + y*H1s/3.0
    W2s = W1s/3.0
    H2s = H1s/3.0
    Z2s = Z1s/3.0

    K2c = -K1c
    W2c = -W1c
    H2c = -H1c
    Z2c = -Z1c


    return (K1s, K1c, K2s, K2c, K3, K4, K5, K6s, K7, K8, K9,
            W1s, W1c, W2s, W2c, W3, W4, W5, W6s, W7, W8, W9,
            H1s, H1c, H2s, H2c, H3, H4, H5, H6s, H7, H8, H9,
            Z1s, Z1c, Z2s, Z2c, Z3, Z4, Z5, Z6s, Z7, Z8, Z9, m, sigmaM)


# def getFitParams(massless):
#     signal = [trueValues[param] for param in signalParams if (param != "$K_{1s}$" and (("2s" not in param and "2c" not in param) or not massless))]
#     background = [trueValues[param] for param in backgroundParams]
#     return (signal, background, trueValues["$f$"])

# def getFitParamNames(massless):
#     signal = [param for param in signalParams if (param != "$K_{1s}$" and (("2s" not in param and "2c" not in param) or not massless))]
#     background = [param for param in backgroundParams]
#     return signal + background + ["$f$"]

import json

q2_ranges = ['0.10_0.98', '1.10_6.00', '15.00_19.00']
q2_range_names = ['low', 'central', 'high']
integratedLuminosities = [23, 50, 300]
q2_yields = [[760, 1550, 2030], [1650, 3400, 4400], [9900, 20000, 26700]]



# def getFitParams():
#     with open(generationFile) as f:
#         data = json.load(f)
#         signal = [data[param][q2_ranges[q2_range]][0] for param in data]
#     signal += [trueValues["$m_{B^0_s}$"], trueValues[r"$\sigma_m$"]]
#     background = [trueValues[param] for param in backgroundParams]
#     return (signal, background, trueValues["$f$"])

with open('theory.json') as f:
    theoryData = json.load(f)
with open('alternative.json') as f:
    alternativeData = json.load(f)

def getFitParams(timeDependent, q2_range, generationName):
    data = theoryData if generationName == 'theory' else alternativeData
    if not timeDependent:
        signalParams = ['K1c', 'K3', 'K4', 'W5', 'W6s', 'K7', 'W8', 'W9']#, 'm', 'sigmaM']
        signal = [data[param][q2_ranges[q2_range]][0] for param in signalParams]
        signal += [trueValues["$m_{B^0_s}$"], trueValues[r"$\sigma_m$"]]
        return signal, [trueValues['$k_m$']], trueValues['$f$']
    else:
        signal = [data[param][q2_ranges[q2_range]][0] for param in data]
        signal += [trueValues["$m_{B^0_s}$"], trueValues[r"$\sigma_m$"]]
        background = [trueValues[param] for param in backgroundParams]
        return (signal, background, trueValues["$f$"])

def getFitParamNames(timeDependent):
    if not timeDependent:
        return ['$K_{1c}$', '$K_{3}$', '$K_{4}$', '$W_{5}$', '$W_{6s}$', '$K_{7}$', '$W_{8}$', '$W_{9}$', "$m_{B^0_s}$", r'$\sigma_{m}$', '$k_m$', "$f$"]
    signal = [param for param in signalParams if (param != "$K_{1s}$" and (("2s" not in param and "2c" not in param)))]
    background = [param for param in backgroundParams]
    return signal + background + ["$f$"]


