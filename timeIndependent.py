import functools
import time

from iminuit import Minuit
from jax import jit, random
import jax
import jax.numpy as jnp

import tools
import run
import generation
import numpy as np
@jit
def timeIndependentSignal(cosThetaL, cosThetaK, phi, mass,
                    K1c, K3, K4, W5, W6s, K7, W8, W9, m, sigmaM):

    K1s = 3.0*(1.0 - K1c)/4.0
    K2s = K1s/3.0
    K2c = -K1c

    # K1s = (4 - 3*K1c + K2c + 2*K2s)/6
    cosThetaL2 = cosThetaL * cosThetaL
    cosThetaK2 = cosThetaK * cosThetaK
    cos2ThetaL = 2.0 * cosThetaL2 - 1.0
    sinThetaK2 = 1.0 - cosThetaK2
    sinThetaL2 = 1.0 - cosThetaL2
    sinThetaL = jnp.sqrt(sinThetaL2)
    sinThetaK = jnp.sqrt(sinThetaK2)
    sin2ThetaL = 2.0 * sinThetaL * cosThetaL
    sin2ThetaK = 2.0 * sinThetaK * cosThetaK

    signal = (9.0 / 32.0 / jnp.pi) * (
            K1s * sinThetaK2 +
            K1c * cosThetaK2 +
            K2s * sinThetaK2 * cos2ThetaL +
            K2c * cosThetaK2 * cos2ThetaL +
            K3 * sinThetaK2 * sinThetaL2 * jnp.cos(2 * phi) +
            K4 * sin2ThetaK * sin2ThetaL * jnp.cos(phi) +
            W5 * sin2ThetaK * sinThetaL * jnp.cos(phi) +
            W6s * sinThetaK2 * cosThetaL +
            K7 * sin2ThetaK * sinThetaL * jnp.sin(phi) +
            W8 * sin2ThetaK * sin2ThetaL * jnp.sin(phi) +
            W9 * sinThetaK2 * sinThetaL2 * jnp.sin(2 * phi)
    ) * tools.gaussDistribution(mass, m, sigmaM)

    return signal

@jit
def timeIndependentBackground(cosThetaL, cosThetaK, phi, mass, kM):
    massDistr = tools.exponentialDistribution(mass, kM, run.massRange)
    normalization = 0.5 * 0.5 * 1/(2*jnp.pi)
    return massDistr * normalization

@jit
def projectSignalCosThetaL(cosThetaL, K1c, K3, K4, W5, W6s, K7, W8, W9, m, sigmaM):
    K1s = 3.0*(1.0 - K1c)/4.0
    K2s = K1s/3.0
    K2c = -K1c
    signal = 3.0/8.0 * (K1c + 2*K1s - K2c - 2*K2s + 2*W6s*cosThetaL + 2*K2c*cosThetaL**2 + 4*K2s*cosThetaL**2)
    return signal
@jit
def projectBackgroundAngles(angle, *backgroundParams):
    return jnp.full_like(angle, 1/(angle[-1] - angle[0]))

@jit
def projectSignalCosThetaK(cosThetaK, K1c, K3, K4, W5, W6s, K7, W8, W9, m, sigmaM):
    K1s = 3.0*(1.0 - K1c)/4.0
    K2s = K1s/3.0
    K2c = -K1c
    signal = -(3.0/8.0)*(K2s + (-3*K1c + K2c)*cosThetaK**2 - K2s*cosThetaK**2 + 3*K1s*(-1 + cosThetaK**2))
    return signal

@jit
def projectSignalPhi(phi, K1c, K3, K4, W5, W6s, K7, W8, W9, m, sigmaM):
    K1s = 3.0*(1.0 - K1c)/4.0
    K2s = K1s/3.0
    K2c = -K1c
    signal = (3*K1c + 6*K1s - K2c - 2*K2s + 4*K3*jnp.cos(2*phi) + 4*W9*jnp.sin(2*phi))/(8*jnp.pi)
    return signal

@jit
def projectSignalMass(mass, K1c, K3, K4, W5, W6s, K7, W8, W9, m, sigmaM):
    signal = tools.gaussDistribution(mass, m, sigmaM)
    return signal
@jit
def projectBackgroundMass(mass, kM):
    background = tools.exponentialDistribution(mass, kM, run.massRange)
    return background

@jit
def timeIndependentTotal(cosThetaL, cosThetaK, phi, mass, K1c, K3, K4, W5, W6s, K7, W8, W9, m, sigmaM, kM, f):
    return (f*timeIndependentSignal(cosThetaL, cosThetaK, phi, mass, K1c, K3, K4, W5, W6s, K7, W8, W9, m, sigmaM) +
            (1-f)*timeIndependentBackground(cosThetaL, cosThetaK, phi, mass, kM))

@jit
def timeIndependentLikelihood(params, cosThetaL, cosThetaK, phi, mass):
    pdf = timeIndependentTotal(cosThetaL, cosThetaK, phi, mass, *params)
    return -jnp.sum(jnp.log(pdf))

def timeIndependentHelper(K1c, K3, K4, W5, W6s, K7, W8, W9, m, sigmaM, kM, f):
    return lambda cosThetaL, cosThetaK, phi, mass: timeIndependentLikelihood((K1c, K3, K4, W5, W6s, K7, W8, W9, m, sigmaM, kM, f), cosThetaL, cosThetaK, phi, mass)

@jit
def timeIndependentValue(params, cosThetaL, cosThetaK, phi, mass):
    return timeIndependentHelper(*params)(cosThetaL, cosThetaK, phi, mass)

timeIndependentGradHelper = jax.grad(timeIndependentValue, argnums=0)

class TimeIndependentNLL:
    def __init__(self, cosThetaL, cosThetaK, phi, mass):
        self.cosThetaL = cosThetaL
        self.cosThetaK = cosThetaK
        self.phi = phi
        self.mass = mass
        self.gradientHelper = timeIndependentGradHelper

    def __call__(self, K1c, K3, K4, W5, W6s, K7, W8, W9, m, sigmaM, kM, f):
        likelihood = timeIndependentHelper(K1c, K3, K4, W5, W6s, K7, W8, W9, m, sigmaM, kM, f)(self.cosThetaL, self.cosThetaK, self.phi, self.mass)
        if jnp.isfinite(likelihood):
            return float(likelihood)
        return 1e12

    @property
    def grad(self):
        return lambda *params: self.gradientHelper(params, self.cosThetaL, self.cosThetaK, self.phi, self.mass)


@functools.partial(jit, static_argnums=(1,))
def generate(key, N, signalPars, backgroundPars, f):
    keys = random.split(key, 10)
    mul = 1000
    signal = random.uniform(keys[0], shape=(N,)) < f

    cosThetaLSignal = random.uniform(keys[1], shape=(mul*N,), minval=-1.0, maxval=1.0)
    cosThetaKSignal = random.uniform(keys[2], shape=(mul*N,), minval=-1.0, maxval=1.0)
    phiSignal = random.uniform(keys[3], shape=(mul*N,), minval=-jnp.pi, maxval=jnp.pi)
    massSignal = random.uniform(keys[4], shape=(mul*N,), minval=run.massRange[0], maxval=run.massRange[1])
    signalProbabilities = timeIndependentSignal(cosThetaLSignal, cosThetaKSignal, phiSignal, massSignal, *signalPars)
    signalProbabilities /= signalProbabilities.max()

    cosThetaLBackground = random.uniform(keys[5], shape=(mul*N,), minval=-1.0, maxval=1.0)
    cosThetaKBackground = random.uniform(keys[6], shape=(mul*N,), minval=-1.0, maxval=1.0)
    phiBackground = random.uniform(keys[7], shape=(mul*N,), minval=-jnp.pi, maxval=jnp.pi)
    massBackground = random.uniform(keys[8], shape=(mul*N,), minval=run.massRange[0], maxval=run.massRange[1])
    backgroundProbabilities = timeIndependentBackground(cosThetaLBackground, cosThetaKBackground, phiBackground, massBackground, *backgroundPars)
    backgroundProbabilities /= backgroundProbabilities.max()

    cutoff = random.uniform(keys[9], shape=(mul*N,), minval=0.0, maxval=1.0)
    signalAccept = jnp.nonzero(signalProbabilities > cutoff, size=N, fill_value=-1)[0]
    backgroundAccept = jnp.nonzero(backgroundProbabilities > cutoff, size=N, fill_value=-1)[0]

    cosThetaLSignal = cosThetaLSignal[signalAccept]
    cosThetaKSignal = cosThetaKSignal[signalAccept]
    phiSignal = phiSignal[signalAccept]
    massSignal = massSignal[signalAccept]

    cosThetaLBackground = cosThetaLBackground[backgroundAccept]
    cosThetaKBackground = cosThetaKBackground[backgroundAccept]
    phiBackground = phiBackground[backgroundAccept]
    massBackground = massBackground[backgroundAccept]

    cosThetaL = jnp.where(signal, cosThetaLSignal, cosThetaLBackground)
    cosThetaK = jnp.where(signal, cosThetaKSignal, cosThetaKBackground)
    phi = jnp.where(signal, phiSignal, phiBackground)
    mass = jnp.where(signal, massSignal, massBackground)

    return cosThetaL, cosThetaK, phi, mass




def runToys(nEvents, nToys, q2_range, generationName):
    key = jax.random.key(0)

    paramNames = generation.getFitParamNames(False)
    pars = generation.getFitParams(False, q2_range, generationName)
    trueValues = np.array(pars[0] + pars[1] + [pars[2]])
    savedData = []
    values = np.empty(shape=(nToys, len(paramNames)))
    errors = np.empty(shape=(nToys, len(paramNames)))
    pulls = np.empty(shape=(nToys, len(paramNames)))

    i = 0
    failedCounter = 0
    while i < nToys:
        key, key1 = jax.random.split(key)
        tStart = time.time()
        pars = generation.getFitParams(False, q2_range, generationName)
        data = generate(key1, nEvents, *pars)
        nll = TimeIndependentNLL(*data)
        m = Minuit(nll, *pars[0], *pars[1], pars[2])
        limits = [(-1 - 4*abs(param), 1 + 4*abs(param)) for param in trueValues]
        limits[generation.getFitParamNames(False).index("$f$")] = (0, 1)

        m.limits = limits
        m.strategy = 0
        m.errordef = Minuit.LIKELIHOOD
        m.migrad()
        m.hesse()
        if not m.fmin.is_valid or not m.fmin.has_accurate_covar or m.fmin.has_parameters_at_limit or m.fmin.hesse_failed:
            print(m.fmin)
            print(m.params)
            print(f"Failed fit #{i} in {time.time() - tStart:.3f} seconds")
            failedCounter += 1
            continue
        vals = np.array(m.values)
        errs = np.array(m.errors)
        values[i, :] = vals
        errors[i, :] = errs
        pulls[i, :] = (vals - trueValues) / errs
        if i < 10:
            savedData.append(data)
        # print(f"Performed fit #{i} in {time.time() - tStart:.3f} seconds")
        i += 1
    if failedCounter > 0:
        print(f"Failed {failedCounter}/{nToys} fits")
    return paramNames, pulls, values, savedData

if __name__ == "__main__":
    from plotTimeIndependent import plot
    nToys = 1000
    for generationName in ['theory', 'alternative']:
        for q2_range in range(3):
            for luminosity in range(3):
                dir = f'timeIndependent/{generationName}_{generation.q2_range_names[q2_range]}_{generation.integratedLuminosities[luminosity]}'
                nEvents = int(generation.q2_yields[luminosity][q2_range] / generation.trueValues["$f$"])
                names, pulls, values, savedData = runToys(nEvents, nToys, q2_range, generationName)
                plot(names, pulls, values, savedData, nEvents, nToys, dir, q2_range, luminosity, generationName)
                print(f'Completed {dir}')
