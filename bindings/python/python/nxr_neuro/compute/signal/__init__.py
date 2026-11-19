"""Transforms and filters of signals (`nxr::signal`): the Welch power and cross-spectral density, the five filters, the band analytic signal and envelope, the time, graph
and joint wavelet transforms."""

from .forms import (coexactStream, exactPotential, flat, intrinsicComplex, modesOf, phaseCurrent, phaseGradient, projectOneForm, projectTwoForm, sharp, splitPhase,
                    synthesizeOneForm, synthesizeTwoForm, vertexGradient)
from .filtering import bandpass, bandstop, decimate, decimationHalo, highpass, lowpass, notch
from .graph import GraphBank, graphBank, gwt, igwt
from .joint import JointCoefficients, ijwt, jwt
from .spectral import WelchResult, psd, xpsd
from .wavelets import TimeBank, bandAnalytic, bandEnvelope, iwt, timeBank, wt

__all__ = [
    "coexactStream", "exactPotential", "flat", "intrinsicComplex", "modesOf", "phaseCurrent", "phaseGradient", "projectOneForm", "projectTwoForm", "sharp", "splitPhase",
    "synthesizeOneForm", "synthesizeTwoForm", "vertexGradient",
    "GraphBank", "JointCoefficients", "TimeBank", "WelchResult", "bandAnalytic", "bandEnvelope", "bandpass", "bandstop", "decimate", "decimationHalo", "graphBank", "gwt", "highpass", "igwt",
    "ijwt", "iwt", "jwt", "lowpass", "notch", "psd", "timeBank", "wt", "xpsd",
]
