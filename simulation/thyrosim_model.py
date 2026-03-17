# Full thyrosim pipeline

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


class Thyrosim:

    def __init__(self, dial, inf, kdelay, params):

        self.d1, self.d2, self.d3, self.d4 = dial
        self.u1, self.u4 = inf
        self.kdelay = kdelay

        for k, v in params.items():
            setattr(self, k, v)

        self.p44 *= self.d2
        self.p46 *= self.d4

    def derivatives(self, t, q):

        p = self
        qDot = np.zeros(19)

        q1F = (p.p7 + p.p8*q[0] + p.p9*q[0]**2 + p.p10*q[0]**3) * q[0]
        q4F = (p.p24 + p.p25*q[0] + p.p26*q[0]**2 + p.p27*q[0]**3) * q[3]

        SR3 = (p.p19 * q[18]) * p.d3
        SR4 = (p.p1 * q[18]) * p.d1

        fCIRC = 1 + (p.p32/(p.p31*np.exp(-q[8])) - 1) * (1/(1+np.exp(10*q[8]-55)))
        SRTSH = (p.p30 + p.p31*fCIRC*np.sin(np.pi/12*t - p.p33)) * np.exp(-q[8])

        fdegTSH = p.p34 + p.p35/(p.p36 + q[6])
        fLAG = p.p41 + 2*q[7]**11/(p.p42**11 + q[7]**11)
        f4 = p.p37 + 5*p.p37/(1 + np.exp(2*q[7]-7))
        NL = p.p13/(p.p14 + q[1])

        qDot[0] = SR4 + p.p3*q[1] + p.p4*q[2] - (p.p5+p.p6)*q1F + p.p11*q[10] + p.u1
        qDot[1] = p.p6*q1F - (p.p3+p.p12+NL)*q[1]
        qDot[2] = p.p5*q1F - (p.p4+p.p15/(p.p16+q[2])+p.p17/(p.p18+q[2]))*q[2]

        qDot[3] = SR3 + p.p20*q[4] + p.p21*q[5] - (p.p22+p.p23)*q4F + p.p28*q[12] + p.u4
        qDot[4] = p.p23*q4F + NL*q[1] - (p.p20+p.p29)*q[4]
        qDot[5] = p.p22*q4F + p.p15*q[2]/(p.p16+q[2]) + p.p17*q[2]/(p.p18+q[2]) - p.p21*q[5]

        qDot[6] = SRTSH - fdegTSH*q[6]
        qDot[7] = f4/p.p38*q[0] + p.p37/p.p39*q[3] - p.p40*q[7]
        qDot[8] = fLAG*(q[7]-q[8])

        qDot[9]  = -p.p43*q[9]
        qDot[10] = p.p43*q[9] - (p.p44+p.p11)*q[10]
        qDot[11] = -p.p45*q[11]
        qDot[12] = p.p45*q[11] - (p.p46+p.p28)*q[12]

        qDot[13] = -p.kdelay*q[13] + q[6]
        qDot[14] = p.kdelay*(q[13] - q[14])
        qDot[15] = p.kdelay*(q[14] - q[15])
        qDot[16] = p.kdelay*(q[15] - q[16])
        qDot[17] = p.kdelay*(q[16] - q[17])
        qDot[18] = p.kdelay*(q[17] - q[18])

        return qDot


# Parameter
def create_base_params():
    return {f"p{i}": 1.0 for i in range(1, 49)}

# Scaling stuff
def apply_patient_scaling(params, height, weight, sex):

    scale = (weight / 70.0)

    params["p1"] *= scale
    params["p19"] *= scale

    if sex.lower() == "female":
        params["p1"] *= 0.9
        params["p19"] *= 0.9

    return params


# Simulation wrapper
def simulate_patient(
    height,
    weight,
    sex,
    lt4_dose,
    lt3_dose,
    rtf,
    t_end=200
    ):

    params = create_base_params()
    params = apply_patient_scaling(params, height, weight, sex)

    # RTF applied here
    params["p1"] *= rtf

    model = Thyrosim(
        dial=[1,1,1,1],
        inf=[lt4_dose, lt3_dose],
        kdelay=0.1,
        params=params
    )

    initial_conditions = np.ones(19)

    sol = solve_ivp(
        model.derivatives,
        (0, t_end),
        initial_conditions,
        method="DOP853"
    )

    t = sol.t
    states = sol.y

    # Outputs
    q0 = states[0]
    q3 = states[3]
    tsh = states[6]

    ft4 = (params["p7"] + params["p8"]*q0 + params["p9"]*q0**2 + params["p10"]*q0**3) * q0
    ft3 = (params["p24"] + params["p25"]*q0 + params["p26"]*q0**2 + params["p27"]*q0**3) * q3

    tt3 = q3  # approximation

    df = pd.DataFrame({
        "time": t,
        "FT4": ft4,
        "FT3": ft3,
        "TT3": tt3,
        "TSH": tsh
    })

    return df