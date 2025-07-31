# python version 3.6.5
import sys
import numpy as np
import pandas as pd

import os
from datetime import datetime, timedelta
import argparse
import math
import json
import random
import math
import urllib.request as urllib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.ticker import LogLocator, FixedLocator
from pathlib import Path
from PIL import Image
import io

VERBOSE = 1  # verbose error reporting

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def calc_vapor(tmp):
    """Calculate vapor pressure from temperature in Celsius."""
    alogvp = -2937.4 / (tmp + 273.15) - 4.9283 * math.log10(tmp + 273.15) + 23.5471
    vapor = 10 ** alogvp
    return vapor

def calc_ccl(data):
    """Estimate Convective Condensation Level (CCL) from sounding data."""
    tmp_data = []
    p_ccl = t_ccl = -999.0
    kcnt = 0

    # Interpolate data
    for n in range(1, len(data)):
        ta1, ta2 = float(data[n-1]['ta']), float(data[n]['ta'])
        td1, td2 = float(data[n-1]['td']), float(data[n]['td'])
        p1, p2 = float(data[n-1]['pres']), float(data[n]['pres'])

        trate = (ta2 - ta1) / (p1 - p2)
        drate = (td2 - td1) / (p1 - p2)

        loopcnt = 0
        for m in range(int(p1), int(p2) - 1, -1):
            if m == int(p1):
                tmp_data.append({'pres': m, 'ta': ta1, 'td': td1})
            elif m == int(p2):
                continue
            else:
                loopcnt += 1
                ta = ta1 + trate * loopcnt
                td = td1 + drate * loopcnt
                tmp_data.append({'pres': m, 'ta': ta, 'td': td})

        if n == len(data) - 1:
            tmp_data.append({
                'pres': float(data[-1]['pres']),
                'ta': float(data[-1]['ta']),
                'td': float(data[-1]['td']),
            })

    # Calculate dewpoint and mixing ratio
    estd = calc_vapor(float(tmp_data[0]['td']))
    dws = 0.622 * estd / float(tmp_data[0]['pres'])

    for entry in tmp_data[1:]:
        estd = calc_vapor(entry['ta'])
        tws = 0.622 * estd / entry['pres']
        if dws >= tws:
            p_ccl = entry['pres']
            t_ccl = entry['ta']
            break

    return {'p': p_ccl, 't': t_ccl + 273.15 if t_ccl != -999.0 else -999.0}

def satarry(pbase, tbase, maxpres):
    Tk = 273.15
    g = 9.81
    Rv = 461.51
    Rd = 287.05
    Cpd = 1005.
    HL = 2.5 * 10**6
    HLD = HL ** 2
    EIP = 0.622
    EIPD = EIP ** 2
    HLCP = HL / Cpd
    GAMMAd = g / Cpd
    JSTEP = -1
    HIGH = 0.
    TEM1 = tbase
    isat = 0

    TSATM = []
    PSATM = []

    for j in range(1050, maxpres, JSTEP):
        Es = 6.107 * math.exp(HL / Rv * (1. / Tk - 1. / TEM1))
        PARTU = EIP * HL * Es / (Rd * TEM1)
        PARTL = EIPD * HLD * Es / (Cpd * Rd * TEM1**2)
        GAMMAs1 = GAMMAd * (1. + PARTU / j) / (1. + PARTL / j)
        DZ1 = Rd * TEM1 / g * math.log(j / (j + JSTEP))
        TEM2 = TEM1 - GAMMAs1 * DZ1

        if j <= pbase:
            HIGH += DZ1
            TEM1 = TEM2
            TSATM.append(TEM1)
            PSATM.append(j)
            isat += 1

    return isat, TSATM, PSATM

def calc_lcl(pres, t, td):
    """Calculate Lifting Condensation Level (LCL)."""
    t += 273.15
    td += 273.15
    RCP = 0.286

    estd = -2937.4 / td - 4.9283 * math.log10(td) + 23.5471
    estd = 10 ** estd
    ws = 0.622 * estd / pres
    pup = pres

    while True:
        pup -= 1.
        tnew = t * (pup / pres) ** RCP
        esnew = -2937.4 / tnew - 4.9283 * math.log10(tnew) + 23.5471
        esnew = 10 ** esnew
        wsnew = 0.622 * esnew / pup
        if ws >= wsnew:
            plcl = pup
            tlcl = tnew
            break

    return {'p': plcl, 't': tlcl}

def satarry1(plcl, tlcl):
    """Moist descent above LCL up to 1050 hPa."""
    Tk = 273.15
    g = 9.81
    Rv = 461.51
    Rd = 287.05
    Cpd = 1005.
    HL = 2.5 * 10**6
    HLD = HL ** 2
    EIP = 0.622
    EIPD = EIP ** 2
    GAMMAd = g / Cpd
    JSTEP = -1
    HIGH = 0.
    TEM1 = tlcl
    isat = 0

    TSATM = []
    PSATM = []

    for j in range(int(plcl), 1051):
        Es = 6.107 * math.exp(HL / Rv * (1. / Tk - 1. / TEM1))
        PARTU = EIP * HL * Es / (Rd * TEM1)
        PARTL = EIPD * HLD * Es / (Cpd * Rd * TEM1**2)
        GAMMAs1 = GAMMAd * (1. + PARTU / j) / (1. + PARTL / j)
        DZ1 = Rd * TEM1 / g * math.log(j / (j + JSTEP))
        TEM2 = TEM1 + GAMMAs1 * DZ1

        HIGH -= DZ1
        TEM1 = TEM2
        TSATM.append(TEM1)
        PSATM.append(float(j))
        isat += 1

    return isat, TSATM, PSATM

def calc_tw(data):
    """Calculate wet-bulb temperature for each data point in profile."""
    C2K = 273.15

    for k in range(len(data)):
        entry = data[k]
        if float(entry['pres']) < 300 or float(entry['ta']) <= -999 or float(entry['td']) <= -999:
            continue
        else:
            pres = float(entry['pres'])

        lcl = calc_lcl(pres, float(entry['ta']), float(entry['td']))
        isat, TSATM, PSATM = satarry1(lcl['p'], lcl['t'])

        for m in range(isat):
            if int(pres) == int(PSATM[m]):
                data[k]['tw'] = TSATM[m] - C2K
                break

    return data

def fprtohh(pr):
    if pr < 0:
        return -999
    elif pr >= 234.52:
        return 44308 * (1. - (pr / 1013.25)**0.19023)
    else:
        return 10769 + 6381.6 * math.log(234.52 / pr)

def std_atmos(pres_pa):
    """Standard atmosphere height (meters) from pressure (Pa)."""
    pr = pres_pa * 0.01
    height = 44307.692 * (1.0 - (pr / 1013.25)**0.190)
    return height

def calc_lfc(data, lcl):
    AMIS = -999.
    C2K = 273.15
    TSATM, PSATM = [], []
    PLFC = TLFC = AMIS

    maxpres = int(data[-1]['pres'])
    isat, TSATM, PSATM = satarry(lcl['p'], lcl['t'], maxpres)

    # Interpolation to estimate temperature at LCL level
    for n in range(1, len(data)):
        if float(data[n]['pres']) <= lcl['p']:
            rate = (float(data[n]['ta']) - float(data[n-1]['ta'])) / (fprtohh(float(data[n]['pres'])) - fprtohh(float(data[n-1]['pres'])))
            diff_h = fprtohh(lcl['p']) - fprtohh(float(data[n-1]['pres']))
            templcl = float(data[n-1]['ta']) + rate * diff_h
            break

    chk = False
    icnt = 0
    diff_i = diff_j = 0
    for n in range(1, len(data)):
        if float(data[n]['pres']) >= lcl['p']:
            continue

        if icnt == 0:
            atemp = float(data[n]['ta'])
            btemp = templcl
            pmbmi = lcl['p']
        else:
            atemp = float(data[n]['ta'])
            btemp = float(data[n-1]['ta'])
            pmbmi = float(data[n-1]['pres'])
        icnt += 1

        for m in range(isat):
            if int(pmbmi) == int(PSATM[m]):
                diff_j = btemp - (TSATM[m] - C2K)
                for l in range(m+1, isat):
                    if int(data[n]['pres']) == int(PSATM[l]):
                        diff_i = atemp - (TSATM[l] - C2K)
                        break

                if (diff_i * diff_j) < 0:
                    drate = (atemp - btemp) / (-float(data[n]['pres']) + pmbmi)
                    dtmp, utmp = btemp, atemp
                    dpmb, upmb = pmbmi, float(data[n]['pres'])
                    chk = True
                    break
        if chk:
            vtemp = dtmp
            dtrs = 1.0
            for m in range(isat):
                if int(PSATM[m]) <= int(dpmb) and int(PSATM[m]) >= int(upmb):
                    diff = abs(vtemp - (TSATM[m] - C2K))
                    if diff < dtrs:
                        dtrs = diff
                        PLFC = PSATM[m]
                        TLFC = TSATM[m]
                    vtemp += drate
            break

    return {'p': PLFC, 't': TLFC}

def calc_el(data, lcl, ccl, lfc):
    AMIS = -999.
    C2K = 273.15
    TSATM, PSATM = [], []
    PEL = TEL = AMIS
    chk = False

    maxpres = int(data[-1]['pres'])
    ref_point = ccl if (lfc['p'] == ccl['p'] and lfc['t'] == ccl['t']) else lcl
    isat, TSATM, PSATM = satarry(ref_point['p'], ref_point['t'], maxpres)

    for n in range(1, len(data)):
        if float(data[n-1]['pres']) >= lfc['p']:
            continue

        atemp = float(data[n]['ta'])
        btemp = float(data[n-1]['ta'])

        for m in range(isat):
            if int(data[n-1]['pres']) == int(PSATM[m]):
                diff_j = btemp - (TSATM[m] - C2K)
                for l in range(m+1, isat):
                    if int(data[n]['pres']) == int(PSATM[l]):
                        diff_i = atemp - (TSATM[l] - C2K)
                        break

                if (diff_i * diff_j <= 0) or (n == len(data) - 1):
                    drate = (atemp - btemp) / (-float(data[n]['pres']) + float(data[n-1]['pres']))
                    dtmp, utmp = btemp, atemp
                    dpmb, upmb = float(data[n-1]['pres']), float(data[n]['pres'])
                    chk = True
                    break
        if chk:
            break

    if chk:
        if diff_j > diff_i:
            return {'p': lfc['p'], 't': lfc['t']}
        vtemp = dtmp
        dtrs = 1.0
        for m in range(isat):
            if int(PSATM[m]) <= int(dpmb) and int(PSATM[m]) >= int(upmb):
                diff = abs(vtemp - (TSATM[m] - C2K))
                if diff < dtrs:
                    dtrs = diff
                    PEL = PSATM[m]
                    TEL = TSATM[m]
                if (m == isat - 1) and ((TSATM[m] - C2K) >= (vtemp + dtrs)):
                    PEL = float(data[-1]['pres'])
                    TEL = float(data[-1]['ta']) + C2K
                vtemp += drate

    return {'p': PEL, 't': TEL}

def calc_cape(data, lcl, ccl, lfc, el):
    polygon = []
    AMIS = -999.
    C2K = 273.15
    TSATM, PSATM = [], []
    TCAPE = AMIS

    if lfc['p'] in [AMIS, el['p']] or el['p'] == AMIS:
        return {'value': TCAPE, 'polygon': polygon}

    maxpres = int(data[-1]['pres'])
    isat, TSATM, PSATM = satarry(lcl['p'], lcl['t'], maxpres)

    Tenv, Penv, Henv, Tpcl = [], [], [], []
    for k in reversed(range(0, len(data)-1)):
        p = float(data[k]['pres'])
        if float(data[k]['ta']) > AMIS and el['p'] < p < lfc['p']:
            Tenv.append(float(data[k]['ta']) + C2K)
            Penv.append(p)
            gh = float(data[k].get('gh', AMIS))
            Henv.append(gh if gh != AMIS else std_atmos(p * 100))
            for n in range(isat):
                if int(PSATM[n]) == int(p):
                    Tpcl.append(TSATM[n])
                    break

    def height_interp(p, idx):
        hlow = float(data[idx - 1].get('gh', AMIS))
        hupr = float(data[idx].get('gh', AMIS))
        if hlow == AMIS:
            hlow = std_atmos(float(data[idx - 1]['pres']) * 100)
        if hupr == AMIS:
            hupr = std_atmos(float(data[idx]['pres']) * 100)
        plow = float(data[idx - 1]['pres'])
        pupr = float(data[idx]['pres'])
        delp = (math.log(p) - math.log(plow)) / (math.log(pupr) - math.log(plow))
        return hlow + (hupr - hlow) * delp

    if el['p'] <= data[-1]['pres']:
        Hel = height_interp(el['p'], len(data)-1)
    else:
        Hel = height_interp(el['p'], next(i for i, d in enumerate(data) if el['p'] > float(d['pres'])))
    Hlfc = height_interp(lfc['p'], next(i for i, d in enumerate(data) if lfc['p'] > float(d['pres'])))

    if len(Tpcl) == 0:
        return {'value': TCAPE, 'polygon': polygon}

    tr_Tel = 2. * (Tpcl[0] - Tenv[0]) / (el['t'] + Tenv[0])
    sum_el = 0.5 * (Hel - Henv[0]) * tr_Tel * 9.8

    kk = len(Tenv) - 1
    tr_Tlfc = 2. * (Tpcl[kk] - Tenv[kk]) / (lfc['t'] + Tenv[kk])
    sum_lfc = 0.5 * (Henv[kk] - Hlfc) * tr_Tlfc * 9.8

    TSUM = sum_el + sum_lfc
    for k in range(kk):
        delh = Henv[k] - Henv[k + 1]
        delt = 0.5 * ((Tpcl[k] - Tenv[k]) + (Tpcl[k + 1] - Tenv[k + 1]))
        avet = 0.5 * (Tenv[k] + Tenv[k + 1])
        TSUM += (delh * delt / avet) * 9.8


    if TSUM > 0:
        # First loop: Add points between LFC and EL
        for n in range(isat):
            if PSATM[n] > lfc["p"] or PSATM[n] < el["p"]:
                continue
            d = {
                "ta": TSATM[n] - C2K,
                "pres": PSATM[n]
            }
            polygon.append(d)

        # Add the EL point
        polygon.append({
            "ta": el["t"] - C2K,
            "pres": el["p"]
        })

        # Add environmental profile points
        for k in range(len(Tenv)):
            d = {
                "ta": Tenv[k] - C2K,
                "pres": Penv[k]
            }
            polygon.append(d)

        # Handle special case when LFC == LCL
        if lfc["p"] == lcl["p"]:
            for k in range(len(data)):
                if float(data[k]["pres"]) < lfc["p"]:
                    break

            # Linear interpolation to get temperature at LFC pressure
            ta1 = float(data[k - 1]["ta"])
            ta2 = float(data[k]["ta"])
            p1 = float(data[k - 1]["pres"])
            p2 = float(data[k]["pres"])
            
            interpolated_ta = ta1 - (ta1 - ta2) / (p1 - p2) * (p1 - lfc["p"])
            
            polygon.append({
                "ta": interpolated_ta,
                "pres": lfc["p"]
            })

        # Add the LFC point
        polygon.append({
            "ta": lfc["t"] - C2K,
            "pres": lfc["p"]
        })          
    

    return {'value': TSUM if TSUM > 0 else AMIS, 'polygon': polygon}

def calcHailSize(data, ccl):
    hail = {'size': -999}
    Blev = delta1 = delta2 = size = wbz = -999

    if ccl['p'] == -999.:
        return hail

    # calculate temperature -5 degree level(B level)
    ok = 0
    for k, entry in enumerate(data):
        if float(entry['pres']) > ccl['p']:
            continue
        if float(entry['ta']) < -5:
            ok += 1
            break

    if ok == 1:
        T = float(data[k]['ta'])
        P = float(data[k]['pres'])
        while True:
            RATIO = (T - float(data[k-1]['ta'])) / (P - float(data[k-1]['pres']))
            if T < -5:
                P += 1
                T += RATIO
            if not (T < -5 and RATIO >= 0) or P == float(data[k-1]['pres']):
                break
        Blev = P

    # calculate b
    maxpres = int(float(data[-1]['pres']))
    isat, TSATM, PSATM = satarry(ccl['p'], ccl['t'], maxpres)

    b = None
    for m in range(isat):
        if PSATM[m] > Blev:
            continue
        elif int(PSATM[m]) == int(Blev):
            b = TSATM[m] - 273.15
            break
    delta1 = b + 5 if b is not None else -999

    # calculate c
    RCP = 0.286
    c = (-5 + 273.15) * (ccl['p'] / Blev)**RCP - 273.15
    delta2 = c + 5

    size = determine_size(delta1, delta2)

    # calculate wet-bulb freezing level(wbz)
    ok = 0
    for k, entry in enumerate(data):
        ta = float(entry['ta'])
        tw = float(entry['tw'])
        pres = float(entry['pres'])

        if k == 0 and tw < 0:
            break
        if tw < 0:
            ok += 1
            break

        tw0 = tw
        pres0 = pres

    if ok == 1:
        T = tw
        P = pres
        while True:
            RATIO = (T - tw0) / (P - pres0)
            if T < 0:
                P += 1
                T += RATIO
            if not (T < 0 and RATIO >= 0):
                break
        wbz = fprtohh(P)
    else:
        wbz = 0

    size = adjust_for_wbz(size, wbz)

    hail.update({'delta1': delta1, 'delta2': delta2, 'size': size, 'wbz': wbz})
    return hail

def determine_size(delta1, delta2):
    table = [
        (50, [(10,99), (9,85), (8,75), (7,60), (6,50), (5,45), (4,35), (3,25), (2,20), (1,10), (0,5)]),
        (45, [(11,99), (10,90), (9,80), (8,65), (7,60), (6,50), (5,45), (4,35), (3,25), (2,20), (1,5)]),
        (40, [(12,99), (11,85), (10,80), (9,70), (8,60), (7,55), (6,45), (5,40), (4,30), (3,20), (2,15), (1,5)]),
        (35, [(12,80), (11,75), (10,70), (9,60), (8,55), (7,50), (6,45), (5,40), (4,30), (3,20), (2,15), (1,5)]),
        (30, [(12,65), (10,60), (9,55), (8,50), (7,45), (6,40), (5,35), (4,25), (3,20), (2,10)]),
        (25, [(13,55), (10,50), (9,45), (7,40), (6,35), (5,30), (4,20), (3,15), (2,5)]),
        (20, [(13,50), (11,45), (9,40), (8,35), (7,30), (6,25), (5,20), (4,15), (3,5)]),
        (15, [(11,35), (10,30), (8,25), (6,20), (5,15), (4,10)]),
        (10, [(12,25), (10,20), (7,15), (6,10), (5,5)]),
        (0, [(12,15), (8,10), (6,5)]),
    ]

    for delta2_thresh, pairs in table:
        if delta2 >= delta2_thresh:
            for d1_thresh, s in pairs:
                if delta1 >= d1_thresh:
                    return s
            break
    return 0

def adjust_for_wbz(size, wbz):
    if wbz >= 4400:
        return 0
    elif wbz >= 4150:
        return 5 if size >= 100 else 0
    elif wbz >= 3950:
        return 10 if size >= 75 else 5 if size >= 50 else 0
    elif wbz >= 3750:
        return 15 if size >= 75 else 10 if size >= 50 else 5 if size >= 25 else 0
    elif wbz >= 3550:
        return 30 if size >= 125 else 25 if size >= 100 else 20 if size >= 50 else 10 if size >= 25 else 5 if size >= 20 else 0
    elif wbz >= 3350:
        return 75 if size >= 125 else 65 if size >= 100 else 50 if size >= 75 else 25 if size >= 50 else 15 if size >= 25 else 10 if size >= 20 else 5 if size >= 10 else 0
    else:
        return size

def calc_midval(Pm, Pl, Pu, Tl, Tu):
    """Interpolate value at Pm between Pl and Pu for T."""
    return Tl + (Pm - Pl) * (Tu - Tl) / (Pu - Pl)

def calc_cin(data, lcl, ccl, lfc, el):
    polygon = []

    AMIS = -999.0
    C2K = 273.15
    cin = {}
    
    TSATM = []
    PSATM = []
    TCIN = AMIS

    if lfc['p'] == AMIS or el['p'] == AMIS:
        cin['value'] = TCIN if TCIN >= 0 else AMIS
        cin['polygon'] = polygon
        return cin

    # Step 1: Generate moist adiabatic profile from LCL to LFC
    maxpres = int(float(data[-1]['pres']))
    isat, TSATM, PSATM = satarry(lcl['p'], lcl['t'], maxpres)

    # Loop 1: Between LCL and LFC
    for n in range(isat):
        if PSATM[n] > lcl['p'] or PSATM[n] < lfc['p']:
            continue
        d = {
            "ta": TSATM[n] - C2K,
            "pres": PSATM[n]
        }
        polygon.append(d)

    # Step 2: Fill the CIN area from LCL to LFC
    Tenv = []
    Penv = []
    Henv = []
    Tpcl = []

    for k in reversed(range(len(data))):
        d = data[k]
        if d['pres'] == "SFC":
            continue

        pres = d['pres']
        ta = d['ta'] if d['ta'] is not None else AMIS
        gh = d['gh'] if d.get('gh') not in (None, AMIS) else std_atmos(pres * 100)

        if ta > AMIS and lcl['p'] > pres > lfc['p']:
            Tenv.append(ta + C2K)
            Penv.append(pres)
            Henv.append(gh)

            for n in range(isat):
                if int(PSATM[n]) == int(pres):
                    Tpcl.append(TSATM[n])
                    break

    for k in reversed(range(len(data))):
        d = data[k]
        if d['pres'] == "SFC":
            continue

        pres = d['pres']
        ta = d['ta'] if d['ta'] is not None else AMIS
        gh = d['gh'] if d.get('gh') not in (None, AMIS) else std_atmos(pres * 100)

        if ta > AMIS and pres >= lcl['p']:
            break

    pres1 = data[k]['pres']
    pres2 = data[k+1]['pres']
    ta1 = data[k]['ta']
    ta2 = data[k+1]['ta']
    Tlcle = calc_midval(lcl['p'], pres1, pres2, ta1, ta2)

    gh1 = data[k]['gh'] if data[k].get('gh') not in (None, AMIS) else std_atmos(pres1 * 100)
    gh2 = data[k+1]['gh'] if data[k+1].get('gh') not in (None, AMIS) else std_atmos(pres2 * 100)
    Hlcle = calc_midval(lcl['p'], pres1, pres2, gh1, gh2)

    Tenv.append(Tlcle + C2K)
    Penv.append(lcl['p'])
    Henv.append(Hlcle)
    Tpcl.append(lcl['t'])

    if Tpcl[-1] >= Tenv[-1]:
        cin['value'] = TCIN if TCIN >= 0 else AMIS
        cin['polygon'] = polygon
        return cin

    Hlcl = Hlcle
    Hlfc = None

    for k in range(len(data)):
        if lfc['p'] > data[k]['pres']:
            pres_low = data[k-1]['pres']
            pres_up = data[k]['pres']
            gh_low = data[k-1]['gh'] if data[k-1].get('gh') not in (None, AMIS) else std_atmos(pres_low * 100)
            gh_up = data[k]['gh'] if data[k].get('gh') not in (None, AMIS) else std_atmos(pres_up * 100)
            delP = (np.log(lfc['p']) - np.log(pres_low)) / (np.log(pres_up) - np.log(pres_low))
            Hlfc = gh_low + (gh_up - gh_low) * delP
            break

    for k in range(len(data)):
        if lcl['p'] > data[k]['pres']:
            pres_low = data[k-1]['pres']
            pres_up = data[k]['pres']
            gh_low = data[k-1]['gh'] if data[k-1].get('gh') not in (None, AMIS) else std_atmos(pres_low * 100)
            gh_up = data[k]['gh'] if data[k].get('gh') not in (None, AMIS) else std_atmos(pres_up * 100)
            delP = (np.log(lcl['p']) - np.log(pres_low)) / (np.log(pres_up) - np.log(pres_low))
            Hlcl = gh_low + (gh_up - gh_low) * delP
            break


    tr_Tlfc = 2 * (Tpcl[0] - Tenv[0]) / (lfc['t'] + Tenv[0])
    Sum_lfc = 0.5 * (Hlfc - Henv[0]) * tr_Tlfc * 9.8
    TSUM1 = Sum_lfc

    for k in range(len(Tenv) - 1):
        delh = Henv[k] - Henv[k+1]
        delt = 0.5 * ((Tpcl[k] - Tenv[k]) + (Tpcl[k+1] - Tenv[k+1]))
        avet = 0.5 * (Tenv[k] + Tenv[k+1])
        sum_cin = (delh * delt / avet) * 9.8
        TSUM1 += sum_cin

    TCIN = -TSUM1

    # Step 6: Dry adiabat from LCL down
    RCP = 0.286
    P0 = 100000.0  # Reference pressure in Pa
    DT = 0.0

    JMAX = int(data[0]['pres'])  # surface pressure (hPa)
    PSATM = []
    TSATM = []
    IDRY = 0

    for j in range(int(lcl['p']), JMAX + 1):
        PSATM.append(float(j))
        P = PSATM[IDRY] * 100.0  # Pa
        PPP = (P0 / P) ** RCP
        T = DT + (data[0]['ta'] + C2K) / PPP

        if IDRY == 0:
            PSATM[IDRY] = lcl['p']
            TSATM.append(lcl['t'])
            DT = lcl['t'] - T
        else:
            TSATM.append(T)

        if j == JMAX:
            PLBL = PSATM[IDRY]
            TLBL = TSATM[IDRY]

        IDRY += 1

    # Find LBL crossing
    MXBL = IDRY
    chk = False
    for n in range(len(data) - 1, 0, -1):
        if data[n-1]['pres'] <= lcl['p']:
            continue

        ATEMP = data[n-1]['ta']
        BTEMP = data[n]['ta']

        for m in range(IDRY):
            if int(data[n]['pres']) == int(PSATM[m]):
                DIFF_J = BTEMP - (TSATM[m] - C2K)
                DIFF_I = None

                for l in range(m+1,IDRY):
                    if int(data[n-1]['pres']) == int(PSATM[l]):
                        DIFF_I = ATEMP - (TSATM[l] - C2K)
                        break

                if DIFF_I is not None and DIFF_I * DIFF_J <= 0:
                    # Found crossing
                    DRATE = (ATEMP - BTEMP) / (-data[n]['pres'] + data[n-1]['pres'])
                    DPMB = data[n-1]['pres']
                    DTMP = BTEMP
                    UPMB = data[n]['pres']
                    UTMP = ATEMP
                    chk = True
                    break

        if chk:
            break

    if chk:
        VTEMP = DTMP
        DTRS = 1.0
        for m in range(IDRY):
            if int(PSATM[m]) <= int(DPMB) and int(PSATM[m]) >= int(UPMB):
                DIFF = abs(VTEMP - (TSATM[m] - C2K))
                if DIFF < DTRS:
                    DTRS = DIFF
                    PLBL = PSATM[m]
                    TLBL = TSATM[m]
                    MXBL = m
                VTEMP += DRATE

    # Step 7: Find LBL
    Tenv = []
    Penv = []
    Henv = []
    Tpcl = []
    kk = 0

    # Find ICHK if PLBL == 1000 hPa
    ICHK = 0
    if PLBL == 1000.0:
        for k in range(len(data)):
            if data[k]['pres'] < PLBL:
                ICHK = k
                break

    if ICHK > 0:
        P = data[ICHK - 1]['pres'] * 100.0  # Pa
        T = data[ICHK - 1]['ta'] + C2K
        Penv.append(data[ICHK - 1]['pres'])
        gh = data[ICHK - 1].get('gh')
        if gh is None or float(gh) == AMIS:
            Henv.append(std_atmos(data[ICHK - 1]['pres'] * 100))
        else:
            Henv.append(gh)
        Tenv.append(T)
        Tpcl.append(TLBL)
        kk += 1

        for j in range(IDRY):
            if int(PSATM[j]) == int(data[ICHK - 1]['pres']):
                PLBL = PSATM[j]
                TLBL = TSATM[j]
                break

    for k in range(len(data)):
        pres = data[k]['pres']
        ta = data[k]['ta'] if data[k]['ta'] is not None else AMIS
        gh = data[k]['gh'] if data[k].get('gh') not in (None, AMIS) else std_atmos(pres * 100)

        if ta > AMIS and (pres < PLBL) and (pres > lcl['p']):
            Penv.append(pres)
            Tenv.append(ta + C2K)
            Henv.append(gh)

            for n in range(IDRY):
                if int(PSATM[n]) == int(data[k]['pres']):
                    Tpcl.append(TSATM[n])
                    #if Tpcl[kk] >= Tenv[kk]:
                    #    cin['value'] = TCIN
                    #    return cin
                    break

            kk += 1

    # Finally, add LCL point
    Tenv.append(Tlcle + C2K)
    Penv.append(lcl['p'])
    Henv.append(Hlcle)
    Tpcl.append(lcl['t'])

    # Calculate geopotential height at LBL
    HLBL = AMIS
    KCHK = -1
    Hlow, Plow, Hupr, Pupr = AMIS, AMIS, AMIS, AMIS

    for k in range(len(data)):
        if PLBL > data[k]['pres']:
            Hlow = data[k-1]['gh'] if data[k-1].get('gh') not in (None, AMIS) else std_atmos(data[k-1]['pres'] * 100)
            Plow = data[k-1]['pres']
            Hupr = data[k]['gh'] if data[k].get('gh') not in (None, AMIS) else std_atmos(data[k]['pres'] * 100)
            Pupr = data[k]['pres']
            KCHK = k
            break

    if KCHK > 0 and Hupr != AMIS and Hlow != AMIS:
        delP = (np.log(PLBL) - np.log(Plow)) / (np.log(Pupr) - np.log(Plow))
        HLBL = Hlow + (Hupr - Hlow) * delP

    # Step 8: Integrate LBL to LCL CIN
    tr_Tlbl = 2 * (Tpcl[0] - Tenv[0]) / (TLBL + Tenv[0])
    Sum_lbl = (Henv[0] - HLBL) * tr_Tlbl * 9.8 if PLBL < 1000 else 0.5 * (Henv[0] - HLBL) * tr_Tlbl * 9.8
    TSUM2 = Sum_lbl

    for k in range(len(Tenv) - 1):
        delh = Henv[k+1] - Henv[k]
        delt = 0.5 * ((Tpcl[k] - Tenv[k]) + (Tpcl[k+1] - Tenv[k+1]))
        avet = 0.5 * (Tenv[k] + Tenv[k+1])
        sum_cin = (delh * delt / avet) * 9.8
        TSUM2 += sum_cin

    TCIN -= TSUM2

    if TCIN > 0:
        # Loop 2: Reverse data slice between PLBL and LFC
        for k in range(len(data) - 1, -1, -1):
            pres_k = float(data[k]["pres"])
            if pres_k >= PLBL or pres_k <= lfc['p']:
                continue
            d = {
                "ta": float(data[k]["ta"]),
                "pres": pres_k
            }
            polygon.append(d)

        # Conditional point if top matches PLBL but ta doesn't match TLBL
        if int(PLBL) == int(float(data[0]["pres"])) and (TLBL - C2K) != float(data[0]["ta"]):
            d = {
                "ta": float(data[0]["ta"]),
                "pres": PLBL
            }
            polygon.append(d)

        # Add point at (TLBL - C2K, PLBL)
        polygon.append({
            "ta": TLBL - C2K,
            "pres": PLBL
        })

        # Loop 3: Dry ascent layer (reverse through IDRY)
        for j in range(IDRY - 1, -1, -1):
            d = {
                "ta": TSATM[j] - C2K,
                "pres": PSATM[j]
            }
            polygon.append(d)


    cin['value'] = TCIN if TCIN >= 0 else AMIS
    cin['polygon'] = polygon
    return cin

def describe_cape_cin_lm(cape, cin):
    desc = ""

    if cape > 100 and cape < 500:
            options = [
                f"CAPE is {cape:.0f} J/kg, considered low. Shallow convection may occur, but the potential for significant thunderstorms is limited.",
                f"A CAPE of {cape:.0f} J/kg suggests marginal instability. Convective clouds could develop under favorable surface forcing.",
                f"{cape:.0f} J/kg of CAPE represents limited atmospheric instability, potentially supporting isolated weak convection."
            ]
            desc = random.choice(options) + "\n"
    elif cape >= 500 and cape < 1000:
        options = [
            f"The CAPE value of {cape:.0f} J/kg indicates moderate instability. Under adequate lifting mechanisms, deep convection is possible.",
            f"{cape:.0f} J/kg of CAPE suggests sufficient energy for thunderstorm development, given appropriate environmental triggers.",
            f"Moderate CAPE levels (~{cape:.0f} J/kg) suggest the environment can support convective activity if inhibition is overcome."
        ]
        desc = random.choice(options) + "\n"
    elif cape >= 1000:
        options = [
            f"CAPE is elevated at {cape:.0f} J/kg, indicating strong atmospheric instability capable of supporting deep, vigorous convection.",
            f"With {cape:.0f} J/kg of CAPE, the atmosphere contains ample energy for thunderstorm initiation and vertical development.",
            f"The high CAPE value of {cape:.0f} J/kg reflects a strongly unstable profile, favorable for organized convective storms."
        ]
        if cape >= 2000:
            options = [opt + " Severe thunderstorms are possible if additional conditions such as lift and moisture are present." for opt in options]
        desc = random.choice(options) + "\n"

    if cin > 75:
        options = [
            f"The CIN value of {cin:.0f} J/kg reflects a cap, requiring significant lifting or surface heating to overcome.",
            f"A CIN of {cin:.0f} J/kg may prevent convective initiation without strong synoptic or mesoscale forcing.",
            f"With CIN at {cin:.0f} J/kg, vertical motion is likely to be suppressed in the absence of sufficient triggering mechanisms."
        ]
        desc += random.choice(options) + "\n"
    elif cin > 150:
        options = [
            f"CIN is strong at {cin:.0f} J/kg. This level of inhibition is typically sufficient to suppress convection entirely without substantial forcing.",
            f"The convective cap is pronounced, with CIN exceeding 150 J/kg, indicating a stable boundary layer resistant to vertical parcel ascent.",
            f"At {cin:.0f} J/kg, the inhibition is substantial, making convective development unlikely under typical diurnal heating conditions."
        ]
        desc += random.choice(options) + "\n"

    return desc

def describe_cape_cin(cape, cin):
    desc = ""

    # Figurative descriptions for CAPE
    if cape > 100 and cape < 500:
        options = [
            "A narrow blue-shaded area (CAPE) suggests limited instability. Shallow convection might occur under favorable surface conditions.",
            "The modest extent of the blue region (CAPE) hints at marginal atmospheric instability with potential for weak convective development.",
            "Only a thin blue layer (CAPE) appears on the diagram, reflecting weak buoyancy and a limited chance for thunderstorms."
        ]
        desc = random.choice(options) + "\n"
    elif cape >= 500 and cape < 1000:
        options = [
            "A moderately sized blue-shaded region (CAPE) implies moderate instability, which may support thunderstorm development if lifting is present.",
            "The blue area (CAPE) is noticeable but not expansive, suggesting an environment that can support convection if other triggers align.",
            "Blue shading (CAPE) of moderate depth and width suggests some potential for vertical motion and convective activity."
        ]
        desc = random.choice(options) + "\n"
    elif cape >= 1000:
        options = [
            "A large and vertically extended blue-shaded area (CAPE) indicates strong instability, favoring the development of deep convection.",
            "The expansive and deep blue region (CAPE) reflects a highly unstable atmosphere, primed for thunderstorm formation.",
            "A pronounced blue column (CAPE) stretching through much of the troposphere reveals ample energy for vigorous vertical development."
        ]
        if cape >= 2000:
            options = [opt + " The setup may support severe storms if lifting and moisture are also sufficient." for opt in options]
        desc = random.choice(options) + "\n"

    # Figurative descriptions for CIN
    if cin > 150:
        options = [
            "A thick yellow-shaded area (CIN) overlays the lower atmosphere, suggesting a strong cap suppressing vertical motion.",
            "The dense yellow layer (CIN) near the surface reflects significant inhibition, likely preventing convection unless strong forcing is present.",
            "A heavy yellow band (CIN) suggests a stable lower atmosphere resistant to convective initiation under normal conditions."
        ]
        desc += random.choice(options) + "\n"
    elif cin > 75:
        options = [
            "A noticeable yellow-shaded layer (CIN) sits at the base of the sounding, indicating moderate inhibition that could delay or prevent storm development.",
            "The presence of a yellow band (CIN) near the surface suggests some resistance to upward motion, requiring heating or lifting to overcome.",
            "Yellow shading (CIN) in the lower levels hints at a convective cap, limiting parcel ascent unless sufficient triggering occurs."
        ]
        desc += random.choice(options) + "\n"

    return desc

def describe_lcl_lm(lcl):
    if lcl > 900:
        options = [
            f"The LCL is relatively low at {lcl:.0f} hPa, suggesting a moist boundary layer and low cloud base height.",
            f"With an LCL of {lcl:.0f} hPa, condensation is likely to occur at low altitudes, indicative of high near-surface humidity.",
            f"A low LCL (~{lcl:.0f} hPa) implies efficient cloud formation at the lower layer."
        ]
    elif lcl > 800:
        options = [
            f"LCL is moderately low at {lcl:.0f} hPa. Cloud base heights are elevated relative to saturated profiles, but not unusually so.",
            f"{lcl:.0f} hPa LCL suggests a moderately moist boundary layer, where cloud formation may occur with modest lifting.",
            f"The lifting condensation level at {lcl:.0f} hPa represents a transitional moisture regime, requiring moderate ascent to reach saturation."
        ]
    else:
        options = [
            f"The LCL is high at {lcl:.0f} hPa, reflecting a dry lower troposphere and indicating clouds will form at elevated levels.",
            f"A high LCL (~{lcl:.0f} hPa) is consistent with drier surface conditions and suppressed low-level cloud formation.",
            f"LCL at {lcl:.0f} hPa implies that air parcels must ascend significantly before condensation occurs."
        ]

    return random.choice(options) + "\n"

def describe_lfc_el_lm(lfc, el, cape):
    desc = ""
    if cape < 500:
        return desc

    if lfc > 900:
        options = [
            f"The LFC is low at {lfc:.0f} hPa, indicating that parcels require minimal ascent to become positively buoyant. This is favorable for convective development.",
            f"A low LFC (~{lfc:.0f} hPa) reflects a reduced barrier to parcel buoyancy, increasing the likelihood of spontaneous convection under minimal forcing.",
            f"With an LFC near {lfc:.0f} hPa, the atmosphere supports easy transition to free convection once parcels reach saturation."
        ]

        desc = random.choice(options) + "\n"
    elif lfc > 700:
        options = [
            f"LFC is moderate at {lfc:.0f} hPa, requiring parcels to ascend through a larger depth before reaching positive buoyancy.",
            f"A moderately high LFC of {lfc:.0f} hPa suggests that convection will require more substantial lifting or boundary-layer heating.",
            f"The LFC at {lfc:.0f} hPa indicates parcels must traverse a neutral or stable layer before initiating buoyant ascent."
        ]

        desc = random.choice(options) + "\n"
    elif lfc > 0:
        options = [
            f"A high LFC at {lfc:.0f} hPa indicates that significant vertical motion or synoptic-scale lifting is needed for parcels to become buoyant.",
            f"Convection is unlikely without strong forcing, as the LFC is elevated to {lfc:.0f} hPa.",
            f"The elevated LFC (~{lfc:.0f} hPa) implies resistance to parcel ascent and reduces the probability of convective initiation."
        ]

        desc = random.choice(options) + "\n"

    if el > 0 and el <= 250:
        options = [
            f"EL is high at {el:.0f} hPa, suggesting deep vertical development and the potential for intense convection, including overshooting tops.",
            f"An EL of {el:.0f} hPa is indicative of strong updraft potential and storm tops penetrating into the upper troposphere or lower stratosphere.",
            f"The elevated EL (~{el:.0f} hPa) supports deep convective towers and is characteristic of vigorous storm systems."
        ]

        desc += random.choice(options) + "\n"
    elif el > 0 and el <= 400:
        options = [
            f"EL is moderately high at {el:.0f} hPa, conducive to substantial vertical growth in convective clouds.",
            f"With an EL near {el:.0f} hPa, the environment is supportive of mature thunderstorms reaching high altitudes.",
            f"An EL of {el:.0f} hPa indicates sufficient depth for organized convection and storm development."
        ]

        desc += random.choice(options) + "\n"

    return desc

def describe_lfc_el(lfc, el, cape):
    desc = ""
    if cape < 500:
        return desc  # Not enough CAPE for meaningful LFC/EL interpretation

    # Interpret LFC (lowest point of the blue region)
    if lfc > 900:
        options = [
            "The blue-shaded area begins low in the atmosphere, indicating that parcels become buoyant with minimal ascent. This favors easy convective initiation.",
            "With the base of the blue region positioned close to the surface, the atmosphere allows parcels to rise freely with little lifting required.",
            "The low start of the CAPE region reveals minimal inhibition for parcel ascent, encouraging convection even under weak surface forcing."
        ]
        desc = random.choice(options) + "\n"
    elif lfc > 700:
        options = [
            "The blue shading starts noticeably above the surface, meaning air parcels need to rise through a neutral or slightly stable layer before becoming buoyant.",
            "With the CAPE region not beginning until mid-levels, some lifting or heating is needed to overcome initial resistance to ascent.",
            "The lower boundary of the blue region sits at a moderate height, requiring moderate vertical motion for convection to initiate."
        ]
        desc = random.choice(options) + "\n"
    elif lfc > 0:
        options = [
            "The blue-shaded area starts high in the atmosphere, suggesting parcels face strong resistance to ascent and require substantial forcing to become buoyant.",
            "When the CAPE region doesn't appear until high levels, it reflects a stable layer below that suppresses spontaneous convection.",
            "A delayed start of the blue shading implies significant atmospheric resistance, making convective initiation unlikely without strong dynamic lifting."
        ]
        desc = random.choice(options) + "\n"

    # Interpret EL (highest point of the blue region)
    if el > 0 and el <= 250:
        options = [
            "The blue region extends high into the upper atmosphere, indicating the potential for very tall convective towers and vigorous updrafts.",
            "A deep vertical extent of the blue shading points to strong instability, capable of supporting overshooting tops and intense convection.",
            "The CAPE area reaches into the upper levels, suggesting a storm environment with robust vertical development."
        ]
        desc += random.choice(options) + "\n"
    elif el > 0 and el <= 400:
        options = [
            "The top of the blue shading reaches moderately high levels, signaling support for deep but typical thunderstorm development.",
            "With the CAPE region extending well above mid-levels, the atmosphere supports sustained updrafts and organized convection.",
            "The vertical reach of the blue-shaded region indicates an environment capable of supporting strong storm growth."
        ]
        desc += random.choice(options) + "\n"

    return desc

def describe_conclusion(rn):
    if rn < 0.1:
        desc = f"Based on the analysis, the probability of precipitation is low."
    elif rn < 1:
        desc = f"Based on the analysis, the probability of precipitation is moderate."
    elif rn < 5:
        desc = f"Based on the analysis, the probability of precipitation is high."
    else:
        desc = f"Based on the analysis, the probability of precipitation is very high."

    return desc

def describe_moisture_wind_lm(low_ttd, mid_ttd, upper_ttd, low_rot, mid_rot, upper_rot):
    desc = ""

    def is_saturated(ttd): return ttd <= 1
    def is_moist(ttd): return ttd <= 3

    if low_ttd > 3:
        if is_saturated(mid_ttd) and is_moist(upper_ttd):
            options = [
                "Despite dry conditions in the boundary layer, elevated moisture supports potential precipitation aloft, though evaporation is likely before reaching the surface.",
                "Mid and upper-level moisture may lead to precipitation generation, but a dry sub-cloud layer could result in virga.",
                "Precipitation may form above, but dry lower levels could inhibit surface rainfall through evaporation."
            ]
            desc += random.choice(options) + "\n"
            if mid_rot >= 30 and upper_rot >= 30:
                options = [
                    "Veering winds aloft indicate favorable shear, potentially enhancing convective organization.",
                    "Directional shear in the mid and upper layers may assist in storm structure development.",
                    "Winds veering with height suggest increased vertical shear, which can support organized convection."
                ]
                desc += random.choice(options) + "\n"
        elif is_moist(mid_ttd) and is_moist(upper_ttd):
            options = [
                "The mid and upper troposphere are sufficiently moist, but a dry boundary layer may result in evaporation of falling precipitation.",
                "Moisture is present aloft. However, dry air near the surface could suppress precipitation reaching the ground.",
                "Clouds may develop aloft, but dry lower layers are likely to limit surface-level precipitation."
            ]
            desc += random.choice(options) + "\n"
        elif is_moist(mid_ttd):
            options = [
                "Moisture in the mid-levels supports the development of mid-tropospheric cloud layers.",
                "Mid-tropospheric humidity could lead to cloud formation at those levels.",
                "There is enough moisture in the mid-level to support cloud cover."
            ]
            desc += random.choice(options) + "\n"
        elif is_moist(upper_ttd):
            options = [
                "Upper-tropospheric moisture may result in high-level cloud formation.",
                "Some cirriform clouds may form due to upper-level saturation.",
                "High-altitude clouds could form despite dry conditions below."
            ]
            desc += random.choice(options) + "\n"
        else:
            options = [
                "The atmosphere is predominantly dry, and significant cloud development is unlikely.",
                "Dry conditions throughout the column suggest limited potential for cloud or precipitation formation.",
                "Overall atmospheric dryness indicates a suppressed convective environment."
            ]
            desc += random.choice(options) + "\n"
    else:
        if is_saturated(mid_ttd) and is_moist(upper_ttd):
            options = [
                "Saturated lower levels with moisture aloft support widespread cloud development and potential precipitation.",
                "The atmosphere is favorably moist from the surface through the mid-troposphere, increasing the chance of rainfall.",
                "Vertical moisture continuity enhances the likelihood of rain or stratiform precipitation."
            ]
            desc += random.choice(options) + "\n"
            if mid_rot >= 30 and upper_rot >= 30:
                options = [
                    "Veering winds in the mid and upper levels enhance vertical shear, supporting storm development.",
                    "Shear induced by veering profiles may favor storm organization.",
                    "Mid-to-upper level veering indicates enhanced convective potential in a moist environment."
                ]
                desc += random.choice(options) + "\n"
        elif is_moist(mid_ttd) and is_moist(upper_ttd):
            options = [
                "Moisture through much of the troposphere supports widespread cloud formation and convective potential.",
                "The atmospheric column exhibits sufficient humidity to support rain-producing cloud layers.",
                "Widespread moist conditions enhance the probability of precipitation and cloud depth."
            ]
            desc += random.choice(options) + "\n"
        elif is_moist(mid_ttd):
            options = [
                "Moisture in the lower and mid-levels may support low to mid-level stratiform cloud formation.",
                "The presence of moisture in the boundary and mid-layers may support cloudiness.",
                "Conditions may favor low-to-mid level cloud development due to sufficient humidity."
            ]
            desc += random.choice(options) + "\n"
        elif is_moist(upper_ttd):
            options = [
                "Upper and lower level moisture is present, but mid-layer is dry.",
                "Split moisture layers may result in cloud decks at separate levels with limited vertical continuity.",
                "High clouds and shallow low clouds may coexist."
            ]
            desc += random.choice(options) + "\n"
        else:
            if is_saturated(low_ttd):
                options = [
                    "The saturated boundary layer may support fog or drizzle.",
                    "Surface-level moisture may lead to low clouds or stratiform precipitation.",
                    "Low-level saturation is favorable for near-surface cloud formation."
                ]
                desc += random.choice(options) + "\n"
            else:
                options = [
                    "Low-level moisture supports shallow cloud development.",
                    "Moisture is present in the boundary layer.",
                    "Only the lower troposphere is moist, suggesting shallow stratiform cloud potential."
                ]
                desc += random.choice(options) + "\n"

    return desc

def describe_moisture_wind(low_ttd, mid_ttd, upper_ttd, low_rot, mid_rot, upper_rot):
    desc = ""

    def is_saturated(ttd): return ttd <= 1
    def is_moist(ttd): return ttd <= 3

    if low_ttd > 3:
        if is_saturated(mid_ttd) and is_moist(upper_ttd):
            options = [
                "Although the red and green lines are far apart near the surface, they run close together higher up, suggesting moisture is trapped aloft while the surface remains dry. Precipitation may form but evaporate before reaching the ground.",
                "Mid and upper-level saturation is visible with tight red-green spacing, while a wide gap near the surface implies a dry sub-cloud layer—likely resulting in virga.",
                "While cloud development is supported aloft, the dry lower layer may prevent rainfall from reaching the ground, leading to elevated precipitation or evaporation."
            ]
            desc += random.choice(options) + "\n"
            if mid_rot >= 30 and upper_rot >= 30:
                options = [
                    "Wind barbs turning clockwise with height suggest veering, a sign of favorable wind shear that may help organize convection.",
                    "The clockwise rotation of wind barbs in the mid and upper layers points to directional shear, enhancing storm potential.",
                    "Layered veering of wind barbs supports the development of structured updrafts in the presence of moisture aloft."
                ]
                desc += random.choice(options) + "\n"
        elif is_moist(mid_ttd) and is_moist(upper_ttd):
            options = [
                "Moisture is visible in the mid and upper levels, with wide spacing of the red and green lines near the surface. This suggests rain may form aloft but evaporate on descent.",
                "The diagram shows a moist upper atmosphere above a dry boundary layer, hinting at limited surface precipitation despite cloud formation.",
                "Though the upper layers are moist, the dry lower atmosphere likely reduces rainfall efficiency at the surface."
            ]
            desc += random.choice(options) + "\n"
        elif is_moist(mid_ttd):
            options = [
                "The red and green lines are closer together in the mid-levels, indicating possible mid-level cloud development.",
                "Moisture is concentrated in the mid-troposphere, which may produce layered clouds without surface rain.",
                "The middle layers appear moist, favoring mid-altitude cloud formation."
            ]
            desc += random.choice(options) + "\n"
        elif is_moist(upper_ttd):
            options = [
                "Moisture is confined to the upper levels, with red and green lines near each other only at high altitudes. Expect high clouds or cirrus formations.",
                "The sounding shows potential for cirriform cloud development due to upper-level moisture.",
                "Only the upper troposphere is moist, favoring thin high-level cloud layers."
            ]
            desc += random.choice(options) + "\n"
        else:
            options = [
                "The wide separation of red and green lines throughout the profile indicates a dry atmosphere with little cloud potential.",
                "Dry conditions dominate all layers, suggesting limited or no cloud development.",
                "The Skew-T shows an unsupportive environment for convection, with insufficient moisture throughout."
            ]
            desc += random.choice(options) + "\n"
    else:
        if is_saturated(mid_ttd) and is_moist(upper_ttd):
            options = [
                "Red and green lines lie close together from the surface through the mid and upper levels, indicating widespread moisture and favorable conditions for precipitation.",
                "A saturated lower atmosphere with moist layers above supports cloud growth and possible rainfall.",
                "Consistent moisture across layers, shown by the tight spacing of temperature and dewpoint lines, suggests high rain potential."
            ]
            desc += random.choice(options) + "\n"
            if mid_rot >= 30 and upper_rot >= 30:
                options = [
                    "Wind barbs veering with height add favorable shear to an already moist profile, enhancing storm development.",
                    "Clockwise-turning wind barbs in mid and upper levels indicate directional shear, supporting convective organization.",
                    "The combination of vertical moisture and veering wind profile may foster structured updrafts and convective storms."
                ]
                desc += random.choice(options) + "\n"
        elif is_moist(mid_ttd) and is_moist(upper_ttd):
            options = [
                "Moist conditions from the surface through the upper atmosphere favor widespread cloud formation and precipitation.",
                "Red and green lines track closely through much of the profile, supporting a deep, moist column ideal for rainfall.",
                "Ample moisture throughout the sounding suggests a supportive environment for cloud depth and precipitation."
            ]
            desc += random.choice(options) + "\n"
        elif is_moist(mid_ttd):
            options = [
                "Moisture in the lower and mid-levels is evident from the closely spaced red and green lines, which may support low-to-mid level cloud formation.",
                "Boundary-layer and mid-level humidity may lead to stratiform cloudiness.",
                "The lower part of the sounding is sufficiently moist to develop layered clouds."
            ]
            desc += random.choice(options) + "\n"
        elif is_moist(upper_ttd):
            options = [
                "The surface and upper levels appear moist, but a dry mid-layer introduces a split moisture profile, potentially forming clouds at separated heights.",
                "Upper and lower moisture without mid-level support may result in scattered clouds at various levels.",
                "Split red-green closeness in lower and upper levels suggests stacked cloud decks without vertical connection."
            ]
            desc += random.choice(options) + "\n"
        else:
            if is_saturated(low_ttd):
                options = [
                    "Red and green lines nearly touching near the surface point to boundary-layer saturation, favoring fog or low stratus development.",
                    "A saturated near-surface layer suggests potential for drizzle or low cloud decks.",
                    "Moisture trapped at the surface may lead to shallow cloud cover or misty conditions."
                ]
                desc += random.choice(options) + "\n"
            else:
                options = [
                    "Only the lower levels show some moisture, evident in the partial overlap of the red and green lines. Shallow stratiform clouds are possible.",
                    "The boundary layer holds some humidity, which may support low-level cloud development.",
                    "Limited low-level moisture may result in isolated, shallow clouds near the surface."
                ]
                desc += random.choice(options) + "\n"

    return desc

def layer_description(tdd, rotation, label):
    if tdd > 3:
        moisture = "Dry"
    elif tdd > 1:
        moisture = "Moist"
    else:
        moisture = "Saturated"

    if rotation > 30:
        wind = "Wind Veering"
    elif rotation < -30:
        wind = "Wind Backing"
    else:
        wind = ""

    line = f"{label} layer: {moisture}"
    if wind:
        line += f", {wind}"
    return line + "\n"

# === Sample Data Loop for Fine-Tuning JSONL ===
def generate_prompt_response_vlm(row):
    system_message = "You are a weather forecaster analyzing atmospheric soundings shown in a Skew-T log-P diagram\nThe diagram uses a logarithmic vertical pressure axis (hPa), so pressure layers are not evenly spaced. Use the following visual anchors:\nLower layer (1000-850 hPa): This is located in the bottom quarter of the diagram, close to the surface. It represents the boundary layer where surface temperature, dew point, and CIN typically appear.\nMid layer (850-500 hPa): Appears in the second quarter from the bottom of the plot. This region often contains most of the CAPE and developing updrafts.\nUpper layer (500-250 hPa): This is around the middle third of the diagram, despite covering less pressure range. This layer includes the top of convection (EL), cirrus clouds, and upper-level wind shear.\nUse the following visual references:\nRed line: temperature profile\nGreen line: dew point temperature\nBlue shaded area: CAPE (Convective Available Potential Energy)\nYellow shaded area: CIN (Convective Inhibition)\nWind barbs: on the right-hand side, changing with height\nKey interpretation rules:\nWhere the red and green lines are close, the layer is moist; far apart implies dryness.\nThe LFC is the bottom of the blue area; the EL is the top of the blue area.\nClockwise turning wind barbs with height suggest veering (warm air advection); counterclockwise suggests backing."

    user_prompt = "Below is a brief summary of the atmospheric sounding:\n\n"
    if row['cape'] > 0:
        user_prompt += f"CAPE: {row['cape']:.0f} J/kg\n"
    if row['cin'] > 0:
        user_prompt += f"CIN: {row['cin']:.0f} J/kg\n"
    #if row['lcl'] > 0:
    #    user_prompt += f"LCL: {row['lcl']:.0f} hPa\n"
    if row['lfc'] > 0:
        user_prompt += f"LFC: {row['lfc']:.0f} hPa\n"
    if row['el'] > 0:
        user_prompt += f"EL: {row['el']:.0f} hPa\n"

    user_prompt += layer_description(row['avg_ttd_low'], row['rotation_low'], "Lower")
    user_prompt += layer_description(row['avg_ttd_mid'], row['rotation_mid'], "Mid")
    user_prompt += layer_description(row['avg_ttd_upper'], row['rotation_upper'], "Upper")

    user_prompt += "\n\nPlease describe the atmospheric profile based on the provided Skew-T log-P diagram and the summary. Reason carefully, and conclude with a precipitation probability category: Low, Moderate, High, or Very High."
    #user_prompt = "Please describe the atmospheric profile based on the provided Skew-T log-P diagram. Reason carefully, and conclude with a precipitation probability category: Low, Moderate, High, or Very High."

    response = ""
    if row['cape'] > 0:
        response += describe_cape_cin(row['cape'], row['cin'])
    #if row['lcl'] > 0:
    #    response += describe_lcl(row['lcl'])
    if row['lfc'] > 0:
        response += describe_lfc_el(row['lfc'], row['el'], row['cape'])

    response += describe_moisture_wind(row['avg_ttd_low'], row['avg_ttd_mid'], row['avg_ttd_upper'], row['rotation_low'], row['rotation_mid'], row['rotation_upper'])

    response += describe_conclusion(row['rn3h'])

    return system_message.strip(), user_prompt.strip(), response.strip()

def generate_prompt_response_lm(row):
    system_message  = "Please provide a weather analysis based on this atmospheric sounding. Reason carefully and explain your conclusions.\n"
    system_message += "Based on your reasoning, please provide the predicted precipitation probability category (Low, Moderate, High, or Very High).\n"

    user_prompt = ""
    if row['cape'] > 0:
        user_prompt += f"CAPE: {row['cape']:.0f} J/kg\n"
    if row['cin'] > 0:
        user_prompt += f"CIN: {row['cin']:.0f} J/kg\n"
    #if row['lcl'] > 0:
    #    user_prompt += f"LCL: {row['lcl']:.0f} hPa\n"
    if row['lfc'] > 0:
        user_prompt += f"LFC: {row['lfc']:.0f} hPa\n"
    if row['el'] > 0:
        user_prompt += f"EL: {row['el']:.0f} hPa\n"

    user_prompt += layer_description(row['avg_ttd_low'], row['rotation_low'], "Lower")
    user_prompt += layer_description(row['avg_ttd_mid'], row['rotation_mid'], "Mid")
    user_prompt += layer_description(row['avg_ttd_upper'], row['rotation_upper'], "Upper")

    response = ""
    if row['cape'] > 0:
        response += describe_cape_cin_lm(row['cape'], row['cin'])
    #if row['lcl'] > 0:
    #    response += describe_lcl_lm(row['lcl'])
    if row['lfc'] > 0:
        response += describe_lfc_el_lm(row['lfc'], row['el'], row['cape'])

    response += describe_moisture_wind_lm(row['avg_ttd_low'], row['avg_ttd_mid'], row['avg_ttd_upper'], row['rotation_low'], row['rotation_mid'], row['rotation_upper'])

    response += describe_conclusion(row['rn3h'])

    return system_message.strip(), user_prompt.strip(), response.strip()

def generate_qa(row):
    system_message = "You are a weather forecaster analyzing atmospheric soundings shown in a Skew-T log-P diagram\nThe diagram uses a logarithmic vertical pressure axis (hPa), so pressure layers are not evenly spaced. Use the following visual anchors:\nLower layer (1000-850 hPa): This is located in the bottom quarter of the diagram, close to the surface. It represents the boundary layer where surface temperature, dew point, and CIN typically appear.\nMid layer (850-500 hPa): Appears in the second quarter from the bottom of the plot. This region often contains most of the CAPE and developing updrafts.\nUpper layer (500-250 hPa): This is around the middle third of the diagram, despite covering less pressure range. This layer includes the top of convection (EL), cirrus clouds, and upper-level wind shear.\nUse the following visual references:\nRed line: temperature profile\nGreen line: dew point temperature\nBlue shaded area: CAPE (Convective Available Potential Energy)\nYellow shaded area: CIN (Convective Inhibition)\nWind barbs: on the right-hand side, changing with height\nKey interpretation rules:\nWhere the red and green lines are close, the layer is moist; far apart implies dryness.\nThe LFC is the bottom of the blue area; the EL is the top of the blue area.\nClockwise turning wind barbs with height suggest veering (warm air advection); counterclockwise suggests backing."

    user_prompt = "Your task is to identify basic thermodynamic and wind profile features based on visual inspection.\n"

    questions = []
    answers = []

    # lower layer
    if row['avg_ttd_low'] > 3:
        question = user_prompt + "Is the lower layer (1000-850 hPa, the bottom quarter of the diagram) dry, moist, or saturated? Please provide both explanation and answer."
        answer = "The temperature (red) and dew point (green) lines are far apart in the lower layer (1000-850 hPa, the bottom quarter of the diagram), indicating low relative humidity. Therefore, the lower layer is dry."

        questions.append(question.strip())
        answers.append(answer.strip())
    elif row['avg_ttd_low'] > 1:
        question = user_prompt + "Is the lower layer (1000-850 hPa, the bottom quarter of the diagram) dry, moist, or saturated? Please provide both explanation and answer."
        answer = "The temperature (red) and dew point (green) lines are moderately close in the lower layer (1000-850 hPa, the bottom quarter of the diagram), suggesting fair humidity. Therefore, the lower layer is moist."

        questions.append(question.strip())
        answers.append(answer.strip())
    else:
        question = user_prompt + "Is the lower layer (1000-850 hPa, the bottom quarter of the diagram) dry, moist, or saturated? Please provide both explanation and answer."
        answer = "The temperature (red) and dew point (green) lines nearly overlap in the lower layer (1000-850 hPa, the bottom quarter of the diagram), indicating high relative humidity. Therefore, the lower layer is saturated."

        questions.append(question.strip())
        answers.append(answer.strip())

    # mid layer
    if row['avg_ttd_mid'] > 3:
        question = user_prompt + "Is the mid layer (850-500 hPa, the second quarter from the bottom) dry, moist, or saturated? Please provide both explanation and answer."
        answer = "The temperature (red) and dew point (green) lines are far apart in the mid layer (850-500 hPa, the second quarter from the bottom), indicating low relative humidity. Therefore, the mid layer is dry."

        questions.append(question.strip())
        answers.append(answer.strip())
    elif row['avg_ttd_mid'] > 1:
        question = user_prompt + "Is the mid layer (850-500 hPa, the second quarter from the bottom) dry, moist, or saturated? Please provide both explanation and answer."
        answer = "The temperature (red) and dew point (green) lines are moderately close in the mid layer (850-500 hPa, the second quarter from the bottom), suggesting fair humidity. Therefore, the mid layer is moist."

        questions.append(question.strip())
        answers.append(answer.strip())
    else:
        question = user_prompt + "Is the mid layer (850-500 hPa, the second quarter from the bottom) dry, moist, or saturated? Please provide both explanation and answer."
        answer = "The temperature (red) and dew point (green) lines nearly overlap in the mid layer (850-500 hPa, the second quarter from the bottom), indicating high relative humidity. Therefore, the mid layer is saturated."

        questions.append(question.strip())
        answers.append(answer.strip())

    # upper layer
    if row['avg_ttd_upper'] > 3:
        question = user_prompt + "Is the upper layer (500-250 hPa, the middle third of the diagram) dry, moist, or saturated? Please provide both explanation and answer."
        answer = "The temperature (red) and dew point (green) lines are far apart in the upper layer (500-250 hPa, the middle third of the diagram), indicating low relative humidity. Therefore, the upper layer is dry."

        questions.append(question.strip())
        answers.append(answer.strip())
    elif row['avg_ttd_upper'] > 1:
        question = user_prompt + "Is the upper layer (500-250 hPa, the middle third of the diagram) dry, moist, or saturated? Please provide both explanation and answer."
        answer = "The temperature (red) and dew point (green) lines are moderately close in the upper layer (500-250 hPa, the middle third of the diagram), suggesting fair humidity. Therefore, the upper layer is moist."

        questions.append(question.strip())
        answers.append(answer.strip())
    else:
        question = user_prompt + "Is the upper layer (500-250 hPa, the middle third of the diagram) dry, moist, or saturated? Please provide both explanation and answer."
        answer = "The temperature (red) and dew point (green) lines nearly overlap in the upper layer (500-250 hPa, the middle third of the diagram), indicating high relative humidity. Therefore, the upper layer is saturated."

        questions.append(question.strip())
        answers.append(answer.strip())

    # lower layer - wind
    if row['rotation_low'] > 30:
        question = user_prompt + "Is the lower level (1000-850 hPa, the bottom quarter of the diagram) wind veering or backing? Please provide both explanation and answer."
        answer = "The wind barbs on the right-hand side turn clockwise from surface to 850 hPa, which indicates veering. Therefore, the lower layer wind is veering."

        questions.append(question.strip())
        answers.append(answer.strip())
    elif row['rotation_low'] < -30:
        question = user_prompt + "Is the lower level (1000-850 hPa, the bottom quarter of the diagram) wind veering or backing? Please provide both explanation and answer."
        answer = "The wind barbs on the right-hand side turn counterclockwise from surface to 850 hPa, which indicates backing. Therefore, the lower layer wind is backing."

        questions.append(question.strip())
        answers.append(answer.strip())

    # mid layer - wind
    if row['rotation_mid'] > 30:
        question = user_prompt + "Is the mid level (850-500 hPa, the second quarter from the bottom) wind veering or backing? Please provide both explanation and answer."
        answer = "The wind barbs on the right-hand side turn clockwise from 850 to 500 hPa, which indicates veering. Therefore, the mid layer wind is veering."

        questions.append(question.strip())
        answers.append(answer.strip())
    elif row['rotation_mid'] < -30:
        question = user_prompt + "Is the mid level (850-500 hPa, the second quarter from the bottom) wind veering or backing? Please provide both explanation and answer."
        answer = "The wind barbs on the right-hand side turn counterclockwise from 850 to 500 hPa, which indicates backing. Therefore, the mid layer wind is backing."

        questions.append(question.strip())
        answers.append(answer.strip())

    # upper layer - wind
    if row['rotation_upper'] > 30:
        question = user_prompt + "Is the upper level (500-250 hPa, the middle third of the diagram) wind veering or backing? Please provide both explanation and answer."
        answer = "The wind barbs on the right-hand side turn clockwise from 500 to 250 hPa, which indicates veering. Therefore, the upper layer wind is veering."

        questions.append(question.strip())
        answers.append(answer.strip())
    elif row['rotation_upper'] < -30:
        question = user_prompt + "Is the upper level (500-250 hPa, the middle third of the diagram) wind veering or backing? Please provide both explanation and answer."
        answer = "The wind barbs on the right-hand side turn counterclockwise from 500 to 250 hPa, which indicates backing. Therefore, the upper layer wind is backing."

        questions.append(question.strip())
        answers.append(answer.strip())

    # CAPE
    if row['cape'] > 100 and row['cape'] < 500:
        question = user_prompt + "Is the CAPE in the diagram strong, moderate, or weak? Please provide both explanation and answer."
        answer = "The blue CAPE region is shallow and narrow, often confined to a thin layer, which indicates limited buoyant energy and weak potential for deep convective development. Therefore, the CAPE is weak."

        questions.append(question.strip())
        answers.append(answer.strip())
    elif row['cape'] >= 500 and row['cape'] < 1000:
        question = user_prompt + "Is the CAPE in the diagram strong, moderate, or weak? Please provide both explanation and answer."
        answer = "The blue CAPE area is present and spans a modest vertical range, which suggests some atmospheric instability, supportive of convection, but not extreme. Therefore, the CAPE is moderate."

        questions.append(question.strip())
        answers.append(answer.strip())
    elif row['cape'] >= 1000:
        question = user_prompt + "Is the CAPE in the diagram strong, moderate, or weak? Please provide both explanation and answer."
        answer = "The blue CAPE area is deep and wide, extending through a significant portion of the troposphere, which indicates strong instability and potential for vigorous convection or thunderstorms. Therefore, the CAPE is strong."

        questions.append(question.strip())
        answers.append(answer.strip())

    # CIN
    if row['cin'] > 150:
        question = user_prompt + "Is the CIN in the diagram strong, moderate, or weak? Please provide both explanation and answer."
        answer = "A significant layer of yellow CIN area is present, suppressing surface-based convection. Therefore, the CIN is strong."

        questions.append(question.strip())
        answers.append(answer.strip())
    elif row['cin'] > 75:
        question = user_prompt + "Is the CIN in the diagram strong, moderate, or weak? Please provide both explanation and answer."
        answer = "The noticeable yellow CIN area is present, requiring stronger lifting or heating to overcome the stable layer and reach the level of free convection. Therefore, the CIN is moderate."

        questions.append(question.strip())
        answers.append(answer.strip())
    elif row['cin'] > 25:
        question = user_prompt + "Is the CIN in the diagram strong, moderate, or weak? Please provide both explanation and answer."
        answer = "The noticeable yellow CIN area is minimal, providing only slight resistance to rising air parcels. Therefore, the CIN is weak."

        questions.append(question.strip())
        answers.append(answer.strip())
 
    # LFC & EL
    if row['cape'] >= 500:
        # LFC
        if row['lfc'] > 900:
            question = user_prompt + "Is the LFC in the diagram high, moderate, or low? Please provide both explanation and answer."
            answer = "The base of the blue region is near the surface, meaning convection can initiate easily with minimal lifting. Therefore, the LFC is low."

            questions.append(question.strip())
            answers.append(answer.strip())
        elif row['lfc'] > 700:
            question = user_prompt + "Is the LFC in the diagram high, moderate, or low? Please provide both explanation and answer."
            answer = "The base of the blue region is at a mid-level pressure, suggesting that some lifting or heating is needed for parcels to reach the level of free convection. Therefore, the LFC is moderate."

            questions.append(question.strip())
            answers.append(answer.strip())
        elif row['lfc'] > 0:
            question = user_prompt + "Is the LFC in the diagram high, moderate, or low? Please provide both explanation and answer."
            answer = "The base of the blue region is elevated,  meaning surface parcels require significant lifting to become buoyant. Therefore, the LFC is high."

            questions.append(question.strip())
            answers.append(answer.strip())

        # EL
        if row['el'] > 0 and row['el'] <= 250:
            question = user_prompt + "Is the EL in the diagram high, moderate, or low? Please provide both explanation and answer."
            answer = "The top of the blue region is above 250 hPa, indicating deep convection is possible with strong updrafts and significant storm growth potential. Therefore, the EL is high."

            questions.append(question.strip())
            answers.append(answer.strip())
        elif row['el'] > 0 and row['el'] <= 400:
            question = user_prompt + "Is the EL in the diagram high, moderate, or low? Please provide both explanation and answer."
            answer = "The top of the blue region is between 250 and 400 hPa, which supports moderate convective cloud depth and updraft strength. Therefore, the EL is moderate."

            questions.append(question.strip())
            answers.append(answer.strip())
        elif row['el'] > 0:
            question = user_prompt + "Is the EL in the diagram high, moderate, or low? Please provide both explanation and answer."
            answer = "The top of the blue region is below 400 hPa, indicating shallow convection and limited vertical development. Therefore, the EL is low."

            questions.append(question.strip())
            answers.append(answer.strip())

    return system_message.strip(), questions, answers


def draw_skew(obj):
    data = obj['data']
    cape = {"value": obj['cape'], "polygon": obj['cape_polygon']}
    cin  = {"value": obj['cin'], "polygon": obj['cin_polygon']}

    pressure = np.array([d.get('pres') for d in data], dtype=float)
    temperature = np.array([d.get('ta') for d in data], dtype=float)
    dewpoint = np.array([d.get('td') for d in data], dtype=float)
    u = np.array([d.get('u') for d in data], dtype=float)
    v = np.array([d.get('v') for d in data], dtype=float)

    # === Set up Skew-T axes ===
    fig, ax = plt.subplots(figsize=(5.5,5))

    # Log-pressure axis (y-axis)
    p_top, p_base = 90, 1050
    ax.set_yscale('log')
    ax.set_ylim(p_base, p_top)
    #ax.invert_yaxis()
    ax.set_ylabel('Pressure (hPa)')
    yticks = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]

    # Optional: custom tick labels (e.g., with 'hPa' suffix)
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.set_yticklabels([f"{p}" for p in yticks])
    ax.tick_params(axis='y', which='minor', left=False)
    ax.set_yticks(yticks)

    # Remove axis spines (outline box)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Skewed temperature axis (x-axis)
    T_min, T_max = -50, 50
    ax.set_xlim(T_min, T_max)
    ax.set_xlabel('Temperature (°C)')
    xticks = [-50,-40,-30,-20,-10,0,10,20,30,40]
    ax.set_xticks(xticks)

    fig_dpi = fig.dpi                      # default is usually 100 dpi

    # Height in pixels
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    height_in_inches = bbox.height
    height_px = height_in_inches * fig_dpi
    width_in_inches = bbox.width
    width_px = width_in_inches * fig_dpi

    # Skewed x-axis
    def y_transform(pressure, h=height_px, w=width_px, p_top=90, p_base=1050, T_min=-50, T_max=50):
        # log-scale y-axis transformation
        return h * (np.log10(pressure) - np.log10(p_top)) / (np.log10(p_base) - np.log10(p_top)) * (T_max - T_min)/w

    def skew_transform(temp, pressure, skew_deg=55, p_base=1050):
        skew_rad = np.deg2rad(skew_deg)
        return temp + (y_transform(p_base, h=height_px) - y_transform(pressure, h=height_px)) / np.tan(skew_rad)

    temps = np.arange(-120, 50, 10)
    skew_deg=55
    skew_rad = np.deg2rad(skew_deg)
    for d in temps:
        x1 = d + (y_transform(p_base, h=height_px) - y_transform(p_top, h=height_px)) / np.tan(skew_rad)
        x2 = d
        y1 = p_top
        y2 = p_base

        linewidth = 0.5
        color = 'gray'

        if d % 10 == 0:
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, linestyle='--')
        else:
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, linestyle='--')


    # Plot temperature and dewpoint
    skew_t = skew_transform(temperature, pressure)
    skew_td = skew_transform(dewpoint, pressure)
    ax.plot(skew_t, pressure, 'r', label='Temperature')
    ax.scatter(skew_t, pressure, color='red', s=10)
    ax.plot(skew_td, pressure, 'g', label='Dewpoint')
    ax.scatter(skew_td, pressure, color='green', s=10)

    if cape["value"] > 0:
        points_temp = [d["ta"] for d in cape["polygon"]]
        points_pres = [d["pres"] for d in cape["polygon"]]
        skew_points = skew_transform(points_temp, points_pres)
        points = list(zip(skew_points, points_pres))
        poly_patch = Polygon(points, closed=True, edgecolor='black', facecolor='blue', alpha=0.5, label='CAPE')
        ax.add_patch(poly_patch)

    if cin["value"] > 0:
        points_temp = [d["ta"] for d in cin["polygon"]]
        points_pres = [d["pres"] for d in cin["polygon"]]
        skew_points = skew_transform(points_temp, points_pres)
        points = list(zip(skew_points, points_pres))
        poly_patch = Polygon(points, closed=True, edgecolor='black', facecolor='orange', alpha=0.5, label='CIN')
        ax.add_patch(poly_patch)

    # Draw barbs at right edge
    barb_x = T_max - 10

    # Constant x-location for barbs
    x_barb = np.full_like(pressure, barb_x)
    mask = np.isfinite(u)
    ax.barbs(x_barb[mask], pressure[mask], u[mask], v[mask], length=6, linewidth=0.8)

    # === Cosmetic ===
    ax.grid(True, axis='y', which='major', linestyle='--', lw=0.5)
    ax.legend(loc='lower left', fontsize=8)
    plt.title("Skew-T log-P Diagram")
    plt.tight_layout()
    #plt.show()
    #fname = f"./valid_imgs/skew_{obj['stn_id']}_{obj['case']:%Y%m%d%H}.png"
    #print(fname)
    #plt.savefig(fname)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()  # Close the figure to free memory

    img = Image.open(buf).convert("RGB")
    buf.close()
    return img

def calculate_veering_backing(layer):
    #print(layer)
    start = layer.iloc[0]  # Highest pressure (lowest level)
    end = layer.iloc[-1]   # Lowest pressure (highest level)
    delta_u = end['u'] - start['u']
    delta_v = end['v'] - start['v']
    angle_change = np.arctan2(delta_v, delta_u) * (180 / np.pi)  # Angle in degrees
    # Normalize angle between -180 and 180
    if angle_change > 180:
        angle_change -= 360
    if angle_change < -180:
        angle_change += 360
    return angle_change

def get_data():
    data = []

    with (Path("./skew_data_2024060100_2024083112.jsonl")).open() as f:
        for line in f:
            data.append(json.loads(line))

    d = random.sample(range(len(data)), k=1)

    return data[d[0]]

def process_data(raw_data):
    data = raw_data['data']
    lcl = calc_lcl(data[0]["pres"], data[0]["ta"], data[0]["td"])
    ccl = calc_ccl(data)
    if ccl["p"] >= lcl["p"]:
        lfc = {}
        ccl["p"] = lcl["p"]
        ccl["t"] = lcl["t"]
        lfc["p"] = lcl["p"]
        lfc["t"] = lcl["t"]
    else:
        lfc = calc_lfc(data, lcl)

    el = calc_el(data, lcl, ccl, lfc)
    cape = calc_cape(data, lcl, ccl, lfc, el)
    cin = calc_cin(data, lcl, ccl, lfc, el)

    df = pd.DataFrame(data)

    # Calculate T - Td
    df['ttd'] = df['ta'] - df['td']

    # Define level ranges (in hPa)
    low_level = (df['pres'] <= 1000) & (df['pres'] >= 850) & (df['pres'] != 'SFC')
    mid_level = (df['pres'] < 850) & (df['pres'] >= 500) & (df['pres'] != 'SFC')
    upper_level = (df['pres'] < 500) & (df['pres'] >= 250) & (df['pres'] != 'SFC')

    # Calculate ttd averages
    avg_ttd_low = df.loc[low_level, 'ttd'].mean()
    avg_ttd_mid = df.loc[mid_level, 'ttd'].mean()
    avg_ttd_upper = df.loc[upper_level, 'ttd'].mean()

    # Calculate wind-rotation averages
    rotation_low = calculate_veering_backing(df.loc[low_level])
    rotation_mid = calculate_veering_backing(df.loc[mid_level])
    rotation_upper = calculate_veering_backing(df.loc[upper_level])

    processed_data = {"stn_id": raw_data['stn_id'], "case": raw_data['case'], "data": data, "rn3h": raw_data['rn3h'],
                      "lcl": lcl['p'], "lfc": lfc['p'], "el": el['p'], "cape": cape['value'], "cin": cin['value'], "cape_polygon": cape['polygon'], "cin_polygon": cin['polygon'],
                      "avg_ttd_low": avg_ttd_low, "avg_ttd_mid": avg_ttd_mid, "avg_ttd_upper": avg_ttd_upper, "rotation_low": rotation_low, "rotation_mid": rotation_mid, "rotation_upper": rotation_upper}

    return processed_data

def main():
    raw_data = get_data()
    processed_data = process_data(raw_data)
    image = draw_skew(processed_data)
    #image.show()
    
    # VLM text generator
    system_message, user_prompt, response = generate_prompt_response_vlm(processed_data) 
    #print(system_message)
    #print(user_prompt)
    #print(response)

    # LM text generator
    system_message, user_prompt, response = generate_prompt_response_lm(processed_data) 
    #print(system_message)
    #print(user_prompt)
    #print(response)

if __name__ == "__main__":
    sys.exit(main())
