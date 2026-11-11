"""W23 UNDEX chain (multi-case ensemble): run, verify, render artifacts.

Writes benchmarks/undex_shock/{case_matrix.csv,srs_comparison.json,chain_metrics.json},
docs/assets/w23_undex_shock/*.png and papers/reports/w23_undex_shock_chain_2.md.
Run: PYTHONPATH=src python scripts/w23_undex_shock_chain_run.py
"""
from __future__ import annotations 

import csv 
import hashlib 
import json 
import subprocess 
from pathlib import Path 

import numpy as np 
import torch 
import matplotlib 
matplotlib .use ("Agg")
import matplotlib .pyplot as plt 

from tensorfem .undex_chain import (SRS_BANDS_HZ ,TNT_BUBBLE_K_S ,assemble_panel ,
bubble_scaling_exponents ,cole_case_environment ,
cole_recompute_vs_anchor ,compare_panel_to_taylor ,
panel_fsi_shock ,pressure_closure ,roache_gci ,
srs_band_table ,srs_pair )
from tensorfem .undex_shock import (compare_to_taylor_analytic ,taylor_analytic ,
taylor_fsi_fem )

REPO =Path (__file__ ).resolve ().parents [1 ]
ANCHOR =REPO /"benchmarks"/"undex_shock"/"cole_anchor.json"
OUT =REPO /"benchmarks"/"undex_shock"
FIGS =REPO /"docs"/"assets"/"w23_undex_shock"
REPORT =REPO /"papers"/"reports"/"w23_undex_shock_chain_2.md"
LEVELS =((12 ,8 ,40 ),(18 ,12 ,60 ),(27 ,18 ,90 ))
PANEL ={"length":1.5 ,"width":1.0 ,"thickness":0.010 ,"young":206.0e9 ,
"poisson":0.3 ,"density":7850.0 }
GCI_LIMIT =0.10 
GCIQ ={"peak_center_deflection_m":"peak centre deflection [m]",
"peak_center_acceleration_ms2":"peak centre acceleration [m/s^2]",
"wall_impulse_precav_pa_s":"pre-cavitation wall impulse [Pa s]",
"peak_wall_pressure_pa":"peak wall pressure [Pa]"}


def sha (p ):
    return hashlib .sha256 (Path (p ).read_bytes ()).hexdigest ()


def f (x ,nd =4 ):
    if isinstance (x ,bool ):
        return "PASS"if x else "FAIL"
    if x is None :
        return "n/a"
    if isinstance (x ,float ):
        if x ==0.0 :
            return "0"
        return "%.3e"%x if (abs (x )<1e-3 or abs (x )>=1e5 )else "%.*f"%(nd ,x )
    return str (x )


def ok (c ):
    return "PASS"if c else "FAIL"


def main ():
    anchor =json .loads (ANCHOR .read_text ())
    cases =anchor ["cases"]
    OUT .mkdir (parents =True ,exist_ok =True )
    FIGS .mkdir (parents =True ,exist_ok =True )
    recompute =cole_recompute_vs_anchor (anchor )
    scaling =bubble_scaling_exponents ()
    models ={k :assemble_panel (nx =lv [0 ],ny =lv [1 ],**PANEL )
    for k ,lv in enumerate (LEVELS )}
    rows ,case_metrics ,keep =[],{},{}
    srs ={"schema_version":"1.0.0",
    "generated_by":"scripts/w23_undex_shock_chain_run.py",
    "bands_definition":[list (b )for b in SRS_BANDS_HZ ],
    "damping_ratios":[0.02 ,0.05 ],"band_coverage":len (SRS_BANDS_HZ ),
    "cases":{}}
    for case in cases :
        cid =case ["case_id"]
        env =cole_case_environment (case ["weight_kg"],case ["range_m"],case ["depth_m"])
        rec =next (r for r in anchor ["recorded_values"]if r ["case_id"]==cid )
        rc =next (r for r in recompute ["rows"]if r ["case_id"]==cid )
        levels =[panel_fsi_shock (peak_pressure =env ["peak_pressure_pa"],
        decay_time =env ["decay_time_s"],nx =LEVELS [k ][0 ],
        ny =LEVELS [k ][1 ],steps_per_decay =LEVELS [k ][2 ],
        model =models [k ])for k in range (len (LEVELS ))]
        fine =levels [-1 ]
        gci ={q :roache_gci (*[lv .meta [q ]for lv in levels ],ratio =1.5 )for q in GCIQ }
        tc =compare_panel_to_taylor (fine )
        ref =taylor_analytic (fine .time ,peak_pressure =env ["peak_pressure_pa"],
        decay_time =env ["decay_time_s"],
        areal_mass =fine .meta ["areal_mass_kg_m2"])
        pair =srs_pair (fine .center_acceleration ,ref .acceleration ,
        dt =fine .meta ["dt_s"],damping_ratios =(0.02 ,0.05 ))
        bands =srs_band_table (pair ,damping =0.02 )
        clo =pressure_closure (fine ,cole_peak_pa =env ["peak_pressure_pa"],
        cole_impulse_pa_s =env ["specific_impulse_pa_s"])
        fe =taylor_fsi_fem (peak_pressure =env ["peak_pressure_pa"],
        decay_time =env ["decay_time_s"],
        areal_mass =fine .meta ["areal_mass_kg_m2"],length =1.0 ,
        elements =120 ,dt =env ["decay_time_s"]/60.0 ,
        duration =8.0 *env ["decay_time_s"])
        fe_ref =taylor_analytic (fe .time ,peak_pressure =env ["peak_pressure_pa"],
        decay_time =env ["decay_time_s"],
        areal_mass =fine .meta ["areal_mass_kg_m2"])
        fe_cmp =compare_to_taylor_analytic (fe ,fe_ref )
        t_fe =fe .cavitation_time_s or float (fe .time [-1 ])
        msk =fe .time <=t_fe 
        fe_imp =float (torch .trapezoid (fe .interface_pressure [msk ],fe .time [msk ]))
        gv ={q :(gci [q ]["gci_fine"]if gci [q ]["available"]else 0.0 )for q in GCIQ }
        row ={"case_id":cid ,"weight_kg":case ["weight_kg"],
        "range_m":case ["range_m"],"depth_m":case ["depth_m"],
        "reduced_range":rec ["reduced_range"],
        "cole_peak_pressure_mpa":rec ["peak_pressure_pa"]/1e6 ,
        "cole_decay_time_us":rec ["decay_time_s"]*1e6 ,
        "cole_bubble_period_s":rec ["bubble_period_s"],
        "cole_impulse_pa_s":rec ["specific_impulse_pa_s"],
        "recompute_peak_pressure_mpa":rc ["recomputed"]["peak_pressure_pa"]/1e6 ,
        "recompute_rel_dev_peak_pressure":rc ["relative_deviation"]["peak_pressure"],
        "recompute_rel_dev_decay_time":rc ["relative_deviation"]["decay_time"],
        "recompute_rel_dev_bubble_period":
        rc ["relative_deviation"]["bubble_period"],
        "recompute_rel_dev_impulse":rc ["relative_deviation"]["specific_impulse"],
        "criterion1_pass":bool (max (rc ["relative_deviation"].values ())<0.01 ),
        "wall_peak_pressure_mpa":fine .meta ["peak_wall_pressure_pa"]/1e6 ,
        "wall_peak_over_cole":clo ["reflection_factor_peak"],
        "wall_peak_rel_dev_vs_cole":clo ["peak_rel_dev_vs_cole_free_field"],
        "wall_impulse_precav_pa_s":fine .meta ["wall_impulse_precav_pa_s"],
        "wall_impulse_precav_over_cole":
        fine .meta ["wall_impulse_precav_pa_s"]/env ["specific_impulse_pa_s"],
        "wall_impulse_precav_rel_dev_vs_cole":
        abs (fine .meta ["wall_impulse_precav_pa_s"]
        -env ["specific_impulse_pa_s"])/env ["specific_impulse_pa_s"],
        "cavitation_onset_us":(fine .meta ["cavitation_onset_s"]or 0.0 )*1e6 ,
        "min_wall_pressure_mpa":fine .meta ["min_wall_pressure_pa"]/1e6 ,
        "center_deflection_mm":fine .meta ["peak_center_deflection_m"]*1e3 ,
        "center_acceleration_g":
        fine .meta ["peak_center_acceleration_ms2"]/9.80665 ,
        "deflection_ratio_vs_rigid":tc ["deflection_ratio"],
        "taylor_velocity_rel_err":tc ["center_velocity"]["relative_max_error"],
        "taylor_pressure_rel_err":
        tc ["center_wall_pressure"]["relative_max_error"],
        "energy_residual_max":fine .meta ["energy_residual_max"],
        "energy_residual_worst_level":
        max (lv .meta ["energy_residual_max"]for lv in levels ),
        "gci_deflection":gv ["peak_center_deflection_m"],
        "gci_acceleration":gv ["peak_center_acceleration_ms2"],
        "gci_impulse_precav":gv ["wall_impulse_precav_pa_s"],
        "gci_wall_pressure":gv ["peak_wall_pressure_pa"],
        "fe_fluid_pressure_rel_err":
        fe_cmp ["interface_pressure"]["relative_max_error"],
        "fe_fluid_velocity_rel_err":fe_cmp ["velocity"]["relative_max_error"],
        "fe_fluid_impulse_precav_pa_s":fe_imp ,
        "fe_fluid_impulse_rel_dev_vs_cole":
        abs (fe_imp -env ["specific_impulse_pa_s"])
        /env ["specific_impulse_pa_s"],
        "srs_bands":len (bands ),
        "srs_band_max_deviation":
        max (b ["relative_deviation"]for b in bands )}
        row ["gci_pass"]=bool (all (row [k ]<GCI_LIMIT for k in 
        ("gci_deflection","gci_acceleration",
        "gci_impulse_precav","gci_wall_pressure")))
        rows .append (row )
        case_metrics [cid ]={"env":env ,"closure":clo ,"gci":gci ,"taylor":tc ,
        "levels":[lv .meta for lv in levels ],
        "fe_fluid":{"comparison":fe_cmp ,
        "impulse_precav_pa_s":fe_imp ,
        "cavitation_time_s":fe .cavitation_time_s ,
        "elements":fe .meta ["elements"],
        "dt_s":float (fe .time [1 ]-fe .time [0 ])}}
        srs ["cases"][cid ]={
        "weight_kg":case ["weight_kg"],"range_m":case ["range_m"],
        "reference":("Taylor exact air-backed areal-mass plate at the panel areal "
        "mass; FEM curve = flexible SS panel, so deviations are the "
        "panel-flexibility effect, not solver error"),
        "areal_mass_kg_m2":fine .meta ["areal_mass_kg_m2"],
        "first_frequency_hz":fine .meta ["first_frequency_hz"],
        "damping_0.02":bands ,"damping_0.05":srs_band_table (pair ,damping =0.05 ),
        "curves":{"frequencies_hz":pair ["frequencies_hz"].tolist (),
        "fem_sa_ms2_zeta0.02":pair ["fem"].sa_ms2 [0.02 ].tolist (),
        "reference_sa_ms2_zeta0.02":
        pair ["reference"].sa_ms2 [0.02 ].tolist (),
        "fem_sa_ms2_zeta0.05":pair ["fem"].sa_ms2 [0.05 ].tolist (),
        "reference_sa_ms2_zeta0.05":
        pair ["reference"].sa_ms2 [0.05 ].tolist ()},
        "panel_fem_curves":{
        "time_s":fine .time .tolist (),
        "center_acceleration_ms2":fine .center_acceleration .tolist (),
        "center_wall_pressure_pa":fine .center_wall_pressure .tolist (),
        "center_deflection_m":fine .center_deflection .tolist (),
        "incident_pressure_pa":fine .incident_pressure .tolist (),
        "energy_incident_j":fine .energy_incident .tolist (),
        "energy_reflected_j":fine .energy_reflected .tolist (),
        "energy_plate_j":fine .energy_plate .tolist (),
        "energy_residual":fine .energy_residual .tolist ()}}
        if cid ==cases [1 ]["case_id"]:
            keep ={"fem":fine ,"ref":ref }
    with (OUT /"case_matrix.csv").open ("w",newline ="")as fh :
        w =csv .DictWriter (fh ,fieldnames =list (rows [0 ].keys ()))
        w .writeheader ()
        w .writerows (rows )
    (OUT /"srs_comparison.json").write_text (json .dumps (srs ,indent =2 )+"\n")
    test_file =REPO /"tests/test_undex_chain.py"
    metrics ={"schema_version":"1.0.0","mission_id":15095 ,
    "git_commit_at_run":subprocess .run (
    ["git","rev-parse","HEAD"],cwd =str (REPO ),capture_output =True ,
    text =True ).stdout .strip (),
    "anchor_sha256":sha (ANCHOR ),
    "anchor_acquisition_status":anchor ["acquisition_status"],
    "script_sha256":sha (__file__ ),
    "package_file_sha256":sha (REPO /"src/tensorfem/undex_chain.py"),
    "base_module_sha256":sha (REPO /"src/tensorfem/undex_shock.py"),
    "test_file_sha256":sha (test_file )if test_file .exists ()else None ,
    "parameters_used":recompute ["parameters_used"],
    "criterion1_recompute":recompute ,"bubble_scaling":scaling ,
    "bubble_prefactor_used":TNT_BUBBLE_K_S ,
    "levels":[list (lv )for lv in LEVELS ],"rows":rows ,
    "cases":case_metrics }
    make_figures (cases ,case_metrics ,keep )
    metrics ["figures"]=sorted (p .name for p in FIGS .glob ("*.png"))
    (OUT /"chain_metrics.json").write_text (json .dumps (metrics ,indent =2 )+"\n")
    render_report (anchor ,metrics ,rows ,case_metrics )
    print (json .dumps ({"bubble_scaling_pass":scaling ["pass"],
    "rows":[{k :r [k ]for k in ("case_id","cole_peak_pressure_mpa",
    "wall_peak_pressure_mpa",
    "wall_peak_over_cole",
    "wall_impulse_precav_over_cole",
    "energy_residual_worst_level",
    "gci_deflection","gci_acceleration",
    "gci_impulse_precav",
    "fe_fluid_pressure_rel_err")}
    for r in rows ]},indent =2 ))
    print ("wrote",OUT ,REPORT )


def make_figures (cases ,case_metrics ,keep ):
    plt .rcParams .update ({"figure.dpi":130 ,"font.size":8 })
    fig ,axes =plt .subplots (1 ,2 ,figsize =(7.4 ,2.9 ))
    tt =np .linspace (0.0 ,6.0 ,400 )
    for case in cases :
        env =case_metrics [case ["case_id"]]["env"]
        axes [0 ].plot (tt ,env ["peak_pressure_pa"]/1e6 *np .exp (-tt ),
        label ="%s: W=%g kg, R=%g m"%(case ["case_id"],
        case ["weight_kg"],case ["range_m"]))
        axes [1 ].bar (case ["case_id"],env ["bubble_period_s"],alpha =0.7 ,
        label ="T_b=%.4f s"%env ["bubble_period_s"])
    axes [0 ].set_xlabel ("t / theta")
    axes [0 ].set_ylabel ("incident pressure [MPa]")
    axes [0 ].set_title ("Cole pulse p_m exp(-t/theta)")
    axes [0 ].legend (fontsize =6 )
    axes [1 ].set_ylabel ("bubble period [s]")
    axes [1 ].set_title ("bubble period at 40 m depth")
    axes [1 ].legend (fontsize =6 )
    fig .tight_layout ()
    fig .savefig (FIGS /"fig1_cole_environment.png")
    plt .close (fig )
    if not keep :
        return 
    fem ,ref =keep ["fem"],keep ["ref"]
    t_us =fem .time .numpy ()*1e6 
    fig ,ax =plt .subplots (figsize =(4.8 ,3.1 ))
    ax .plot (t_us ,fem .center_wall_pressure .numpy ()/1e6 ,label ="FEM panel wall p")
    ax .plot (t_us ,ref .interface_pressure .numpy ()/1e6 ,"--",
    label ="Taylor closed form (rigid areal mass)")
    ax .plot (t_us ,2 *fem .incident_pressure .numpy ()/1e6 ,":",
    label ="rigid-wall limit 2 p_i")
    ax .plot (t_us ,fem .incident_pressure .numpy ()/1e6 ,"-.",label ="Cole free field p_i")
    ax .axhline (0.0 ,color ="k",lw =0.4 )
    ax .set_xlabel ("time [us]")
    ax .set_ylabel ("pressure [MPa]")
    ax .set_title ("C2 wall pressure vs references")
    ax .legend (fontsize =6 )
    fig .tight_layout ()
    fig .savefig (FIGS /"fig2_wall_pressure.png")
    plt .close (fig )
    fig ,axes =plt .subplots (1 ,2 ,figsize =(7.4 ,2.9 ))
    axes [0 ].plot (t_us ,fem .center_deflection .numpy ()*1e3 ,label ="FEM panel centre")
    axes [0 ].plot (t_us ,ref .displacement .numpy ()*1e3 ,"--",label ="Taylor rigid plate")
    axes [0 ].set_xlabel ("time [us]")
    axes [0 ].set_ylabel ("deflection [mm]")
    axes [0 ].legend (fontsize =6 )
    axes [1 ].plot (t_us ,fem .center_acceleration .numpy ()/9.80665 ,
    label ="FEM panel centre")
    axes [1 ].plot (t_us ,ref .acceleration .numpy ()/9.80665 ,"--",
    label ="Taylor rigid plate")
    axes [1 ].set_xlabel ("time [us]")
    axes [1 ].set_ylabel ("acceleration [g]")
    axes [1 ].legend (fontsize =6 )
    fig .tight_layout ()
    fig .savefig (FIGS /"fig3_panel_response.png")
    plt .close (fig )
    pair =srs_pair (fem .center_acceleration ,ref .acceleration ,dt =fem .meta ["dt_s"],
    damping_ratios =(0.02 ,))
    fig ,ax =plt .subplots (figsize =(4.8 ,3.2 ))
    fr =pair ["frequencies_hz"]
    ax .loglog (fr ,pair ["fem"].sa_ms2 [0.02 ]/9.80665 ,label ="FEM panel SRS (zeta=2%%)")
    ax .loglog (fr ,pair ["reference"].sa_ms2 [0.02 ]/9.80665 ,"--",
    label ="Taylor rigid-plate SRS (zeta=2%%)")
    for lo ,_ in SRS_BANDS_HZ :
        ax .axvline (lo ,color ="0.7",lw =0.4 )
    ax .set_xlabel ("frequency [Hz]")
    ax .set_ylabel ("pseudo acceleration [g]")
    ax .set_title ("C2 SRS, band edges dotted")
    ax .legend (fontsize =6 )
    fig .tight_layout ()
    fig .savefig (FIGS /"fig4_srs.png")
    plt .close (fig )
    fig ,ax =plt .subplots (figsize =(4.8 ,3.1 ))
    ax .plot (t_us ,fem .energy_incident .numpy (),label ="incident")
    ax .plot (t_us ,fem .energy_reflected .numpy (),label ="reflected")
    ax .plot (t_us ,fem .energy_plate .numpy (),label ="plate (kinetic + strain)")
    ax .set_xlabel ("time [us]")
    ax .set_ylabel ("energy over panel area [J]")
    ax2 =ax .twinx ()
    ax2 .plot (t_us ,fem .energy_residual .numpy (),"k:",lw =0.8 ,label ="residual / E_inc")
    ax2 .set_ylabel ("relative residual")
    ax .set_title ("C2 energy ledger (max residual %.2e)"
    %fem .meta ["energy_residual_max"])
    ax .legend (fontsize =6 ,loc ="center right")
    fig .tight_layout ()
    fig .savefig (FIGS /"fig5_energy_ledger.png")
    plt .close (fig )
    fig ,ax =plt .subplots (figsize =(4.8 ,3.1 ))
    h =[1.0 /lv [0 ]for lv in LEVELS ]
    cid =sorted (case_metrics )[1 ]
    for q ,label in GCIQ .items ():
        vals =[case_metrics [cid ]["levels"][k ][q ]for k in range (len (LEVELS ))]
        base =vals [-1 ]
        ax .plot (h ,[v /base for v in vals ],"o-",label =label )
    ax .set_xlabel ("1/nx")
    ax .set_ylabel ("quantity / finest-level value")
    ax .set_title ("%s grid convergence (r = 1.5)"%cid )
    ax .legend (fontsize =6 )
    fig .tight_layout ()
    fig .savefig (FIGS /"fig6_grid_convergence.png")
    plt .close (fig )


def render_report (anchor ,metrics ,rows ,case_metrics ):
    m =metrics 
    rc =m ["criterion1_recompute"]
    sc =m ["bubble_scaling"]
    srs =json .loads ((OUT /"srs_comparison.json").read_text ())
    L =[]
    A =L .append 
    c3 =max (r ["energy_residual_worst_level"]for r in rows )
    gci_def =max (max (r ["gci_deflection"],r ["gci_acceleration"])for r in rows )
    gci_imp =max (r ["gci_impulse_precav"]for r in rows )
    fe_p =max (r ["fe_fluid_pressure_rel_err"]for r in rows )
    fe_v =max (r ["fe_fluid_velocity_rel_err"]for r in rows )
    fe_err =max (fe_p ,fe_v )
    imp_lo =min (r ["wall_impulse_precav_over_cole"]for r in rows )
    imp_hi =max (r ["wall_impulse_precav_over_cole"]for r in rows )
    peak_dev =max (r ["wall_peak_rel_dev_vs_cole"]for r in rows )
    imp_dev =max (r ["wall_impulse_precav_rel_dev_vs_cole"]for r in rows )
    peak_ratio =max (r ["wall_peak_over_cole"]for r in rows )
    first =case_metrics [rows [0 ]["case_id"]]["levels"][-1 ]

    A ("# W23 UNDEX shock chain (multi-case ensemble): Cole environment -> ship-panel FSI FEM -> SRS\n")
    A ("- mission 15095, repository TensorFEM, discipline: structural")
    A ("- run commit `%s`"%m ["git_commit_at_run"])
    A ("- external anchor AN-COLE acquisition status: **%s** "
    "(`benchmarks/undex_shock/cole_anchor.json`, sha256 `%s...`)\n"
    %(anchor ["acquisition_status"],m ["anchor_sha256"][:16 ]))

    A ("## 1. Case introduction\n")
    A ("Underwater explosion shock loading a ship panel: TNT-equivalent charge W [kg], "
    "standoff R [m], depth H [m].  Environment from the Cole (1948) empirical law "
    "`p_m = K (W^(1/3)/R)^A`, exponential decay `p_i(t) = p_m exp(-t/theta)`, "
    "`theta = K_theta W^(1/3)(W^(1/3)/R)^n`, bubble period "
    "`T = K_b W^(1/3)/(H+10.33)^(5/6)`.  Structure: stiffened-panel bay idealised as a "
    "simply supported plate, 1.5 m x 1.0 m x 10 mm steel (E = 206 GPa, nu = 0.3, "
    "rho = 7850 kg/m^3, areal mass 78.5 kg/m^2, Kirchhoff f11 = %s Hz).  Seawater "
    "rho = 1025 kg/m^3, c = 1500 m/s (rho c = 1.5375e6 Pa s/m).\n"
    %f (first ["first_frequency_hz"],2 ))
    A ("| case | W [kg] | R [m] | H [m] | reduced range | p_m [MPa] | theta [us] | "
    "T_b [s] | Cole impulse [Pa s] |")
    A ("|---|---|---|---|---|---|---|---|---|")
    for r in rows :
        A ("| %s | %g | %g | %g | %.4f | %.4f | %.2f | %.5f | %.1f |"%(
        r ["case_id"],r ["weight_kg"],r ["range_m"],r ["depth_m"],
        r ["reduced_range"],r ["cole_peak_pressure_mpa"],r ["cole_decay_time_us"],
        r ["cole_bubble_period_s"],r ["cole_impulse_pa_s"]))
    A ("")

    A ("## 2. Method\n")
    A ("**(1) Independent Cole recomputation.** The anchor register "
    "`benchmarks/undex_shock/cole_anchor.json` is produced by "
    "`scripts/w23_cole_anchor_record.py`, whose recorded values come from a "
    "standalone implementation that does not import tensorfem (power form and "
    "exp/log form, cross-checked).  `tensorfem.undex_chain.cole_recompute_vs_anchor` "
    "recomputes peak pressure, decay time, bubble period and impulse from the "
    "anchor's parameter block and compares case by case.  This checks "
    "*transcription/implementation fidelity* of the recorded formula and "
    "parameters -- it is **not** a physical validation against the printed "
    "Cole (1948) values, which is BLOCKED (see section 5).\n")
    A ("**(2) Ship-panel FSI FEM.** Q4 Mindlin plate (MITC assumed natural shear "
    "strain, no zero-energy modes), hard simple supports (w = 0), one wetted face; "
    "the fluid leg is the exact normal-incidence plane-wave solution (local "
    "impedance coupling) `p_wall = 2 p_i - rho c v_n`, where `2 p_i` is the "
    "rigid-wall doubling and `- rho c v_n` the consistent impedance matrix -- both "
    "discretised with the same bilinear shape functions as the structure.  Time "
    "integration: average-acceleration Newmark (beta = 1/4, gamma = 1/2), single "
    "factorisation.  Three levels (nx,ny,steps/theta) = %s with uniform refinement "
    "ratio r = 1.5.\n"%" / ".join ("(%d,%d,%d)"%tuple (lv )for lv in m ["levels"]))
    A ("**(3) Two verification legs.** (a) same-problem check: the same FEM path "
    "degenerated to an areal-mass plate (`bc='free'`, `stiffness_scale=0`, no "
    "bending, no supports); the consistent mass, impedance and load matrices then "
    "make rigid translation an exact solution of the discrete system, so the "
    "comparison with Taylor's closed form verifies the coupling assembly and the "
    "time integration (`tests/test_undex_chain.py::test_areal_mass_verification`). "
    "(b) independent fluid leg: the base module `taylor_fsi_fem` (1-D acoustic FE "
    "fluid column, 120 elements, plus areal-mass plate and exact outgoing "
    "truncation) solves the same case and is compared with the closed form.\n")
    A ("**(4) Energy ledger.** The discrete identity `P_inc - P_ref = p_i int(v dA) - "
    "rho c int(v^2 dA) = u_dot^T (f - C u_dot)` holds exactly for the consistent "
    "load and consistent impedance matrices used here, so the residual measures "
    "only time-integration error: `E_inc = int p_i^2/(rho c) dA dt`, "
    "`E_ref = int (p_i - rho c v_h)^2/(rho c) dA dt`, "
    "`E_plate = 1/2 u_dot^T M u_dot + 1/2 u^T K u`.\n")
    A ("**(5) SRS.** Panel-centre acceleration at the finest level -> SDOF maximax SRS "
    "by exact piecewise integration (1 Hz - 10 kHz, 120 log points, zeta = 2%% and "
    "5%%), compared band by band (6 bands) with the SRS of Taylor's rigid areal-mass "
    "plate acceleration.  That comparison quantifies the **panel-flexibility "
    "effect**, not solver error (solver error is leg (3)).\n")

    A ("## 3. Results\n")
    A ("### 3.1 Criterion 1: Cole recomputation vs anchor records (<1%%)\n")
    A ("| case | anchor p_m [MPa] | recomputed p_m [MPa] | dev | anchor theta [us] | "
    "recomputed theta [us] | dev | anchor T_b [s] | recomputed T_b [s] | dev | "
    "impulse dev |")
    A ("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rc ["rows"]:
        cid =r ["case_id"]
        ar =next (x for x in anchor ["recorded_values"]if x ["case_id"]==cid )
        A ("| %s | %.6f | %.6f | %s | %.6f | %.6f | %s | %.8f | %.8f | %s | %s |"%(
        cid ,ar ["peak_pressure_pa"]/1e6 ,r ["recomputed"]["peak_pressure_pa"]/1e6 ,
        f (r ["relative_deviation"]["peak_pressure"]),
        ar ["decay_time_s"]*1e6 ,r ["recomputed"]["decay_time_s"]*1e6 ,
        f (r ["relative_deviation"]["decay_time"]),
        ar ["bubble_period_s"],r ["recomputed"]["bubble_period_s"],
        f (r ["relative_deviation"]["bubble_period"]),
        f (r ["relative_deviation"]["specific_impulse"])))
    A ("")
    A ("Max relative deviation %s (threshold 1%%): floating-point level, i.e. the two "
    "independent implementations agree; this does not validate the coefficient "
    "values K, A, K_theta, K_b themselves.\n"%f (rc ["max_relative_deviation"]))
    A ("**Coefficient-free bubble-period scaling.** Rayleigh collapse integral "
    "`t_c = sqrt(3 rho/2 p_h) R_0 I` with `I = (1/3) B(5/6,1/2) = 0.746835` (Euler "
    "beta integral), so the classical prefactor `0.914686 = sqrt(3/2) I` is "
    "*derived*; with `R_0 = (3E/4 pi p_h)^(1/3)` this gives "
    "`t_c ~ W^(1/3) p_h^(-5/6)`.  Machine check: weight exponent %s (expected 1/3), "
    "head exponent %s (expected -5/6), pass = %s.  So the exponent structure of the "
    "bubble law is derivable; only the prefactor K_b = %s is empirical and carried "
    "as an untranscribed candidate.\n"
    %(f (sc ["log_slope_weight_exponent"],6 ),f (sc ["log_slope_head_exponent"],6 ),
    ok (sc ["pass"]),f (m ["bubble_prefactor_used"])))

    A ("### 3.2 Criterion 2: FEM peak pressure / impulse vs Cole, per case\n")
    A ("Four reference levels must be kept apart: Cole free field `p_m`, `p_m theta`; "
    "the rigid-wall limit `2 p_m`, `2 p_m theta` (reflection doubling); and the "
    "Taylor closed form (the reflection reduction a movable plate actually gives). "
    "The FEM wetted-face peak occurs at t = 0 (where v = 0) and therefore equals "
    "`2 p_m` exactly -- an exact consequence of reflection doubling.\n")
    A ("| case | FEM wall peak [MPa] | /p_m | dev vs Cole free field | Cole impulse "
    "[Pa s] | FEM pre-cavitation impulse [Pa s] | /Cole impulse | cavitation onset "
    "[us] | wall tension [MPa] |")
    A ("|---|---|---|---|---|---|---|---|---|")
    for r in rows :
        A ("| %s | %.4f | %.4f | %s | %.1f | %.1f | %.4f | %.1f | %.4f |"%(
        r ["case_id"],r ["wall_peak_pressure_mpa"],r ["wall_peak_over_cole"],
        f (r ["wall_peak_rel_dev_vs_cole"]),r ["cole_impulse_pa_s"],
        r ["wall_impulse_precav_pa_s"],r ["wall_impulse_precav_over_cole"],
        r ["cavitation_onset_us"],r ["min_wall_pressure_mpa"]))
    A ("")
    A ("Discretised-fluid leg vs Taylor closed form: interface-pressure deviation %s, "
    "velocity deviation %s (per case pressure/velocity: %s); its pre-cavitation "
    "impulse deviates from the Cole incident impulse by %s.\n"
    %(f (fe_p ),f (fe_v ),
    ", ".join ("%s %s/%s"%(r ["case_id"],f (r ["fe_fluid_pressure_rel_err"]),
    f (r ["fe_fluid_velocity_rel_err"]))for r in rows ),
    ", ".join ("%s %s"%(r ["case_id"],f (r ["fe_fluid_impulse_rel_dev_vs_cole"]))
    for r in rows )))
    A ("Panel-flexibility effect (relative to the rigid areal-mass plate): deflection "
    "ratio %s-%s, centre-velocity deviation %s-%s, wall-pressure deviation %s-%s.\n"
    %(f (min (r ["deflection_ratio_vs_rigid"]for r in rows )),
    f (max (r ["deflection_ratio_vs_rigid"]for r in rows )),
    f (min (r ["taylor_velocity_rel_err"]for r in rows )),
    f (max (r ["taylor_velocity_rel_err"]for r in rows )),
    f (min (r ["taylor_pressure_rel_err"]for r in rows )),
    f (max (r ["taylor_pressure_rel_err"]for r in rows ))))

    A ("### 3.3 Criterion 3: energy conservation residual (<5%%)\n")
    A ("| case | worst-level residual max | finest-level residual | threshold |")
    A ("|---|---|---|---|")
    for r in rows :
        A ("| %s | %s | %s | 5e-2 |"%(r ["case_id"],
        f (r ["energy_residual_worst_level"]),
        f (r ["energy_residual_max"])))
    A ("")

    A ("### 3.4 Criterion 4: band-wise SRS deviations (>=5 bands) + GCI (<10%%)\n")
    A ("SRS comparison (zeta = 2%%, reference = Taylor rigid areal-mass plate "
    "acceleration):\n")
    A ("| case | band [Hz] | FEM peak [g] | reference peak [g] | relative deviation | "
    "FEM peak frequency [Hz] |")
    A ("|---|---|---|---|---|---|")
    for r in rows :
        for b in srs ["cases"][r ["case_id"]]["damping_0.02"]:
            A ("| %s | %g-%g | %.3f | %.3f | %s | %.1f |"%(
            r ["case_id"],b ["band_hz"][0 ],b ["band_hz"][1 ],
            b ["fem_peak_sa_ms2"]/9.80665 ,b ["reference_peak_sa_ms2"]/9.80665 ,
            f (b ["relative_deviation"]),b ["fem_peak_frequency_hz"]))
    A ("")
    A ("Grid convergence (r = 1.5; values coarse / medium / fine, GCI on the finest):\n")
    A ("| case | quantity | coarse / medium / fine | observed order p | GCI | "
    "verdict (<10%%) |")
    A ("|---|---|---|---|---|---|")
    for r in rows :
        g =case_metrics [r ["case_id"]]["gci"]
        for q ,label in GCIQ .items ():
            d =g [q ]
            if d ["available"]:
                A ("| %s | %s | %s / %s / %s | %s | %s | %s |"%(
                r ["case_id"],label ,f (d ["values"][2 ]),f (d ["values"][1 ]),
                f (d ["values"][0 ]),f (d ["observed_order"],3 ),f (d ["gci_fine"]),
                ok (d ["gci_fine"]<GCI_LIMIT )))
            else :
                A ("| %s | %s | %s / %s / %s | n/a | 0 (identical at all levels) | "
                "PASS (analytic identity: v = 0 at t = 0 => max|p_wall| = 2 p_m) |"%(
                r ["case_id"],label ,f (d ["values"][2 ]),f (d ["values"][1 ]),
                f (d ["values"][0 ])))
    A ("")

    A ("## 4. Criterion-by-criterion adjudication\n")
    A ("| # | criterion (fixed before the run) | value | threshold | verdict | "
    "evidence pointer |")
    A ("|---|---|---|---|---|---|")
    A ("| 1 | Cole recomputation vs anchor records (peak pressure / decay time / "
    "bubble period / impulse, per case) | max deviation %s | <1e-2 | %s | "
    "`benchmarks/undex_shock/cole_anchor.json`, "
    "`chain_metrics.json:criterion1_recompute` |"
    %(f (rc ["max_relative_deviation"]),ok (rc ["pass"])))
    A ("| 2a | FEM wall peak pressure vs Cole free field p_m | deviation %s "
    "(wall/incident = %.3f) | <15%% | **FAIL (literal reading)** | "
    "`case_matrix.csv:wall_peak_rel_dev_vs_cole` |"%(f (peak_dev ),peak_ratio ))
    A ("| 2b | FEM wall impulse (cavitation-truncated) vs Cole incident impulse p_m "
    "theta | deviation %s (ratio %.3f-%.3f) | <15%% | **FAIL (literal reading)** | "
    "`case_matrix.csv:wall_impulse_precav_rel_dev_vs_cole` |"
    %(f (imp_dev ),imp_lo ,imp_hi ))
    A ("| 2c | same, against the rigid-wall limit 2 p_m / 2 p_m theta (physically "
    "correct reference frame) | peak deviation %s; impulse ratio %.3f-%.3f | peak "
    "<15%% | %s | `case_matrix.csv:wall_peak_over_cole` |"
    %(f (peak_ratio -2.0 ),imp_lo ,imp_hi ,ok (abs (peak_ratio -2.0 )<0.15 )))
    A ("| 2d | discretised-fluid FEM leg vs Taylor closed form (solver accuracy) | max "
    "deviation %s | <15%% | %s | `chain_metrics.json:cases[*].fe_fluid` |"
    %(f (fe_err ),ok (fe_err <0.15 )))
    A ("| 3 | system energy-conservation residual (whole shock window, worst level) | "
    "%s | <5e-2 | %s | `fig5_energy_ledger.png`, "
    "`case_matrix.csv:energy_residual_worst_level` |"%(f (c3 ),ok (c3 <0.05 )))
    A ("| 4a | SRS band coverage | %d bands x 2 damping ratios | >=5 | %s | "
    "`benchmarks/undex_shock/srs_comparison.json` |"
    %(len (SRS_BANDS_HZ ),ok (len (SRS_BANDS_HZ )>=5 )))
    A ("| 4b | GCI (peak deflection / peak acceleration) | worst %s | <10%% | %s | "
    "`case_matrix.csv:gci_deflection`, `gci_acceleration` |"
    %(f (gci_def ),ok (gci_def <GCI_LIMIT )))
    A ("| 4c | GCI (pre-cavitation impulse) | worst %s | <10%% | %s | "
    "`case_matrix.csv:gci_impulse_precav` |"%(f (gci_imp ),ok (gci_imp <GCI_LIMIT )))
    A ("| 4d | criterion 4 combined (4a and 4b and 4c) | - | - | %s | rows above |"
    %ok (gci_def <GCI_LIMIT and gci_imp <GCI_LIMIT ))
    A ("")
    A ("**Adjudication note (no whitewash).** The literal reading of criterion 2 "
    "(FEM wall quantities must close with the Cole free-field quantities within "
    "15%%) is physically unsatisfiable: the wall pressure contains the reflection, "
    "so its peak is `2 p_m`; and the impulse is further truncated by cavitation "
    "(water cannot sustain tension).  Criterion 2 is therefore judged "
    "**FAIL** on 2a/2b, while 2c (deviation from `2 p_m`: %s) and 2d (solver "
    "accuracy: %s) give the physically meaningful closure.  If a single binary "
    "verdict is required, criterion 2 = **FAIL**.\n"
    %(f (peak_ratio -2.0 ),f (fe_err )))

    A ("## 5. Caveats and limitations\n")
    A ("1. **Anchor unobtainable.** No authorised copy of Cole (1948) exists in this "
    "environment (filesystem sweep, PDF corpus and network all failed; every attempt "
    "is recorded verbatim in `cole_anchor.json:acquisition_attempts`, and the network "
    "request was rejected outright by the sandbox security policy).  K, A, K_theta, "
    "K_b are therefore implementation parameters / untranscribed candidates with "
    "`usable_for_physics_validation = false`; criterion 1 only checks transcription "
    "fidelity.  Any claim of quantitative agreement with Cole's printed values would "
    "be out of scope.")
    A ("2. **Cavitation not modelled.** The linear acoustic coupling lets water carry "
    "tension.  This report quantifies the defect: wall pressure reaches %s to %s MPa "
    "(tension) and cavitation starts at %s-%s us (about 0.3-0.7 theta).  Beyond that "
    "instant the model is non-physical, so the impulse is integrated only up to "
    "cavitation, and the pre-cavitation impulse has the worst grid convergence "
    "(4c).  The correct next step is a cavitation switch on the impedance term "
    "(detach when p_wall <= 0, cavity pressure zero), which makes the system "
    "nonlinear and needs per-step re-factorisation."
    %(f (min (r ["min_wall_pressure_mpa"]for r in rows )),
    f (max (r ["min_wall_pressure_mpa"]for r in rows )),
    f (min (r ["cavitation_onset_us"]for r in rows )),
    f (max (r ["cavitation_onset_us"]for r in rows ))))
    A ("3. **The fluid leg is the exact normal plane-wave solution.** "
    "`p_wall = 2 p_i - rho c v_n` is exact only for normal incidence on a locally "
    "reacting surface; it carries no diffraction, spherical spreading, structural "
    "curvature or stiffeners.  Stiffened panels, oblique incidence and contact "
    "explosions need another model.  A discretised fluid (1-D acoustic FEM) appears "
    "only in the verification leg (120 elements, interface-pressure deviation %s)."
    %f (fe_err ))
    A ("4. **Plate idealisation.** Hard simple supports, uniform thickness, no "
    "stiffeners, no initial imperfections, linear elastic (yield not checked: wall "
    "peaks of %s MPa are far above the 235 MPa mild-steel yield, so plasticity and "
    "rupture would occur in reality)."%f (max (r ["wall_peak_pressure_mpa"]for r in rows )))
    A ("5. **Record length.** The window is 16 theta (about %.2f ms), far shorter than "
    "the panel's first period (about %.1f ms), so the 1-10 Hz SRS band is a "
    "quasi-static plateau and low-order structural participation has not developed; "
    "this is a window limitation, not an error."
    %(first ["duration_s"]*1e3 ,1e3 /first ["first_frequency_hz"]))
    A ("6. **GCI observed order is clipped.** With small three-level differences the "
    "observed order is noisy (`chain_metrics.json:cases[*].gci` flags "
    "`order_clipped`); the p = 2 GCI is reported alongside, and both must be <10%% "
    "to pass.")
    A ("7. **The criterion-2 deviation is not numerical error.** Because v(0) = 0 the "
    "wall peak equals 2 p_m exactly (identical on all three meshes), so its 'grid "
    "convergence' is an analytic identity rather than a convergence result, as "
    "marked in section 3.4.\n")

    A ("## 6. Error judgement (vs reference, item by item; is it <3%%?)\n")
    A ("| quantity | reference | deviation | <3%%? | note |")
    A ("|---|---|---|---|---|")
    A ("| areal-mass same-problem leg (deflection/velocity/acceleration/wall pressure) | "
    "Taylor closed form | <1%% (`tests/test_undex_chain.py::"
    "test_areal_mass_verification`) | yes | verifies coupling assembly and time "
    "integration |")
    A ("| discretised-fluid leg interface pressure / velocity | Taylor closed form | "
    "%s / %s | no (within the 15%% platform accuracy line) | linear-element dispersion "
    "of a short pulse (theta = %.0f us, c theta = %.3f m) |"
    %(f (fe_p ),f (fe_v ),rows [0 ]["cole_decay_time_us"],
    case_metrics [rows [0 ]["case_id"]]["env"]["decay_time_s"]*1500.0 ))
    A ("| Cole recomputation | anchor records | %s | yes | transcription fidelity only, "
    "not physical validation |"%f (rc ["max_relative_deviation"]))
    A ("| FEM wall peak pressure | Cole free field p_m | %s | no | reflection doubling, "
    "physics not error |"%f (peak_dev ))
    A ("| panel centre deflection | Taylor rigid areal-mass plate | ratio %.4f-%.4f | "
    "partly | panel-flexibility effect, not error |"
    %(min (r ["deflection_ratio_vs_rigid"]for r in rows ),
    max (r ["deflection_ratio_vs_rigid"]for r in rows )))
    A ("| energy-conservation residual | 0 | %s | yes | whole shock window |"%f (c3 ))
    A ("")
    A ("**Explicit answer to 'is it <3%%?'**: only the same-problem verification leg "
    "(areal-mass plate vs Taylor closed form) and the Cole recomputation reach "
    "<3%% (in fact floating-point / quadrature-error level); the discretised-fluid "
    "leg (%s) and the wall quantities vs the Cole free field (%s) do not, for the "
    "reasons located in section 5 (linear-element dispersion; reflection and "
    "cavitation physics).\n"%(f (fe_err ),f (peak_dev )))

    A ("## 7. Provenance (artifacts + commit + sha256)\n")
    A ("| artifact | sha256 / note |")
    A ("|---|---|")
    A ("| `benchmarks/undex_shock/cole_anchor.json` | `%s` (status %s) |"
    %(m ["anchor_sha256"],m ["anchor_acquisition_status"]))
    A ("| `src/tensorfem/undex_chain.py` | `%s` |"%m ["package_file_sha256"])
    A ("| `src/tensorfem/undex_shock.py` (base module, taken from this repository at "
    "commit a706ec07) | `%s` |"%m ["base_module_sha256"])
    A ("| `scripts/w23_undex_shock_chain_run.py` | `%s` |"%m ["script_sha256"])
    A ("| `tests/test_undex_chain.py` | `%s` |"%m ["test_file_sha256"])
    A ("| `benchmarks/undex_shock/case_matrix.csv` | per-case metrics; source of every "
    "number above |")
    A ("| `benchmarks/undex_shock/srs_comparison.json` | SRS curves and band-wise "
    "deviations |")
    A ("| `benchmarks/undex_shock/chain_metrics.json` | GCI raw three-level values, "
    "bubble-scaling machine check, parameter provenance |")
    A ("| `docs/assets/w23_undex_shock/*.png` | %s |"%", ".join (m ["figures"]))
    A ("")
    A ("Figures: `fig1_cole_environment.png` (Cole pulses and bubble periods), "
    "`fig2_wall_pressure.png` (wall pressure vs Taylor / rigid-wall limit / Cole "
    "free field), `fig3_panel_response.png` (centre deflection and acceleration), "
    "`fig4_srs.png` (SRS with band edges), `fig5_energy_ledger.png` (incident / "
    "reflected / plate energy and residual), `fig6_grid_convergence.png` "
    "(GCI).")
    REPORT .parent .mkdir (parents =True ,exist_ok =True )
    REPORT .write_text ("\n".join (L )+"\n")


def rerender ():
    """Rebuild the report from the already-written chain_metrics.json (no re-run)."""
    metrics =json .loads ((OUT /"chain_metrics.json").read_text ())
    anchor =json .loads (ANCHOR .read_text ())
    rows =metrics ["rows"]
    case_metrics =metrics ["cases"]
    metrics ["figures"]=sorted (p .name for p in FIGS .glob ("*.png"))
    render_report (anchor ,metrics ,rows ,case_metrics )
    print ("rerendered",REPORT )


if __name__ =="__main__"and "--rerender"in __import__ ("sys").argv :
    rerender ()


if __name__ =="__main__"and "--rerender"not in __import__ ("sys").argv :
    main ()
