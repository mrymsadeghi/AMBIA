from atlas_codes.regions_per_sections_adult import regs_per_section, Regions_n_colors_List

 

def create_regs_n_colors_per_sec_list(atlasnum):
    regs_per_sec_List = regs_per_section[int(atlasnum)] #list
    regs_n_colors_per_sec_List = [('root', 'not detected', '000000', (0, 0, 0))]
    for elem in Regions_n_colors_List:
        if elem[-3] in regs_per_sec_List:
            regs_n_colors_per_sec_List.append(elem)
   
    Bgr_Color_List = []
    Rgb_Color_List = []
    for elem in regs_n_colors_per_sec_List:
        Bgr_Color_List.append(elem[-1])
        Rgb_Color_List.append((elem[-1][2],elem[-1][1],elem[-1][0]))

    return regs_n_colors_per_sec_List, Rgb_Color_List, Bgr_Color_List


Region_names = ['CTX_bg _L', 'CTX_bg _R', 'FRP1 _L', 'FRP1 _R', 'FRP2/3 _L', 'FRP2/3 _R', 'FRP5 _L', 'FRP5 _R', 'FRP6a _L', 'FRP6a _R', 'FRP6b _L', 'FRP6b _R', 'MOp1 _L', 'MOp1 _R', 'MOp2/3 _L', 'MOp2/3 _R', 'MOp5 _L', 'MOp5 _R', 'MOp6a _L', 'MOp6a _R', 'MOp6b _L', 'MOp6b _R', 'MOs1 _L', 'MOs1 _R', 'MOs2/3 _L', 'MOs2/3 _R', 'MOs5 _L', 'MOs5 _R', 'MOs6a _L', 'MOs6a _R', 'MOs6b _L', 'MOs6b _R', 'SSp-n1 _L', 'SSp-n1 _R', 'SSp-n2/3 _L', 'SSp-n2/3 _R', 'SSp-n4 _L', 'SSp-n4 _R', 'SSp-n5 _L', 'SSp-n5 _R', 'SSp-n6a _L', 'SSp-n6a _R', 'SSp-n6b _L', 'SSp-n6b _R', 'SSp-bfd1 _L', 'SSp-bfd1 _R', 'SSp-bfd2/3 _L', 'SSp-bfd2/3 _R', 'SSp-bfd4 _L', 'SSp-bfd4 _R', 'SSp-bfd5 _L', 'SSp-bfd5 _R', 'SSp-bfd6a _L', 'SSp-bfd6a _R', 'SSp-bfd6b _L', 'SSp-bfd6b _R', 'SSp-ll1 _L', 'SSp-ll1 _R', 'SSp-ll2/3 _L', 'SSp-ll2/3 _R', 'SSp-ll4 _L', 'SSp-ll4 _R', 'SSp-ll5 _L', 'SSp-ll5 _R', 'SSp-ll6a _L', 'SSp-ll6a _R', 'SSp-ll6b _L', 'SSp-ll6b _R', 'SSp-m1 _L', 'SSp-m1 _R', 'SSp-m2/3 _L', 'SSp-m2/3 _R', 'SSp-m4 _L', 'SSp-m4 _R', 'SSp-m5 _L', 'SSp-m5 _R', 'SSp-m6a _L', 'SSp-m6a _R', 'SSp-m6b _L', 'SSp-m6b _R', 'SSp-ul1 _L', 'SSp-ul1 _R', 'SSp-ul2/3 _L', 'SSp-ul2/3 _R', 'SSp-ul4 _L', 'SSp-ul4 _R', 'SSp-ul5 _L', 'SSp-ul5 _R', 'SSp-ul6a _L', 'SSp-ul6a _R', 'SSp-ul6b _L', 'SSp-ul6b _R', 'SSp-tr1 _L', 'SSp-tr1 _R', 'SSp-tr2/3 _L', 'SSp-tr2/3 _R', 'SSp-tr4 _L', 'SSp-tr4 _R', 'SSp-tr5 _L', 'SSp-tr5 _R', 'SSp-tr6a _L', 'SSp-tr6a _R', 'SSp-tr6b _L', 'SSp-tr6b _R', 'SSp-un1 _L', 'SSp-un1 _R', 'SSp-un2/3 _L', 'SSp-un2/3 _R', 'SSp-un4 _L', 'SSp-un4 _R', 'SSp-un5 _L', 'SSp-un5 _R', 'SSp-un6a _L', 'SSp-un6a _R', 'SSp-un6b _L', 'SSp-un6b _R', 'SSs1 _L', 'SSs1 _R', 'SSs2/3 _L', 'SSs2/3 _R', 'SSs4 _L', 'SSs4 _R', 'SSs5 _L', 'SSs5 _R', 'SSs6a _L', 'SSs6a _R', 'SSs6b _L', 'SSs6b _R', 'GU1 _L', 'GU1 _R', 'GU2/3 _L', 'GU2/3 _R', 'GU4 _L', 'GU4 _R', 'GU5 _L', 'GU5 _R', 'GU6a _L', 'GU6a _R', 'GU6b _L', 'GU6b _R', 'VISC1 _L', 'VISC1 _R', 'VISC2/3 _L', 'VISC2/3 _R', 'VISC4 _L', 'VISC4 _R', 'VISC5 _L', 'VISC5 _R', 'VISC6a _L', 'VISC6a _R', 'VISC6b _L', 'VISC6b _R', 'AUDd1 _L', 'AUDd1 _R', 'AUDd2/3 _L', 'AUDd2/3 _R', 'AUDd4 _L', 'AUDd4 _R', 'AUDd5 _L', 'AUDd5 _R', 'AUDd6a _L', 'AUDd6a _R', 'AUDd6b _L', 'AUDd6b _R', 'AUDp1 _L', 'AUDp1 _R', 'AUDp2/3 _L', 'AUDp2/3 _R', 'AUDp4 _L', 'AUDp4 _R', 'AUDp5 _L', 'AUDp5 _R', 'AUDp6a _L', 'AUDp6a _R', 'AUDp6b _L', 'AUDp6b _R', 'AUDpo1 _L', 'AUDpo1 _R', 'AUDpo2/3 _L', 'AUDpo2/3 _R', 'AUDpo4 _L', 'AUDpo4 _R', 'AUDpo5 _L', 'AUDpo5 _R', 'AUDpo6a _L', 'AUDpo6a _R', 'AUDpo6b _L', 'AUDpo6b _R', 'AUDv1 _L', 'AUDv1 _R', 'AUDv2/3 _L', 'AUDv2/3 _R', 'AUDv4 _L', 'AUDv4 _R', 'AUDv5 _L', 'AUDv5 _R', 'AUDv6a _L', 'AUDv6a _R', 'AUDv6b _L', 'AUDv6b _R', 'VISal1 _L', 'VISal1 _R', 'VISal2/3 _L', 'VISal2/3 _R', 'VISal4 _L', 'VISal4 _R', 'VISal5 _L', 'VISal5 _R', 'VISal6a _L', 'VISal6a _R', 'VISal6b _L', 'VISal6b _R', 'VISam1 _L', 'VISam1 _R', 'VISam2/3 _L', 'VISam2/3 _R', 'VISam4 _L', 'VISam4 _R', 'VISam5 _L', 'VISam5 _R', 'VISam6a _L', 'VISam6a _R', 'VISam6b _L', 'VISam6b _R', 'VISl1 _L', 'VISl1 _R', 'VISl2/3 _L', 'VISl2/3 _R', 'VISl4 _L', 'VISl4 _R', 'VISl5 _L', 'VISl5 _R', 'VISl6a _L', 'VISl6a _R', 'VISl6b _L', 'VISl6b _R', 'VISp1 _L', 'VISp1 _R', 'VISp2/3 _L', 'VISp2/3 _R', 'VISp4 _L', 'VISp4 _R', 'VISp5 _L', 'VISp5 _R', 'VISp6a _L', 'VISp6a _R', 'VISp6b _L', 'VISp6b _R', 'VISpl1 _L', 'VISpl1 _R', 'VISpl2/3 _L', 'VISpl2/3 _R',
'VISpl4 _L', 'VISpl4 _R', 'VISpl5 _L', 'VISpl5 _R', 'VISpl6a _L', 'VISpl6a _R', 'VISpl6b _L', 'VISpl6b _R', 'VISpm1 _L', 'VISpm1 _R', 'VISpm2/3 _L', 'VISpm2/3 _R', 'VISpm4 _L', 'VISpm4 _R', 'VISpm5 _L', 'VISpm5 _R', 'VISpm6a _L', 'VISpm6a _R', 'VISpm6b _L', 'VISpm6b _R', 'VISli1 _L', 'VISli1 _R', 'VISli2/3 _L', 'VISli2/3 _R', 'VISli4 _L', 'VISli4 _R', 'VISli5 _L', 'VISli5 _R', 'VISli6a _L', 'VISli6a _R', 'VISli6b _L', 'VISli6b _R', 'VISpor1 _L', 'VISpor1 _R', 'VISpor2/3 _L', 'VISpor2/3 _R', 'VISpor4 _L', 'VISpor4 _R', 'VISpor5 _L', 'VISpor5 _R', 'VISpor6a _L', 'VISpor6a _R', 'VISpor6b _L', 'VISpor6b _R', 'ACAd1 _L', 'ACAd1 _R', 'ACAd2/3 _L', 'ACAd2/3 _R', 'ACAd5 _L', 'ACAd5 _R', 'ACAd6a _L', 'ACAd6a _R', 'ACAd6b _L', 'ACAd6b _R', 'ACAv1 _L', 'ACAv1 _R', 'ACAv2/3 _L', 'ACAv2/3 _R', 'ACAv5 _L', 'ACAv5 _R', 'ACAv6a _L', 'ACAv6a _R', 'ACAv6b _L', 'ACAv6b _R', 'PL1 _L', 'PL1 _R', 'PL2 _L', 'PL2 _R', 'PL2/3 _L', 'PL2/3 _R', 'PL5 _L', 'PL5 _R', 'PL6a _L', 'PL6a _R', 'PL6b _L', 'PL6b _R', 'ILA1 _L', 'ILA1 _R', 'ILA2 _L', 'ILA2 _R', 'ILA2/3 _L', 'ILA2/3 _R', 'ILA5 _L', 'ILA5 _R', 'ILA6a _L', 'ILA6a _R', 'ILA6b _L', 'ILA6b _R', 'ORBl1 _L', 'ORBl1 _R', 'ORBl2/3 _L', 'ORBl2/3 _R', 'ORBl5 _L', 'ORBl5 _R', 'ORBl6a _L', 'ORBl6a _R', 'ORBl6b _L', 'ORBl6b _R', 'ORBm1 _L', 'ORBm1 _R', 'ORBm2 _L', 'ORBm2 _R', 'ORBm2/3 _L', 'ORBm2/3 _R', 'ORBm5 _L', 'ORBm5 _R', 'ORBm6a _L', 'ORBm6a _R', 'ORBm6b _L', 'ORBm6b _R', 'ORBvl1 _L', 'ORBvl1 _R', 'ORBvl2/3 _L', 'ORBvl2/3 _R', 'ORBvl5 _L', 'ORBvl5 _R', 'ORBvl6a _L', 'ORBvl6a _R', 'ORBvl6b _L', 'ORBvl6b _R', 'AId1 _L', 'AId1 _R', 'AId2/3 _L', 'AId2/3 _R', 'AId5 _L', 'AId5 _R', 'AId6a _L', 'AId6a _R', 'AId6b _L', 'AId6b _R', 'AIp1 _L', 'AIp1 _R', 'AIp2/3 _L', 'AIp2/3 _R', 'AIp5 _L', 'AIp5 _R', 'AIp6a _L', 'AIp6a _R', 'AIp6b _L', 'AIp6b _R', 'AIv1 _L', 'AIv1 _R', 'AIv2/3 _L', 'AIv2/3 _R', 'AIv5 _L', 'AIv5 _R', 'AIv6a _L', 'AIv6a _R', 'AIv6b _L', 'AIv6b _R', 'RSPagl1 _L', 'RSPagl1 _R', 'RSPagl2/3 _L', 'RSPagl2/3 _R', 'RSPagl5 _L', 'RSPagl5 _R', 'RSPagl6a _L', 'RSPagl6a _R', 'RSPagl6b _L', 'RSPagl6b _R', 'RSPd1 _L', 'RSPd1 _R', 'RSPd2/3 _L', 'RSPd2/3 _R', 'RSPd5 _L', 'RSPd5 _R', 'RSPd6a _L', 'RSPd6a _R', 'RSPd6b _L', 'RSPd6b _R', 'RSPv1 _L', 'RSPv1 _R', 'RSPv2 _L', 'RSPv2 _R', 'RSPv2/3 _L', 'RSPv2/3 _R', 'RSPv5 _L', 'RSPv5 _R', 'RSPv6a _L', 'RSPv6a _R', 'RSPv6b _L', 'RSPv6b _R', 'PTLp1 _L', 'PTLp1 _R', 'PTLp2/3 _L', 'PTLp2/3 _R', 'PTLp4 _L', 'PTLp4 _R', 'PTLp5 _L', 'PTLp5 _R', 'PTLp6a _L', 'PTLp6a _R', 'PTLp6b _L', 'PTLp6b _R', 'VISa1 _L', 'VISa1 _R', 'VISa2/3 _L', 'VISa2/3 _R', 'VISa4 _L', 'VISa4 _R', 'VISa5 _L', 'VISa5 _R', 'VISa6a _L', 'VISa6a _R', 'VISa6b _L', 'VISa6b _R', 'VISrl1 _L', 'VISrl1 _R', 'VISrl2/3 _L', 'VISrl2/3 _R', 'VISrl4 _L', 'VISrl4 _R', 'VISrl5 _L', 'VISrl5 _R', 'VISrl6a _L', 'VISrl6a _R', 'VISrl6b _L', 'VISrl6b _R', 'TEa1 _L', 'TEa1 _R', 'TEa2/3 _L', 'TEa2/3 _R', 'TEa4 _L', 'TEa4 _R', 'TEa5 _L', 'TEa5 _R', 'TEa6a _L', 'TEa6a _R', 'TEa6b _L', 'TEa6b _R', 'PERI1 _L', 'PERI1 _R', 'PERI2/3 _L', 'PERI2/3 _R', 'PERI5 _L', 'PERI5 _R', 'PERI6a _L', 'PERI6a _R', 'PERI6b _L', 'PERI6b _R', 'ECT1 _L', 'ECT1 _R', 'ECT2/3 _L', 'ECT2/3 _R', 'ECT5 _L', 'ECT5 _R', 'ECT6a _L', 'ECT6a _R', 'ECT6b _L', 'ECT6b _R', 'OLF _L', 'OLF _R', 'MOB _L', 'MOB _R', 'MOBgl _L', 'MOBgl _R', 'MOBgr _L', 'MOBgr _R', 'MOBipl _L', 'MOBipl _R', 'MOBmi _L', 'MOBmi _R', 'MOBopl _L', 'MOBopl _R', 'AOB _L', 'AOB _R', 'AOBgl _L', 'AOBgl _R', 'AOBgr _L', 'AOBgr _R', 'AOBmi _L', 'AOBmi _R', 'AON _L', 'AON _R', 'AONd _L', 'AONd _R', 'AONe _L', 'AONe _R', 'AONl _L', 'AONl _R', 'AONm _L', 'AONm _R', 'AONpv _L', 'AONpv _R', 'AON1 _L', 'AON1 _R', 'TTd _L', 'TTd _R', 'TTd1 _L', 'TTd1 _R', 'TTd2 _L', 'TTd2 _R', 'TTd3 _L', 'TTd3 _R', 'TTd4 _L', 'TTd4 _R', 'TTv _L', 'TTv _R', 'TTv1 _L', 'TTv1 _R', 'TTv2 _L', 'TTv2 _R', 'TTv3 _L', 'TTv3 _R', 'DP _L', 'DP _R', 'DP1 _L', 'DP1 _R', 'DP2/3 _L', 'DP2/3 _R', 'DP5 _L', 'DP5 _R', 'DP6a _L', 'DP6a _R', 'PIR _L', 'PIR _R', 'PIR1 _L', 'PIR1 _R', 'PIR2 _L', 'PIR2 _R', 'PIR3 _L', 'PIR3 _R', 'NLOT1 _L', 'NLOT1 _R', 'NLOT2 _L', 'NLOT2 _R', 'NLOT3 _L', 'NLOT3 _R', 'COAa _L', 'COAa _R', 'COAa1 _L', 'COAa1 _R', 'COAa2 _L', 'COAa2 _R', 'COApl _L', 'COApl _R', 'COApl1 _L', 'COApl1 _R', 'COApl2 _L', 'COApl2 _R', 'COApl3 _L', 'COApl3 _R', 'COApm _L', 'COApm _R', 'COApm1 _L', 'COApm1 _R', 'COApm2 _L', 'COApm2 _R', 'COApm3 _L', 'COApm3 _R', 'PAA _L', 'PAA _R', 'PAA1 _L', 'PAA1 _R', 'PAA2 _L', 'PAA2 _R', 'PAA3 _L', 'PAA3 _R', 'TR _L', 'TR _R', 'TR1 _L', 'TR1 _R', 'TR2 _L',
'TR2 _R', 'TR3 _L', 'TR3 _R', 'HPF _L', 'HPF _R', 'CA1 _L', 'CA1 _R', 'CA1slm _L', 'CA1slm _R', 'CA1so _L', 'CA1so _R', 'CA1sp _L', 'CA1sp _R', 'CA1sr _L', 'CA1sr _R', 'CA2 _L', 'CA2 _R', 'CA2slm _L', 'CA2slm _R', 'CA2so _L', 'CA2so _R', 'CA2sp _L', 'CA2sp _R', 'CA2sr _L', 'CA2sr _R', 'CA3 _L', 'CA3 _R', 'CA3slm _L', 'CA3slm _R', 'CA3slu _L', 'CA3slu _R', 'CA3so _L', 'CA3so _R', 'CA3sp _L', 'CA3sp _R', 'CA3sr _L', 'CA3sr _R', 'DG-mo _L', 'DG-mo _R', 'DG-po _L', 'DG-po _R', 'DG-sg _L', 'DG-sg _R', 'FC _L', 'FC _R', 'IG _L', 'IG _R', 'ENTl1 _L', 'ENTl1 _R', 'ENTl2 _L', 'ENTl2 _R', 'ENTl2/3 _L', 'ENTl2/3 _R', 'ENTl2a _L', 'ENTl2a _R', 'ENTl2b _L', 'ENTl2b _R', 'ENTl3 _L', 'ENTl3 _R', 'ENTl4 _L', 'ENTl4 _R', 'ENTl4/5 _L', 'ENTl4/5 _R', 'ENTl5 _L', 'ENTl5 _R', 'ENTl6a _L', 'ENTl6a _R', 'ENTl6b _L', 'ENTl6b _R', 'ENTm1 _L', 'ENTm1 _R', 'ENTm2 _L', 'ENTm2 _R', 'ENTm2a _L', 'ENTm2a _R', 'ENTm2b _L', 'ENTm2b _R', 'ENTm3 _L', 'ENTm3 _R', 'ENTm4 _L', 'ENTm4 _R', 'ENTm5 _L', 'ENTm5 _R', 'ENTm6 _L', 'ENTm6 _R', 'ENTmv1 _L', 'ENTmv1 _R', 'ENTmv2 _L', 'ENTmv2 _R', 'ENTmv3 _L', 'ENTmv3 _R', 'ENTmv5/6 _L', 'ENTmv5/6 _R', 'PAR _L', 'PAR _R', 'PAR1 _L', 'PAR1 _R', 'PAR2 _L', 'PAR2 _R', 'PAR3 _L', 'PAR3 _R', 'POST _L', 'POST _R', 'POST1 _L', 'POST1 _R', 'POST2 _L', 'POST2 _R', 'POST3 _L', 'POST3 _R', 'PRE _L', 'PRE _R', 'PRE1 _L', 'PRE1 _R', 'PRE2 _L', 'PRE2 _R', 'PRE3 _L', 'PRE3 _R', 'SUB _L', 'SUB _R', 'SUBd-m _L', 'SUBd-m _R', 'SUBd-sp _L', 'SUBd-sp _R', 'SUBd-sr _L', 'SUBd-sr _R', 'SUBv-m _L', 'SUBv-m _R', 'SUBv-sp _L', 'SUBv-sp _R', 'SUBv-sr _L', 'SUBv-sr _R', 'ProS _L', 'ProS _R', 'HATA _L', 'HATA _R', 'APr _L', 'APr _R', 'CTXsp_bg _L', 'CTXsp_bg _R', 'CLA _L', 'CLA _R', 'EPd _L', 'EPd _R', 'EPv _L', 'EPv _R', 'LA _L', 'LA _R', 'BLAa _L', 'BLAa _R', 'BLAp _L', 'BLAp _R', 'BLAv _L', 'BLAv _R', 'BMAa _L', 'BMAa _R', 'BMAp _L', 'BMAp _R', 'PA _L', 'PA _R', 'STR_bg _L', 'STR_bg _R', 'CP _L', 'CP _R', 'ACB _L', 'ACB _R', 'FS _L', 'FS _R', 'OT _L', 'OT _R', 'isl _L', 'isl _R', 'islm _L', 'islm _R', 'OT1 _L', 'OT1 _R', 'OT2 _L', 'OT2 _R', 'OT3 _L', 'OT3 _R', 'LSc _L', 'LSc _R', 'LSr _L', 'LSr _R', 'LSv _L', 'LSv _R', 'SF _L', 'SF _R', 'SH _L', 'SH _R', 'sAMY _L', 'sAMY _R', 'AAA _L', 'AAA _R', 'BA _L', 'BA _R', 'CEAc _L', 'CEAc _R', 'CEAl _L', 'CEAl _R', 'CEAm _L', 'CEAm _R', 'IA _L', 'IA _R', 'MEA _L', 'MEA _R', 'MEAad _L', 'MEAad _R', 'MEAav _L', 'MEAav _R', 'MEApd-a _L', 'MEApd-a _R', 'MEApd-b _L', 'MEApd-b _R', 'MEApd-c _L', 'MEApd-c _R', 'MEApv _L', 'MEApv _R', 'PAL_bg _L', 'PAL_bg _R', 'GPe _L', 'GPe _R', 'GPi _L', 'GPi _R', 'SI _L', 'SI _R', 'MA _L', 'MA _R', 'MS _L', 'MS _R', 'NDB _L', 'NDB _R', 'TRS _L', 'TRS _R', 'BST _L', 'BST _R', 'BSTal _L', 'BSTal _R', 'BSTam _L', 'BSTam _R', 'BSTdm _L', 'BSTdm _R', 'BSTfu _L', 'BSTfu _R', 'BSTju _L', 'BSTju _R', 'BSTmg _L', 'BSTmg _R', 'BSTov _L', 'BSTov _R', 'BSTrh _L', 'BSTrh _R', 'BSTv _L', 'BSTv _R', 'BSTd _L', 'BSTd _R', 'BSTpr _L', 'BSTpr _R', 'BSTif _L', 'BSTif _R', 'BSTtr _L', 'BSTtr _R', 'BSTse _L', 'BSTse _R', 'BAC _L', 'BAC _R', 'TH_bg _L', 'TH_bg _R', 'VAL _L', 'VAL _R', 'VM _L', 'VM _R', 'VPL _L', 'VPL _R', 'VPLpc _L', 'VPLpc _R', 'VPM _L', 'VPM _R', 'VPMpc _L', 'VPMpc _R', 'PoT _L', 'PoT _R', 'SPFm _L', 'SPFm _R', 'SPFp _L', 'SPFp _R', 'SPA _L', 'SPA _R', 'PP _L', 'PP _R', 'MGd _L', 'MGd _R', 'MGv _L', 'MGv _R', 'MGm _L', 'MGm _R', 'LGd _L', 'LGd _R', 'LGd-sh _L', 'LGd-sh _R', 'LGd-co _L', 'LGd-co _R', 'LGd-ip _L', 'LGd-ip _R', 'LP _L', 'LP _R', 'PO _L', 'PO _R', 'POL _L', 'POL _R', 'SGN _L', 'SGN _R', 'Eth _L', 'Eth _R', 'AV _L', 'AV _R', 'AMd _L', 'AMd _R', 'AMv _L', 'AMv _R', 'AD _L', 'AD _R', 'IAM _L', 'IAM _R', 'IAD _L', 'IAD _R', 'LD _L', 'LD _R', 'IMD _L', 'IMD _R', 'MD _L', 'MD _R', 'MDc _L', 'MDc _R', 'MDl _L', 'MDl _R', 'MDm _L', 'MDm _R', 'SMT _L', 'SMT _R', 'PR _L', 'PR _R', 'PVT _L', 'PVT _R', 'PT _L', 'PT _R', 'RE _L', 'RE _R', 'Xi _L', 'Xi _R', 'RH _L', 'RH _R', 'CM _L', 'CM _R', 'PCN _L', 'PCN _R', 'CL _L', 'CL _R', 'PF _L', 'PF _R', 'PIL _L', 'PIL _R', 'RT _L', 'RT _R', 'IGL _L', 'IGL _R', 'IntG _L', 'IntG _R', 'LGv _L', 'LGv _R', 'SubG _L', 'SubG _R', 'MH _L', 'MH _R', 'LH _L', 'LH _R', 'HY_bg _L', 'HY_bg _R', 'SO _L', 'SO _R', 'ASO _L', 'ASO _R', 'NC _L', 'NC _R', 'PVH _L', 'PVH _R', 'PVHmm _L', 'PVHmm _R', 'PVHpml _L', 'PVHpml _R', 'PVHpmm _L', 'PVHpmm _R', 'PVHap _L', 'PVHap _R', 'PVHmpd _L', 'PVHmpd _R', 'PVHpv _L', 'PVHpv _R', 'PVa _L', 'PVa _R', 'PVi _L', 'PVi _R', 'ARH _L', 'ARH _R', 'ADP _L', 'ADP _R', 'AVP _L', 'AVP _R', 'AVPV _L', 'AVPV _R', 'DMH _L',
'DMH _R', 'DMHa _L', 'DMHa _R', 'DMHp _L', 'DMHp _R', 'DMHv _L', 'DMHv _R', 'MEPO _L', 'MEPO _R', 'MPO _L', 'MPO _R', 'OV _L', 'OV _R', 'PD _L', 'PD _R', 'PS _L', 'PS _R', 'PVp _L', 'PVp _R', 'PVpo _L', 'PVpo _R', 'SBPV _L', 'SBPV _R', 'SCH _L', 'SCH _R', 'SFO _L', 'SFO _R', 'VMPO _L', 'VMPO _R', 'VLPO _L', 'VLPO _R', 'AHN _L', 'AHN _R', 'AHNa _L', 'AHNa _R', 'AHNc _L', 'AHNc _R', 'AHNp _L', 'AHNp _R', 'LM _L', 'LM _R', 'MM _L', 'MM _R', 'MMme _L', 'MMme _R', 'MMl _L', 'MMl _R', 'MMm _L', 'MMm _R', 'MMp _L', 'MMp _R', 'MMd _L', 'MMd _R', 'SUM _L', 'SUM _R', 'SUMl _L', 'SUMl _R', 'SUMm _L', 'SUMm _R', 'TMd _L', 'TMd _R', 'TMv _L', 'TMv _R', 'MPN _L', 'MPN _R', 'MPNc _L', 'MPNc _R', 'MPNl _L', 'MPNl _R', 'MPNm _L', 'MPNm _R', 'PMd _L', 'PMd _R', 'PMv _L', 'PMv _R', 'PVHd _L', 'PVHd _R', 'PVHdp _L', 'PVHdp _R', 'PVHf _L', 'PVHf _R', 'PVHlp _L', 'PVHlp _R', 'PVHmpv _L', 'PVHmpv _R', 'VMH _L', 'VMH _R', 'VMHa _L', 'VMHa _R', 'VMHc _L', 'VMHc _R', 'VMHdm _L', 'VMHdm _R', 'VMHvl _L', 'VMHvl _R', 'PH _L', 'PH _R', 'LHA _L', 'LHA _R', 'LPO _L', 'LPO _R', 'PST _L', 'PST _R', 'PSTN _L', 'PSTN _R', 'PeF _L', 'PeF _R', 'RCH _L', 'RCH _R', 'STN _L', 'STN _R', 'TU _L', 'TU _R', 'ZI _L', 'ZI _R', 'A13 _L', 'A13 _R', 'FF _L', 'FF _R', 'ME _L', 'ME _R', 'MB_bg _L', 'MB_bg _R', 'SCop _L', 'SCop _R', 'SCsg _L', 'SCsg _R', 'SCzo _L', 'SCzo _R', 'ICc _L', 'ICc _R', 'ICd _L', 'ICd _R', 'ICe _L', 'ICe _R', 'NB _L', 'NB _R', 'SAG _L', 'SAG _R', 'PBG _L', 'PBG _R', 'MEV _L', 'MEV _R', 'SCO _L', 'SCO _R', 'SNr _L', 'SNr _R', 'VTA _L', 'VTA _R', 'PN _L', 'PN _R', 'RR _L', 'RR _R', 'MRN _L', 'MRN _R', 'SCdg _L', 'SCdg _R', 'SCdw _L', 'SCdw _R', 'SCiw _L', 'SCiw _R', 'SCig _L', 'SCig _R', 'SCig-a _L', 'SCig-a _R', 'SCig-b _L', 'SCig-b _R', 'SCig-c _L', 'SCig-c _R', 'PAG _L', 'PAG _R', 'PRC _L', 'PRC _R', 'INC _L', 'INC _R', 'ND _L', 'ND _R', 'Su3 _L', 'Su3 _R', 'APN _L', 'APN _R', 'MPT _L', 'MPT _R', 'NOT _L', 'NOT _R', 'NPC _L', 'NPC _R', 'OP _L', 'OP _R', 'PPT _L', 'PPT _R', 'RPF _L', 'RPF _R', 'CUN _L', 'CUN _R', 'RN _L', 'RN _R', 'III _L', 'III _R', 'MA3 _L', 'MA3 _R', 'EW _L', 'EW _R', 'IV _L', 'IV _R', 'Pa4 _L', 'Pa4 _R', 'VTN _L', 'VTN _R', 'AT _L', 'AT _R', 'LT _L', 'LT _R', 'DT _L', 'DT _R', 'MT _L', 'MT _R', 'SNc _L', 'SNc _R', 'PPN _L', 'PPN _R', 'IF _L', 'IF _R', 'IPN _L', 'IPN _R', 'IPR _L', 'IPR _R', 'IPC _L', 'IPC _R', 'IPA _L', 'IPA _R', 'IPL _L', 'IPL _R', 'IPI _L', 'IPI _R', 'IPDM _L', 'IPDM _R', 'IPDL _L', 'IPDL _R', 'IPRL _L', 'IPRL _R', 'RL _L', 'RL _R', 'CLI _L', 'CLI _R', 'DR _L', 'DR _R', 'P_bg _L', 'P_bg _R', 'NLL _L', 'NLL _R', 'NLLd _L', 'NLLd _R', 'NLLh _L', 'NLLh _R', 'NLLv _L', 'NLLv _R', 'PSV _L', 'PSV _R', 'PB _L', 'PB _R', 'KF _L', 'KF _R', 'PBlc _L', 'PBlc _R', 'PBld _L', 'PBld _R', 'PBle _L', 'PBle _R', 'PBls _L', 'PBls _R', 'PBlv _L', 'PBlv _R', 'PBme _L', 'PBme _R', 'PBmm _L', 'PBmm _R', 'POR _L', 'POR _R', 'SOCm _L', 'SOCm _R', 'SOCl _L', 'SOCl _R', 'B _L', 'B _R', 'DTN _L', 'DTN _R', 'PDTg _L', 'PDTg _R', 'PCG _L', 'PCG _R', 'PG _L', 'PG _R', 'PRNc _L', 'PRNc _R', 'SG _L', 'SG _R', 'SUT _L', 'SUT _R', 'TRN _L', 'TRN _R', 'V _L', 'V _R', 'P5 _L', 'P5 _R', 'Acs5 _L', 'Acs5 _R', 'PC5 _L', 'PC5 _R', 'I5 _L', 'I5 _R', 'CS _L', 'CS _R', 'CSl _L', 'CSl _R', 'CSm _L', 'CSm _R', 'LC _L', 'LC _R', 'LDT _L', 'LDT _R', 'NI _L', 'NI _R', 'PRNr _L', 'PRNr _R', 'RPO _L', 'RPO _R', 'SLC _L', 'SLC _R', 'SLD _L', 'SLD _R', 'MY_bg _L', 'MY_bg _R', 'AP _L',
'AP _R', 'CNlam _L', 'CNlam _R', 'CNspg _L', 'CNspg _R', 'DCO _L', 'DCO _R', 'VCO _L', 'VCO _R', 'CU _L', 'CU _R', 'GR _L', 'GR _R', 'ECU _L', 'ECU _R', 'NTB _L', 'NTB _R', 'NTS _L', 'NTS _R', 'NTSce _L', 'NTSce _R', 'NTSco _L', 'NTSco _R', 'NTSge _L', 'NTSge _R', 'NTSl _L', 'NTSl _R', 'NTSm _L', 'NTSm _R', 'SPVC _L', 'SPVC _R', 'SPVI _L', 'SPVI _R', 'SPVO _L', 'SPVO _R', 'SPVOcdm _L', 'SPVOcdm _R', 'SPVOmdmd _L', 'SPVOmdmd _R', 'SPVOmdmv _L', 'SPVOmdmv _R', 'SPVOrdm _L', 'SPVOrdm _R', 'SPVOvl _L', 'SPVOvl _R', 'Pa5 _L', 'Pa5 _R', 'VI _L', 'VI _R', 'VII _L', 'VII _R', 'ACVII _L', 'ACVII _R', 'AMB _L', 'AMB _R', 'AMBd _L', 'AMBd _R', 'AMBv _L', 'AMBv _R', 'DMX _L', 'DMX _R', 'GRN _L', 'GRN _R', 'ICB _L', 'ICB _R', 'IO _L', 'IO _R', 'IRN _L', 'IRN _R', 'ISN _L', 'ISN _R', 'LIN _L', 'LIN _R', 'LRN _L', 'LRN _R', 'LRNm _L', 'LRNm _R', 'LRNp _L', 'LRNp _R', 'MARN _L', 'MARN _R', 'MDRNd _L', 'MDRNd _R', 'MDRNv _L', 'MDRNv _R', 'PARN _L', 'PARN _R', 'PAS _L', 'PAS _R', 'PGRNd _L', 'PGRNd _R', 'PGRNl _L', 'PGRNl _R', 'NR _L', 'NR _R', 'PRP _L', 'PRP _R', 'PPY _L', 'PPY _R', 'LAV _L', 'LAV _R', 'MV _L', 'MV _R', 'SPIV _L', 'SPIV _R', 'SUV _L', 'SUV _R', 'x _L', 'x _R', 'XII _L', 'XII _R', 'y _L', 'y _R', 'RM _L', 'RM _R', 'RPA _L', 'RPA _R', 'RO _L', 'RO _R', 'CB_bg _L', 'CB_bg _R', 'LING _L', 'LING _R', 'LINGmo _L', 'LINGmo _R', 'LINGgr _L', 'LINGgr _R', 'CENT2 _L', 'CENT2 _R', 'CENT2mo _L', 'CENT2mo _R', 'CENT2gr _L', 'CENT2gr _R', 'CENT3 _L', 'CENT3 _R', 'CENT3mo _L', 'CENT3mo _R', 'CENT3gr _L', 'CENT3gr _R', 'CUL4, 5 _L', 'CUL4, 5 _R', 'CUL4, 5mo _L', 'CUL4, 5mo _R', 'CUL4, 5gr _L', 'CUL4, 5gr _R', 'DEC _L', 'DEC _R', 'DECmo _L', 'DECmo _R', 'DECgr _L', 'DECgr _R', 'FOTU _L', 'FOTU _R', 'FOTUmo _L', 'FOTUmo _R', 'FOTUgr _L', 'FOTUgr _R', 'PYR _L', 'PYR _R', 'PYRmo _L', 'PYRmo _R', 'PYRgr _L', 'PYRgr _R', 'UVU _L', 'UVU _R', 'UVUmo _L', 'UVUmo _R', 'UVUgr _L', 'UVUgr _R', 'NOD _L', 'NOD _R', 'NODmo _L', 'NODmo _R', 'NODgr _L', 'NODgr _R', 'SIM _L', 'SIM _R', 'SIMmo _L', 'SIMmo _R', 'SIMgr _L', 'SIMgr _R', 'ANcr1 _L', 'ANcr1 _R', 'ANcr1mo _L', 'ANcr1mo _R', 'ANcr1gr _L', 'ANcr1gr _R', 'ANcr2 _L', 'ANcr2 _R', 'ANcr2mo _L', 'ANcr2mo _R', 'ANcr2gr _L', 'ANcr2gr _R', 'PRM _L', 'PRM _R', 'PRMmo _L', 'PRMmo _R', 'PRMgr _L', 'PRMgr _R', 'COPY _L', 'COPY _R', 'COPYmo _L', 'COPYmo _R', 'COPYgr _L', 'COPYgr _R', 'PFL _L', 'PFL _R', 'PFLmo _L', 'PFLmo _R', 'PFLgr _L', 'PFLgr _R', 'FL _L', 'FL _R', 'FLmo _L', 'FLmo _R', 'FLgr _L', 'FLgr _R', 'FN_bg _L', 'FN_bg _R', 'IP_bg _L', 'IP_bg _R', 'DN_bg _L', 'DN_bg _R', 'VeCB_bg _L', 'VeCB_bg _R', 'fiber tracts_bg _L', 'fiber tracts_bg _R', 'VS_bg _L', 'VS_bg _R']

general_Region_names = ['FRP _R', 'FRP _L', 'MO _R', 'MO _L', 'SS _R', 'SS _L', 'GU _R', 'GU _L', 'VISC _R', 'VISC _L', 'AUD _R', 'AUD _L', 'VIS _R', 'VIS _L', 'ACA _R', 'ACA _L', 'PL _R', 'PL _L', 'ILA _R', 'ILA _L', 'ORB _R', 'ORB _L', 'AI _R', 'AI _L', 'RSP _R', 'RSP _L', 'PTLp _R', 'PTLp _L', 'TEa _R', 'TEa _L', 'PERI _R', 'PERI _L', 'ECT _R', 'ECT _L', 'MOB _R', 'MOB _L', 'AOB _R', 'AOB _L', 'AON _R', 'AON _L', 'TT _R', 'TT _L', 'DP _R', 'DP _L', 'PIR _R', 'PIR _L', 'NLOT _R', 'NLOT _L', 'COA _R', 'COA _L', 'PAA _R', 'PAA _L', 'TR _R', 'TR _L', 'HIP _R', 'HIP _L', 'RHP _R', 'RHP _L', 'CLA _R', 'CLA _L', 'EP _R', 'EP _L', 'LA _R', 'LA _L', 'BLA _R', 'BLA _L', 'BMA _R', 'BMA _L', 'PA _R', 'PA _L', 'STRd _R', 'STRd _L', 'STRv _R', 'STRv _L', 'LSX _R', 'LSX _L', 'sAMY _R', 'sAMY _L', 'PALd _R', 'PALd _L', 'PALv _R', 'PALv _L', 'PALm _R', 'PALm _L', 'PALc _R', 'PALc _L', 'VENT _R', 'VENT _L', 'SPF _R', 'SPF _L', 'SPA _R', 'SPA _L', 'PP _R', 'PP _L', 'GENd _R', 'GENd _L', 'LAT _R', 'LAT _L', 'ATN _R', 'ATN _L', 'MED _R', 'MED _L', 'MTN _R', 'MTN _L', 'ILM _R', 'ILM _L', 'RT _R', 'RT _L', 'GENv _R', 'GENv _L', 'EPI _R', 'EPI _L', 'PVZ _R', 'PVZ _L', 'PVR _R', 'PVR _L', 'MEZ _R', 'MEZ _L', 'LZ _R', 'LZ _L', 'ME _R', 'ME _L', 'SCs _R', 'SCs _L', 'IC _R', 'IC _L', 'NB _R', 'NB _L', 'SAG _R', 'SAG _L', 'PBG _R', 'PBG _L', 'MEV _R', 'MEV _L', 'SNr _R', 'SNr _L', 'VTA _R', 'VTA _L', 'RR _R', 'RR _L', 'MRN _R', 'MRN _L', 'SCm _R', 'SCm _L', 'PAG _R', 'PAG _L', 'PRT _R', 'PRT _L', 'CUN _R', 'CUN _L', 'RN _R', 'RN _L', 'III _R', 'III _L', 'EW _R', 'EW _L', 'IV _R', 'IV _L', 'VTN _R', 'VTN _L', 'AT _R', 'AT _L', 'LT _R', 'LT _L', 'SNc _R', 'SNc _L', 'PPN _R', 'PPN _L', 'RAmb _R', 'RAmb _L', 'P-sen _R', 'P-sen _L', 'P-mot _R', 'P-mot _L', 'P-sat _R', 'P-sat _L', 'MY-sen _R', 'MY-sen _L', 'MY-mot _R', 'MY-mot _L', 'MY-sat _R', 'MY-sat _L', 'LING _R', 'LING _L', 'CENT _R', 'CENT _L', 'CUL _R', 'CUL _L', 'DEC _R', 'DEC _L', 'FOTU _R', 'FOTU _L', 'PYR _R', 'PYR _L', 'UVU _R', 'UVU _L', 'NOD _R', 'NOD _L', 'SIM _R', 'SIM _L', 'AN _R', 'AN _L', 'PRM _R', 'PRM _L', 'COPY _R', 'COPY _L', 'PFL _R', 'PFL _L', 'FL _R', 'FL _L', 'FN _R', 'FN _L', 'IP _R', 'IP _L', 'DN _R', 'DN _L', 'OLF _R', 'OLF _L', 'CTX _R', 'CTX _L', 'HY _R', 'HY _L', 'TH _R', 'TH _L', 'MB _R', 'MB _L', 'P _R', 'P _L', 'MY _R', 'MY _L', 'CB _R', 'CB _L', 'VS _R', 'VS _L', 'fiber tracts _R', 'fiber tracts _L', 'not detected _R', 'not detected _L', 'CTX_bg _L', 'CTX_bg _R', 'CTXsp_bg _L', 'CTXsp_bg _R', 'STR_bg _L', 'STR_bg _R', 'PAL_bg _L', 'PAL_bg _R', 'TH_bg _L', 'TH_bg _R', 'HY_bg _L', 'HY_bg _R', 'MB_bg _L', 'MB_bg _R', 'P_bg _L', 'P_bg _R', 'MY_bg _L', 'MY_bg _R', 'CB_bg _L', 'CB_bg _R', 'FN_bg _L', 'FN_bg _R', 'IP_bg _L', 'IP_bg _R', 'DN_bg _L', 'DN_bg _R', 'VeCB_bg _L', 'VeCB_bg _R', 'fiber tracts_bg _L', 'fiber tracts_bg _R', 'VS_bg _L', 'VS_bg _R']