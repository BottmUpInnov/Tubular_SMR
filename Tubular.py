"""
Created on Fri Apr 18 09:58:28 2020

@author: Jonathan
"""
import streamlit as st
import pandas as pd
import numpy as np
import math as math
from scipy.integrate import solve_ivp
from bokeh.plotting import figure as fg
st.title('Modélisation des écoulements dans un réacteur tubulaire')
## Standar data
# # Caractéristique physique générale
MH   = 1.00794   # Masse molaire de l'Hydrogène
MC   = 12.01070  # Masse molaire du Carbone
MO   = 15.99940  # Masse molaire de l'Oxygèn
MN   = 14.00674  # Masse molaire de l'Azote
MS   = 32.0660   # Masse molaire du Souffre
MH2O = 2*MH+MO   # Masse molaire de H2O
MCO2 = MC+2*MO   # Masse molaire du CO2
MCO  = MC+MO     # Masse molaire du CO
MO2  = 2*MO      # Masse molaire de O2
MH2  = 2*MH      # Masse molaire de H2
MN2  = 2*MN      # Masse molaire de N2
MS2  = 2*MS      # Masse molaire de S2
MNO2 = MN+2*MO
MCH4 = MC+4*MH
# Constantes physique
T0      = 298.15            # (K) reférence temperature
Patm    = 101325            # (Pa)  Pression atmosphérique en Pascal
p       = 1.01325           # (bar) Pression atmosphérique en bar
Pbar    = 10**5              # si P_in en Pa
R       = 8.3144621         #[J.mol-1.K-1) ou m3·Pa·mol-1·K-1
RKJ     = R/1000            #[kJ.mol-1.K-1)
r       = 0.08206           #[L·atm·mol-1·K-1)
Rcal    = 1.9872036         #[cal.K-1.mol-1]
Rkcal   = 1.9872036 * 1e-3   #[cal.K-1.mol-1]
Rbarcm3 = 83.14             # bar·cm³/mol·K
# PCI
PCIH2  = 241.820 # (KJ/mol)
PCICO  = 283.4   # (KJ/mol)
PCIC   = 393.513 # (KJ/mol)
PCICH4 = 802.620 # (KJ/mol)
    # Equation de Boie : PCI = 348.35C + 938.70H - 108.00O + 62,80N + 104,65S            : que nous utiliserons pour le calcul de la biomasse (PRÖLL et HOFBAUER 2008).
    # Équation de Dulong modifiée : PCI = 327,65C + 1283,39H - 138,26O + 24,18N - 92,55S : que nous utiliserons pour le calcul des alcanes et alcools (NIESSEN 2002).
    # donnera des kJ/mol en multipliant par les pourcentage massique de chaque constituant
# PCI ALCANES
boucle = 12 # limite de n : longeur des chaines carbonnées
nHydro = np.zeros((boucle,3))
k = 0
#Matrice des coéfficients C_nH_{2n+2}O_0
for d in range(0,3):
    b = d+1
    for e in range(0,boucle):
        c = e+1
        nHydro[e,d] = (k+1)*(2-b)*((3-b)/2) + (b-1)*(2*c+2)*(3-b) + 1/2*(b-1)*(b-2)*(0) # remplacer le 0 de la dernière parenthèse avec la loi qui lie l'oxygène au Carbone s'il y en as une
        k=k+1
Mhydtot = np.array([MC,MH,MO])   # (g/mol)
Mhyd    = np.dot(nHydro,Mhydtot) # (g/mol) Vecteur de la Masse molaire totale de chaque hydrocarbure
# Matrice des pourcentage massique pour les alcanes
percentnHydro = np.zeros((boucle,3))
k = 0
for d in range(0,3):
    b = d+1
    for e in range(0,boucle):
        c = e+1
        percentnHydro[e,d] = (k+1)*(2-b)*((3-b)/2)*(MC/Mhyd[e])*100 + (b-1)*(2*c+2)*(3-b)*(MH/Mhyd[e])*100 + 1/2*(b-1)*(b-2)*(0)*(MO/Mhyd[e])*100 # (en #)
        k=k+1
PCIHy         = np.array([349.1,958.3,-103.4]) # Etablie pour du carburant liquides/solides/gaz
VecteurPCIHyd = np.dot(percentnHydro, PCIHy)  # (en kJ/kg) Vecteur PCI avec calcul réel loi de bois (produit de Nbr mole de C,H,O par coéf Boie et diviser par Mtot)
PCIHyd        = VecteurPCIHyd * (Mhyd/1000)    # (en kJ/mol) divisé par 1000 pour avoir la masse molaire de l'alcane en Kg/mol
#PCIHyd       = [-802.620,-1428.640,-2043.110,-2657.320,-3244.940,-3855.100,-4464.730,-5074.150,-5684.550,-6294.220,-6903.600,-7513.680] #PROSIM
# PCI des ALCOOLS
nAlco = np.zeros((boucle,3))
k=0
for d in range(0,3):
    b = d+1
    for e in range(0,boucle):
        c = e+1
        nAlco[e,d] = (k+1)*(2-b)*((3-b)/2) + (b-1)*(2*c+2)*(3-b) + 1/2*(b-1)*(b-2)*(1)# remplacer le 0 de la dernière parenthèse avec la loi qui lie l'oxygène au Carbone s'il y en as une
        k=k+1
MAlcotot = np.array([MC,MH,MO])   # (g/mol)
MAlc     = np.dot(nAlco,MAlcotot) # (g/mol) Vecteur de la Masse molaire totale de chaque Alcool
# Matrice des pourcentage massique pour les alcools
percentAlco=np.zeros((boucle,3))
k=0
for d in range(0,3):
    b = d+1
    for e in range(0,boucle):
        c = e+1
        percentAlco[e,d] = (k+1)*(2-b)*((3-b)/2)*(MC/MAlc[e])*100 + (b-1)*(2*c+2)*(3-b)*(MH/MAlc[e])*100 + 1/2*(b-1)*(b-2)*(1)*(MO/MAlc[e])*100 # (en #)
        k=k+1
PCIAl         = [349.1,958.3,-103.4] # Etablie pour du carburant liquides/solides/gaz
VecteurPCIAlc = np.dot(percentAlco,PCIAl) # (en kJ/kg) Vecteur PCI avec calcul réel loi de bois (produit de Nbr mole de C,H,O par coéf Boie et diviser par Mtot)
PCI_Alc=VecteurPCIAlc * (MAlc/1000) # (en kJ/mol) divisé par 1000 pour avoir la masse molaire de l'alcool en Kg/mol
# PCI_Alc =[-638.200,-1235,-1844,-2454,-3064,-3675,-4285,-4895,-5506,-6116,-6726,-7337]#PROMSIM
# Enthalpie de formation de corps connus en (Kj/mol) (Valeurs issues de webbook.nist.gov)
HfCO2     = -393.51  # (kJ/mol) Enthalpie de formation du CO2
HfCO      = -110.53  # (kJ/mol) Enthalpie de formation du CO
HfH2Og    = -241.82  # (kJ/mol) Enthalpie de formation de l'eau à l'état de gaz
HfH2Ol    = -285.83  # (kJ/mol) Enthalpie de formation de l'eau sous forme liquide et sont état standard  # ATTENTION UTILISÉ POUR ETAT STANDARD
HfO2      = 0        # (kJ/mol) Enthalpie de formation de l'O2
HfH2      = 0        # (kJ/mol) Enthalpie de formation de l'H2
HfN2      = 0
HfCsolide = 0        # (kJ/mol) Enthalpie de formation du Carbone Solide
HfCgaz    = 716.7    # (kJ/mol) Enthalpie de formation du Carbone gaz
HfNO2     = 331.80   # (kJ/mol) Enthalpie de formation du NO2
HfSO2     = -296.840 # (kJ/mol) Enthalpie de formation du SO2
HfCH4     = -74.520 # (kJ/mol) Enthalpie de formation du CH4
# ALCANES
VecteurHfHyd = np.array([-74.520,-83.820,-104.680,-125.790,-173.510,-198.660,-224.050,-249.780,-274.680,-300.620,-326.600,-352.130]) # Prosim Enthalpie de formation état standard à 25°C
# ALCOOL
VecteurHfAlc = np.array([-239.1,-276.98,-300.8,-326.4,-351.9,-377.4,-403,-428.5,-454.1,-479.6,-505.1,-530.7]) # Prosim Enthalpie de formation état standard à 25°C
# Température pour état gazeux (en K)
THydgaz     = np.array([111.66,184.55,231.11,272.65,309.22,341.88,371.58,398.83,423.97,447.305,469.078,489.473])
TAlcgaz     = np.array([337.85,351.44,370.35,391.9,410.9,429.9,448.6,467.1,485.2,503,520.3,534.2])
TCO         = 81.63
TVapH2O     = 373.15
TN2 = 77.34
Tcombustion = 1273.15
# Capacité calorifique en J/(mol*K) : Coefficient issue de webbook.nist.gov
# Cp**° = A + B*t + C*t^2 +^ D*t^3 &&        (J/mol*K)
#    CP     = [     D            C              B            A     ]
#coefCpO2    =([1.05523E-10,-1.30899E-06,0.006633418,29.00113091]) #NIST
#coefCpH2    =([1.23828E-11,-4.38209E-07,0.004814927,26.17077672]) #NIST
#coefCpCO    =([1.48773E-10,-1.85019E-06,0.007714251,26.94113761]) #NIST
#coefCpH2Og  =([2.73802E-10,-3.83649E-06,0.018940479,26.21513543]) #NIST
#coefCpCO2   =([3.71136E-10,-4.46606E-06,0.017845691,39.52934079]) #NIST
#coefCpC     =([-4.2104E-11,4.34863E-07,-0.000761443,21.11569558]) #NIST
#coefCpN2    =([1.50023E-10,-1.87779E-06,0.007903931,26.38903761]) #NIST

coefCpC     = np.array([-4.05920E-11,4.18539E-07,-7.08751E-04,2.10677E+01])
coefCpCO    = np.array([-5.21981E-09,1.31000E-05,-4.09918E-03,2.93459E+01])
coefCpCO2   = np.array([9.86340E-10,-9.91857E-06,3.19890E-02,2.95590E+01])
coefCpH2    = np.array([1.16783E-09,-1.37914E-06,2.21141E-03,2.82389E+01])
coefCpH2Og  = np.array([-2.80903E-09,9.39009E-06,1.87828E-03,3.25909E+01])
coefCpN2    = np.array([-4.61790E-09,1.22128E-05,-4.36263E-03,2.94031E+01])
coefCpO2    = np.array([-5.17172E-09,1.05702E-05,7.70289E-04,2.87185E+01])
coefCpNO2   = np.array([-5.70288E-09,2.55131E-06,0.025021883,30.458977])
coefCpCH4   = np.array([3.13202E-08,7.00377E-05,2.80211E-03,3.13702E+01])
# Coefficients calculés à partir des tabulation de valeurs issus de Prosim
# ALCANES :
coefCpHyd = np.array([[-3.13202E-08,7.00377E-05,2.80211E-03,3.13702E+01],
                      [-3.11626E-08,4.07617E-05,8.65107E-02,2.70603E+01],
                      [-1.5789E-08,-1.49755E-05,1.86047E-01,2.38998E+01],
                      [-4.95069E-09,-7.41651E-05,2.81242E-01,2.58623E+01],
                      [2.56050E-08,-1.81924E-04,4.26550E-01,1.15809E+01],
                      [4.26670E-08,-2.49420E-04,5.28636E-01,9.84151E+00],
                      [5.55572E-08,-3.07757E-04,6.25390E-01,9.08857E+00],
                      [6.75516E-08,-3.65856E-04,7.22660E-01,8.10457E+00],
                      [8.22822E-08,-4.28349E-04,8.21600E-01,7.08265E+00],
                      [-3.13202E-08,7.00377E-05,2.80211E-03,3.13702E+01],
                      [6.56853E-08,-4.12749E-04,9.05554E-01,3.05250E+01],
                      [7.28239E-08,-4.52324E-04,9.86836E-01,3.32925E+01]])#PROSIM

# ALCOOLS :
coefCpAlc = np.array([[-2.57170E-08,4.13847E-05,4.57843E-02,2.89423E+01],
                    [-1.24589E-08,-2.41932E-05,1.58490E-01,2.18402E+01],
                    [-2.82831E-09,-7.53780E-05,2.56517E-01,1.60168E+01],
                    [3.33889E-08,-1.99492E-04,4.13170E-01,6.15547E-01],
                    [2.83675E-08,-2.15744E-04,4.82578E-01,3.47867E+00],
                    [4.57131E-08,-2.89751E-04,5.95658E-01,-2.54549E+00],
                    [6.12785E-08,-3.59440E-04,7.07100E-01,-8.37617E+00],
                    [7.85769E-08,-4.33564E-04,8.20100E-01,-1.43042E+01],
                    [9.53942E-08,-5.05736E-04,9.31771E-01,-2.00294E+01],
                    [1.11806E-07,-5.77737E-04,1.04344E+00,-2.56705E+01],
                    [1.28503E-07,-6.49833E-04,1.15550E+00,-3.14534E+01],
                    [1.44634E-07,-7.21116E-04,1.26712E+00,-3.70454E+01]])#PROSIM

VecTint0   = np.array([(298.15)**4,(298.15)**3,(298.15)**2,(298.151)])
VecTintdT0 = np.array([(298.15)**4,(298.15)**3,(298.15)**2,(298.15)]) # remplacer VecTint0 par VecTintdT0 dans les cas ou VectcoefdT est utilisé (betavar en générale pour alerger le calcul)
VecTint02  = np.array([(298.15)**3,(298.15)**2,(298.15),math.log(298.15)])
VectcoefT  = np.array([1/3,1/2,1,1])   # coéficient de l'intégrale du vecteur DT/T à multiplié par VecTint2=[T^3T^2Tlog(T)]
VectcoefdT = np.array([1/4,1/3,1/2,1]) # coéficient de l'intégrale du vecteur DT à multiplié par VecTint=[T^4T^3T^2T1]
# Entropie de formation en J/(mol*K) ou entropie molaire standards à 298 K
SfO2    = 205.043
SfH2    = 130.571
SfCO2   = 213.677
SfCO    = 197.566
SfH2Og  = 188.724
SfH2Ol  = 70.033
SfCgaz  = 157.991
SfN2    = 191.609
SfNO2   = 239.92
SfCH4   = 186.270
SfHyd   = np.array([186.270,229.120,270.200,309.910,263.470,296.060,328.570,361.200,393.670,425.890,458.150,490.660]) #PROSIM Entropie absolue état standard à 25°C
SfAlc   = np.array([127.19,159.86,193.6,225.8,257.6,289.6,321.7,353.7,385.7,417.7,449.8,481.8])   #PROSIM Entropie absolue état standard à 25°C
# Enthalpie libre de formation en J/(mol)
GfCsolide = 0    #(J/mol) PROSIM
GfCgaz    = 0# 671 290  #(J/mol) PROSIM
GfO2      = 0       #(J/mol) PROSIM
GfH2      = 0       #(J/mol) PROSIM
GfCO      = -137150   #(J/mol) PROSIM
GfCO2     = -394370 #(J/mol) PROSIM
GfH2Og    = -228590 #(J/mol) PROSIM
GfH2Ol    = -237214 #(J/mol) PROSIM
GfN2      = 0
GfNO2     =  51328
GfCH4     =  -50490
GfHyd     = np.array([-50490,-31920,-24390,-16700,-9928,-4154,1404,6587,12647,17740,22780,28203]) # (J/mol) Prosim Energie de Gibbs de formation état standard à 25°C
GfAlc     = np.array([-166900,-173860,-167000,-161400,-155800,-150200,-144700,-139000,-133500,-127900,-122300,-116700])       # (J/mol) Prosim Energie de Gibbs de formation état standard à 25°C
# Exergie Chimique kJ/mol
ExCsolide = 410.25
ExCgaz    = 410.25
ExO2      = 3.97
ExH2      = 236.10
ExCO      = 275.10
ExCO2     = 19.87
ExH2Og    = 9.49
ExN2      = 0.72
ExCH4     = 831.96
ExHyd     = np.array([831.96,1496.88,2150.76,2804.80,3457.92,4110.05,4761.95,5413.49,6065.90,6717.34,7368.73,8020.50])
ExAlc     = np.array([717.535,1356.925,2010.135,2662.085,3314.035,3965.985,4617.835,5269.885,5921.735,6573.685,7225.635,7844.085])
# constante Cp at specifique pressure
coefCpH2s    = (29.95+29.55)/2 #(entre 750 et 900k)    J mol-1 K-1 https://webbook.nist.gov/cgi/fluid.cgi?P=29&TLow=740&THigh=900&TInc=10&Applet=on&Digits=5&ID=C1333740&Action=Load&Type=IsoBar&TUnit=K&PUnit=bar&DUnit=mol#2Fl&HUnit=kJ#2Fmol&WUnit=m#2Fs&VisUnit=uPa*s&STUnit=N#2Fm&RefState=DEF
coefCpN2s    = (32.4+31.2)/2   #(entre 750 et 900k)    J mol-1 K-1 https://webbook.nist.gov/cgi/fluid.cgi?P=29&TLow=750&THigh=900&TInc=10&Applet=on&Digits=5&ID=C7727379&Action=Load&Type=IsoBar&TUnit=K&PUnit=bar&DUnit=mol#2Fl&HUnit=kJ#2Fmol&WUnit=m#2Fs&VisUnit=uPa*s&STUnit=N#2Fm&RefState=DEF
coefCpCOs    = (30.07+30.25)/2 #(entre 298 et 500k)    !!! J mol-1 K-1 Vérifier à température plus élevé https://webbook.nist.gov/cgi/fluid.cgi?P=29&TLow=298&THigh=500&TInc=10&Applet=on&Digits=5&ID=C630080&Action=Load&Type=IsoBar&TUnit=K&PUnit=bar&DUnit=mol#2Fl&HUnit=kJ#2Fmol&WUnit=m#2Fs&VisUnit=uPa*s&STUnit=N#2Fm&RefState=DEF
coefCpH2Os   = (41.1+40.4)/2   #(entre 750 et 900k)    J mol-1 K-1 https://webbook.nist.gov/cgi/fluid.cgi?P=29&TLow=750&THigh=900&TInc=10&Applet=on&Digits=5&ID=C7732185&Action=Load&Type=IsoBar&TUnit=K&PUnit=bar&DUnit=mol#2Fl&HUnit=kJ#2Fmol&WUnit=m#2Fs&VisUnit=uPa*s&STUnit=N#2Fm&RefState=DEF
coefCpCO2s   = (53.5+51)/2     #(entre 750 et 900k)    J mol-1 K-1 https://webbook.nist.gov/cgi/fluid.cgi?P=29&TLow=750&THigh=900&TInc=10&Applet=on&Digits=5&ID=C124389&Action=Load&Type=IsoBar&TUnit=K&PUnit=bar&DUnit=mol#2Fl&HUnit=kJ#2Fmol&WUnit=m#2Fs&VisUnit=uPa*s&STUnit=N#2Fm&RefState=DEF
coefCpCH4s   = (55+37.5)/2     #(entre 298 et 650k)    !!! J mol-1 K-1 ATTENTION grosse variation
coefCpO2s    = (33.55+34.47)/2 #(entre 298 et 650k)    J mol-1 K-1 https://webbook.nist.gov/cgi/fluid.cgi?P=30&TLow=750&THigh=900&TInc=10&Applet=on&Digits=5&ID=C7782447&Action=Load&Type=IsoBar&TUnit=K&PUnit=bar&DUnit=mol#2Fl&HUnit=kJ#2Fmol&WUnit=m#2Fs&VisUnit=uPa*s&STUnit=N#2Fm&RefState=DEF
# Viscosité                                                         [Pa.s]
MuCH4   = np.array([2.39E-02,5.28])
MuH2O   = np.array([4.13E-02,-3.52])
MuCO2   = np.array([3.31E-02,8.40])
MuCO    = np.array([3.83E-02,6.90])
MuN2    = np.array([2.98E-02,11.97])
MuH2    = np.array([1.52E-02,5.27])
MuO2    = np.array([7.12E-05,0.01])
# Termal conductivity                                              [W/m.K]
CH4_cond    = np.array([2.03E-04,-3.21E-02])
H2O_cond    = np.array([1.18E-04,-2.02E-02])
CO2_cond    = np.array([7.18E-05,-7.28E-04])
CO_cond     = np.array([6.21E-05,8.61E-03])
N2_cond     = np.array([5.46E-05,1.18E-02])
H2_cond     = np.array([4.88E-04,3.68E-02])
O2_cond     = np.array([7.12E-05,9.13E-03])
## Catalyst data
alpha=1
# Constante d'adsorption de référence (adsorption constant) - bar^(-1)
KCO_648  = alpha*40.91
KH2_648  = alpha*0.02960
KCH4_823 = alpha*0.1791
KH2O_823 = alpha*0.4152
# Constante de taux (rate constants) de référence - bar^(1/2)/(kmol.kg_cat.h)
k_1_648 = alpha*1.842e-4
k_2_648 = alpha*7.558
k_3_648 = alpha*2.193e-5
# Énergie d'activation - kJ/mol
E_1 = alpha*240.1
E_2 = alpha*67.13
E_3 = alpha*243.9
# Enthalpie de changement pour l'adsorption (enthalpy change of adsorption) - kJ/mol
DHCO  = -70.65*alpha
DHH2  = -82.90*alpha
DHCH4 = -38.28*alpha
DHH2O =  88.68*alpha
## Reactor data
# Reference modelling parameter
Z           = 11.2     # (m) Refomer total length
step        = 500      # Thickness step
d           = 0.1016   # (m) Refomer total diameter
A_c         = math.pi*d**2/4 # (m^2) cross-sectional area of the reactor tube
thickness   = 0.005    # hypothèse abstraite
di_comb = d+2*thickness # [m] Furnace internal diameter
k_reactor   = 26       # [W/m.K] thermal conductivity
# Reference reactor parameters
rho_c       = 2355.2   # (kg.m-3) Catalyst density
epsi        = 0.65     # Catalyst void fraction
D_p         = 5e-3     # (m) Catalyst pellet diameter
Wc_tot      = 74.32    # (kg) Total catalyst weight
U           = 100      # (J/K.m2.s) Overall heat transfer coefficient
mu          = 3.720e-05# (kg/m s) Reaction mixture viscosity
#  facteur d'éfficacité des réactions (efficiency factor of reaction)  - sans unité
mufactor    = 1
mu_I        = 0.03*mufactor
mu_II       = 0.03*mufactor
mu_III      = 0.03*mufactor
## Inlet SMR data
# Ref. Nummedal2005
FCH4_in = 1.436    # (mol.s-1) Inlet methane molar flow rate
FH2O_in = 4.821    # (mol.s-1) Inlet water molar flow rate
FH2_in  = 1.751e-1 # (mol.s-1) Inlet hydrogen molar flow rate
FCO_in  = 2.778e-3 # (mol.s-1) Inlet carbon monoxide molar flow rate
FCO2_in = 8.039e-2 # (mol.s-1) Inlet carbon dioxide molar flow rate
FN2_in  = 2.354e-1 # (mol.s-1) Inlet nitrogen molar flow rate
T_in    = 793.15   # (K) Inlet temperature of reaction mixture
P_in    = 29e5     # (Pa) Inlet total pressure
p_in    = 29       # (bar) Inlet total pressure
Wc_in   = 0
# total flow rate
Ft_in   = FCH4_in + FH2O_in + FH2_in + FCO_in + FCO2_in + FN2_in
# entering superficial gas velocity
v_0     = (((FCH4_in + FH2O_in + FH2_in + FCO_in + FCO2_in + FN2_in)* R * T_in)/P_in) /A_c  # [m.s^(-1)]
## Definition Function Reformer_func_ref
def Reformer_func_ref(dz,y):
    ypoint = np.zeros(8)
    # Assignation
    FCH4 = y[0]
    FH2O = y[1]
    FH2  = y[2]
    FCO  = y[3]
    FCO2 = y[4]
    FN2  = y[5]
    T    = y[6]
    P    = y[7]
    # Vecteur (T,1) pour calculs
    TT       = np.array([T,1])
    VecT     = np.array([T**3,T**2,T,1])
    VecTint  = np.array([T**4,T**3,T**2,T,1])
    VecTint2 = np.array([T**3,T**2,T,math.log(T)])
    # Furnace temperature dz
    T_a = ((dz/Z)*170+990) #dz
    # Reactor dimenssions                                                 [m2]
    Sint          = math.pi*d*dz
    Sint_Furnace  = math.pi*di_comb*dz
    # dWc/dz
    dWcdz=A_c*rho_c*(1-epsi)
    # Total molar flow rate                                            [mol/s]
    Ft = FCH4 + FH2O + FH2 + FCO + FCO2 + FN2
    # Mass flow                                                         [kg/s]
    Flowkg  = (FCH4 *MCH4 + FH2O *MH2O + FCO *MCO + FCO2 *MCO2 + FN2 *MN2 + FH2 *MH2)/1000
    # Molar fraction                                                       [-]
    xCH4  = FCH4/Ft
    xH2O  = FH2O/Ft
    xH2   = FH2 /Ft
    xCO   = FCO /Ft
    xCO2  = FCO2/Ft
    xN2   = FN2 /Ft
    # Molar mass of a mixture                                          [g/mol]
    W     = xCH4 *MCH4 + xH2O *MH2O + xCO *MCO + xCO2 *MCO2 + xN2 *MN2 + xH2 *MH2
    Wkg   = W/1000                                              #[kg/mol]
    # Partial pressure                                                   [bar]
    PCH4 = xCH4*P/Pbar
    PH2O = xH2O*P/Pbar
    PH2  = xH2 *P/Pbar
    PCO  = xCO *P/Pbar
    PCO2 = xCO2*P/Pbar
    PN2  = xN2 *P/Pbar
    # Volume flow rate                                                  [m3/s]
    Q = Ft*R*T/P
    # Volume of reactor                                                   [m3]
    V  = A_c*dz
    # Reactor speed  / Superficial gas velocity                          [m/s]
    v = Q/A_c
    # superficial mass velocity                                      [kg/s.m2]
    G0 = ((FCH4*MCH4 + FH2O*MH2O + FH2*MH2+ FCO*MCO+ FCO2*MCO2 + FN2*MN2)/(1000*A_c))
    # Density of a mixture                                             [kg/m3]
    density= (Wkg*P) / (R*T)
    # Viscosité constante                                          [microPa.s]
    Mu  = (xCH4 * np.dot(TT,MuCH4) + xH2O * np.dot(TT,MuH2O) + xH2 * np.dot(TT,MuH2) + xCO * np.dot(TT,MuCO) + xCO2 * np.dot(TT,MuCO2) +xN2 * np.dot(TT,MuN2))*10**(-6)
    # Termal conductivity                                              [W/m.K]
    ki  = (xCH4 * np.dot(TT,CH4_cond) + xH2O * np.dot(TT,H2O_cond) + xH2 * np.dot(TT,H2_cond) + xCO * np.dot(TT,CO_cond) + xCO2 * np.dot(TT,CO2_cond) +xN2*TT*N2_cond)
    # Heat capacity
    xCp = xCH4*(coefCpCH4s) + xH2O*(coefCpH2Os) + xH2*(coefCpH2s) + xCO*(coefCpCOs) + xCO2*(coefCpCO2s) +xN2*(coefCpN2s) #[J.mol-1.K-1]
    FCp = FCH4*(coefCpCH4s) + FH2O*(coefCpH2Os) + FH2*(coefCpH2s) + FCO*(coefCpCOs) + FCO2*(coefCpCO2s) +FN2*(coefCpN2s) #[J.K-1]
    CP  = xCp/(Wkg)                                                                                                      #[J.kg-1.K-1]
    # Prandtl number                                                       [-]
    Pr  = CP*Mu/ki
    # Reynolds number                                                      [-]
    Re  = density*v*d/Mu
    # Nusselt number
    if Re > 2500*np.ones(np.shape(Re)):
        Nu = 0.023*(Re**(0.8))*Pr**(0.4) # Turbulent
    else:
        Nu = 0.33*Pr**(1/3)*Re**0.5      # Laminaire
    # Internal convective heat transfer coefficient                   [W/K.m2]
    Ui  = Nu * (ki/d)
    # Heat resistance of reactor tube                                    [K/W]
    Rcyl    = math.log((di_comb)/(d))/(2*math.pi*dz*k_reactor)
    # External convective heat-transfer coefficient                   [W/m2/K]
    Ue    = 105
    # Overall heat transfer coefficient                                [W/K-1]
    Unew  = 1/(Sint*(1/(Sint*Ui) + Rcyl + 1/(Sint_Furnace*Ue)))
    # Constante de réaction
    Qr_I   = ((((PH2)**3)*(PCO ))/((PCH4)*(PH2O)))
    Qr_II  = ((((PH2))   *(PCO2))/((PCO) *(PH2O)))
    Qr_III = ((((PH2)**4)*(PCO2))/((PCH4)*((PH2O)**2)))
    # Enthalpie standard de réaction                                    [kJ/s]
    Hr0_I   = HfCO  + 3*HfH2 - HfCH4 - HfH2Og
    Hr0_II  = HfCO2 + HfH2   - HfCO  - HfH2Og
    Hr0_III = HfCO2 + 4*HfH2 - HfCH4 - 2*HfH2Og
    # Entropie standard de réaction                                     [kJ/s]
    Sr0_I   = SfCO  + 3*SfH2 - SfCH4 - SfH2Og
    Sr0_II  = SfCO2 + SfH2   - SfCO  - SfH2Og
    Sr0_III = SfCO2 + 4*SfH2 - SfCH4 - 2*SfH2Og
    # Enthalpie libre standard de réaction                                 [J]
    Gr0_I   = 1000*Hr0_I   - T*Sr0_I
    Gr0_II  = 1000*Hr0_II  - T*Sr0_II
    Gr0_III = 1000*Hr0_III - T*Sr0_III
    # CP dT                                                             [J]
    Dcpc_I   = (coefCpCOs  + 3*coefCpH2s - coefCpCH4s   - coefCpH2Os)  *(T-T0)
    Dcpc_II  = (coefCpCO2s + coefCpH2s   - coefCpCOs    - coefCpH2Os)  *(T-T0)
    Dcpc_III = (coefCpCO2s + 4*coefCpH2s - 1*coefCpCH4s - 2*coefCpH2Os)*(T-T0)
    # CP dT/T                                                         [J/K]
    DcpdT_I   = (coefCpCOs  + 3*coefCpH2s - coefCpCH4s - coefCpH2Os)  *(math.log(T/T0))
    DcpdT_II  = (coefCpCO2s + coefCpH2s   - coefCpCOs  - coefCpH2Os)  *(math.log(T/T0))
    DcpdT_III = (coefCpCO2s + 4*coefCpH2s - coefCpCH4s - 2*coefCpH2Os)*(math.log(T/T0))
    # Enthalpie de réaction                                                [J]
    DrHr_I    = 1000*Hr0_I   + Dcpc_I
    DrHr_II   = 1000*Hr0_II  + Dcpc_II
    DrHr_III  = 1000*Hr0_III + Dcpc_III
    # Enthalpie libre de réaction                                          [J]
    DrG_I    = Gr0_I     + Dcpc_I    - T * DcpdT_I
    DrG_II   = Gr0_II    + Dcpc_II   - T * DcpdT_II
    DrG_III  = Gr0_III   + Dcpc_III  - T * DcpdT_III
    # Constante d'équilibre
    K_I     = math.exp(-((DrG_I)   / (R * T)))
    K_II    = math.exp(-((DrG_II)  / (R * T)))
    K_III   = math.exp(-((DrG_III) / (R * T)))
    # rate constant with [kmol.bar^(1/2).kg cat^(-1).h^(-1)],[kmol.kg cat^(-1).h^(-1).bar^(-1)] and [kmol.bar^(1/2).kg cat^(-1).h^(-1)] - RKJ in  KJ/mol.K
    k_I   = (k_1_648*math.exp(-(E_1/RKJ)*((1/T)-(1/648))))
    k_II  = (k_2_648*math.exp(-(E_2/RKJ)*((1/T)-(1/648))))
    k_III = (k_3_648*math.exp(-(E_3/RKJ)*((1/T)-(1/648))))
    # Adsorption constant with [bar^(-1)] except KH2O dimenssionless - RKJ in  KJ/mol.K
    KCO   = (KCO_648 *math.exp(-(DHCO /RKJ)*((1/T)-(1/648))))
    KH2   = (KH2_648 *math.exp(-(DHH2 /RKJ)*((1/T)-(1/648))))
    KCH4  = (KCH4_823*math.exp(-(DHCH4/RKJ)*((1/T)-(1/823))))
    KH2O  = (KH2O_823*math.exp(-(DHH2O/RKJ)*((1/T)-(1/823))))
    # DEN [?]
    DEN   = (1 + KCO*PCO + KH2*PH2 + KCH4*PCH4 + (KH2O*PH2O)/PH2)
    # reaction rates mol.kg cat^(-1).s^(-1)
    rI    = (1000/3600)*mu_I  *(((k_I)  /((PH2**(2.5))))*((PCH4*PH2O)  -(((PH2**3)*PCO) /(K_I)))  )/(DEN**2)
    rII   = (1000/3600)*mu_II *(((k_II) /( PH2       ))*((PCO *PH2O)  -((PH2    *PCO2)/(K_II))) )/(DEN**2)
    rIII  = (1000/3600)*mu_III*(((k_III)/((PH2**(3.5))))*((PCH4*PH2O**2)-(((PH2**4)*PCO2)/(K_III))))/(DEN**2)
    # Mole balances - dF/dWc - Nummedal expression
    ypoint[0] = dWcdz* (- rI         - rIII)
    ypoint[1] = dWcdz* (- rI - rII - 2*rIII)
    ypoint[2] = dWcdz* (3*rI + rII + 4*rIII)
    ypoint[3] = dWcdz* (  rI - rII)
    ypoint[4] = dWcdz* (       rII +   rIII)
    ypoint[5] = 0
    # energy balance
    if T_a-T < 5:
        ypoint[6] = 0;
    else:
        ypoint[6] = dWcdz*(((1/(rho_c*(1-epsi)))*(4/d)*U*(T_a-T)-(rI*DrHr_I + rII*DrHr_II + rIII*DrHr_III))/FCp)
    # pressure drop or momentum balance
    ypoint[7] = dWcdz*(-(((150*mu*(1-epsi))/D_p)+1.75*G0)*(1/(D_p*(epsi**3)*A_c*rho_c))*v_0*(Ft/Ft_in)*(P_in/P)*(T/T_in))
    return ypoint

Z       = st.sidebar.slider('Longueur du réacteur',min_value=1.0, max_value=50.00, value=11.2)
T_in    = st.sidebar.slider('Température en entrée', min_value=500.00, max_value=1100.00, value=793.00)
FCH4_in = st.sidebar.slider('Quantité de méthane en entrée',min_value=.1, max_value=5.00, value=1.436)
FH2O_in = st.sidebar.slider('Quantité de eau en entrée',min_value=.1, max_value=10.00, value=4.821)
FH2_in  = st.sidebar.slider('Quantité de hydrogène en entrée',min_value=.1, max_value=5.00, value=0.1751)
## Solve T furnace Cas étudié par Nummedal2005
# Lengh of reactor division for integration
Ini      = 0.01
fineness = 0.01
nbstep   = int((Z-Ini)/fineness)
dz    = np.linspace(Ini,Z,nbstep)
tspan = np.linspace(Ini, dz, np.size(dz))
sol = solve_ivp(Reformer_func_ref, [Ini, Z], [FCH4_in,FH2O_in,FH2_in,FCO_in,FCO2_in,FN2_in,T_in,P_in], method='RK45', t_eval=dz)
DZ   = sol.t
FCH4 = sol.y[0]
FH2O = sol.y[1]
FH2  = sol.y[2]
FCO  = sol.y[3]
FCO2 = sol.y[4]
FN2  = sol.y[5]
T    = sol.y[6]
P    = sol.y[7]
T_a = ((DZ/Z)*170+990)

st.sidebar.markdown(f"Température de réaction = {T_in}  K")
st.sidebar.markdown(f"Longueur du réacteur = {Z} m")

p = fg(
    title="Températures dans le réacteur",
    x_axis_label="Position dans le réacteur",
    y_axis_label="Température",
    # match_aspect=True,
    tools="pan,reset,save,wheel_zoom",
)
p.line(DZ, T, color="#1f77b4", line_width=3, line_alpha=0.6)
p.line(DZ, T_a, color="#ff7f0e", line_width=3, line_alpha=0.6)
# p.xaxis.fixed_location = 0
# p.yaxis.fixed_location = 0
st.bokeh_chart(p)

#@st.cache
# import plotly_express as px
# fig1 = px.line(x=DZ, y=T, labels={'x':'Longueur du réacteur (m)', 'y':'Température (K)'})
# ts_chart = st.plotly_chart(fig1)

# import plotly.graph_objects as go
# fig = go.Figure()

# Add traces
# fig1 = go.Figure()
# fig1.add_trace(go.Scatter(x=DZ, y=FCH4,
#               mode='lines',
#               name='méthane'))
# fig1.add_trace(go.Scatter(x=DZ, y=FH2O,
#               mode='lines',
#               name='eau'))
# fig1.add_trace(go.Scatter(x=DZ, y=FH2,
#               mode='lines',
#               name='hydrogène'))
# fig1.add_trace(go.Scatter(x=DZ, y=FCO,
#               mode='lines',
#               name='Monoxyde de carbone'))
# fig1.add_trace(go.Scatter(x=DZ, y=FCO2,
#               mode='lines',
#               name='Dioxyde de carbone'))
# st.write(fig1)
#
# fig.add_trace(go.Scatter(x=DZ, y=T,
#               mode='lines',
#               name='T reactor'))
# fig.add_trace(go.Scatter(x=DZ, y=T_a,
#               mode='lines',
#               name='T furnace'))
# st.write(fig)
