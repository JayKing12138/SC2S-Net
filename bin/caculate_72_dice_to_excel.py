import pandas as pd
from collections import defaultdict

# 数据
f1_macro_validate = {'f1_macro_validate': [0.8306327154201751], 'loss_validate': [0.0]}
whole_subject_bundles = defaultdict(
    lambda: [],
    {'AF_left': [0.8594380675748186],
             'AF_right': [0.8527741151534737],
             'ATR_left': [0.8707225056221346],
             'ATR_right': [0.8648531950600576],                                                                                                                                                                 
             'CA': [0.6231370695717646],
             'CC': [0.7849611081830926],
             'CC_1': [0.8659439696604355],
             'CC_2': [0.8142467047839294],
             'CC_3': [0.9105180934062522],
             'CC_4': [0.8024603411724526],
             'CC_5': [0.8708870039705711],
             'CC_6': [0.8285007825763124],
             'CC_7': [0.8524230139438342],
             'CG_left': [0.7694524184270263],
             'CG_right': [0.851390733586855],
             'CST_left': [0.835129472464063],
             'CST_right': [0.8448813280911269],
             'FPT_left': [0.872471640655713],
             'FPT_right': [0.7183011166223356],
             'FX_left': [0.6717808467876664],
             'FX_right': [0.8010886003926193],
             'ICP_left': [0.8094461874650639],
             'ICP_right': [0.7957031154471567],
             'IFO_left': [0.805730166751141],
             'IFO_right': [0.7877542065171956],
             'ILF_left': [0.7720333571548785],
             'ILF_right': [0.8730088872505155],
             'MCP': [0.8648617827313478],
             'MLF_left': [0.8419218798851018],
             'MLF_right': [0.8751158477218611],
             'OR_left': [0.8549952526144435],
             'OR_right': [0.829450184469489],
             'POPT_left': [0.7899221422701382],
             'POPT_right': [0.8515042971754271],
             'SCP_left': [0.8419179621284097],
             'SCP_right': [0.810915545363918],
             'SLF_III_left': [0.8185971405401951],
             'SLF_III_right': [0.834777707102388],
             'SLF_II_left': [0.8416103848171032],
             'SLF_II_right': [0.8029993986581855],
             'SLF_I_left': [0.7959730646323919],
             'SLF_I_right': [0.8117730141455686],
             'STR_left': [0.7990502005490279],
             'STR_right': [0.807551611070657],
             'ST_FO_left': [0.8330817909698792],
             'ST_FO_right': [0.800640857879448],
             'ST_OCC_left': [0.7417433389565284],
             'ST_OCC_right': [0.7829520905902556],
             'ST_PAR_left': [0.818633621648845],
             'ST_PAR_right': [0.7942829010173983],
             'ST_POSTC_left': [0.8985629431750086],
             'ST_POSTC_right': [0.8955251469668613],
             'ST_PREC_left': [0.8673843537691989],
             'ST_PREC_right': [0.8634653011769835],
             'ST_PREF_left': [0.8615896099219457],
             'ST_PREF_right': [0.8445234378192119],
             'ST_PREM_left': [0.8460374207145696],
             'ST_PREM_right': [0.8258067785684037],
             'T_OCC_left': [0.8001204961262358],
             'T_OCC_right': [0.7646617680325436],
             'T_PAR_left': [0.8985261707660056],
             'T_PAR_right': [0.8949822645513695],
             'T_POSTC_left': [0.8611194403398531],
             'T_POSTC_right': [0.8530826227080461],
             'T_PREC_left': [0.8481146931257029],
             'T_PREC_right': [0.8385617987503361],
             'T_PREF_left': [0.8444788618091922],
             'T_PREF_right': [0.8323054048190774],
             'T_PREM_left': [0.8814132580829569],
             'T_PREM_right': [0.8671334441737858],
             'UF_left': [0.8116803718694976],
             'UF_right': [0.8010646047855494]
    }
)

# 创建DataFrame
data = {
    "Metric": list(f1_macro_validate.keys()) + list(whole_subject_bundles.keys()),
    "Value": list(f1_macro_validate.values()) + list(whole_subject_bundles.values()),
}

df = pd.DataFrame(data)

# 保存到Excel文件
df.to_excel("output_data.xlsx", index=False)

print("数据已保存到 output_data.xlsx!")
