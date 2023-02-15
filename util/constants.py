THESIS_PATH = "C:/Users/ikke_/OneDrive/Documenten/Thesis"
RESULTS_DIR = f"{THESIS_PATH}/Results"

###### TASKS 
TASK_501 = 501 # NIH, pancreas
TASK_510 = 510 # MSD, pancreas
TASK_511 = 511 # MSD, pancreas+tumor
TASK_521 = 521 # BTCV, pancreas
TASK_522 = 522 # BTCV, all organs
TASK_523 = 523 # BTCV, pancreas+spleen+veins
TASK_525 = 525 # BTCV, pancreas created again because of stupid results
TASK_526 = 526 # BTCV, all organs with pancreas 1
TASK_527 = 527 # BTCV, pancreas+spleen+veins
TASK_530 = 530 # amos, pancreas
TASK_535 = 535 # amos, all organs
TASK_551 = 551 # NIH, pancreas, cropped for fullres
TASK_561 = 561 # MSD, pancreas, cropped for fullres
TASK_600 = 600 # amos, MRI only, pancreas only
TASK_601 = 601 # amos, MRI only, pancreas only, pretrained by MSD
TASK_605 = 605 # amos, MRI only, all organs
TASK_610 = 610 # P16, batch0, MRI only
TASK_611 = 611 # P16, batch0, MRI only, pretrained AMOS
TASK_612 = 612 # P16, batch1, MRI only  -----> handmatig gedaan!!! niet de echte task612. gemaakt door taks611
TASK_613 = 613 # P16, batch1, MRI only  -----> handmatig gedaan!!! niet de echte task612. gemaakt door task610
TASK_700 = 700 # amos, CT + MRI, all organs
TASK_800 = 800

TASK_NAME_MAPPING = {TASK_501 : "NIH",
                    TASK_510 : "MSD",
                    TASK_525: "BCV",
                    TASK_551 : "NIH",
                    TASK_561 : "MSD",
                    TASK_600 : "AMOS"}


###### CONFIGS
CONFIG_LOW = "3d_lowres"
CONFIG_CAS_FULL = "3d_cascade_fullres"
CONFIG_FULL = "3d_fullres"

###### TRAINING OR TEST
TEST = "Ts"
TRAIN = "Tr"

### TRAINER 
CLASSIC = "nnUNetTrainerV2"
HYBRID = "nnUNetTrainerV2_Hybrid"
HYBRID2 = "nnUNetTrainerV2_Hybrid2"
HYBRID2LR = "nnUNetTrainerV2_Hybrid2LR"
WEIGHT01 = "nnUNetTrainerV2_Loss_DC_CE_weight01"
WEIGHT05 = "nnUNetTrainerV2_Loss_DC_CE_weight05"
WEIGHT09 = "nnUNetTrainerV2_Loss_DC_CE_weight09"
CLASSIC_1500 = "nnUNetTrainerV2_1500"
HYBRID2_1500 = "nnUNetTrainerV2_Hybrid2_1500"
HYBRID2LR_1500 = "nnUNetTrainerV2_Hybrid2LR_1500"

##### MODALITY
ONE = "0000"
TWO = "0001"

CROP_TASK_MAPPING = {TASK_501 : TASK_551,
                     TASK_510 : TASK_561}

MODALITY_MAPPING = {TASK_501 : ONE,
                    TASK_510 : ONE,
                    TASK_521: ONE,
                    TASK_535 : ONE,
                    TASK_551 : ONE,
                    TASK_561 : ONE,
                    TASK_600 : ONE,
                    TASK_605: ONE,
                    TASK_700 : ONE}

                    