GENERAL = [
    'age', 'itt_treat', 'gender', 'weight', 'glucose'
]

GENERAL_FORMATS = {
    'age': float,
    'itt_treat': bool,
    'gender': 'gender'
}

RAND_FORM = [
    'livealone_rand',         # Lived alone before stroke? YNDQ
    'indepinadl_rand',        # Independent in ADL before stroke? YNDQ
    'infarct',                # Recent ischaemic change likely cause of this stroke? INFARCT
    'antiplat_rand',          # Received antiplatelet drugs in last 48 hours? YNDQ
    'atrialfib_rand',         # Patient in atrial fibrillation at randomisation? YNDQ
    'sbprand',                # Systolic BP at randomisation (mm Hg)
    'dbprand',                # Diastolic BP at randomisation (mm Hg)
    'weight',                 # Estimated weight (kg)
    'glucose',                # Blood glucose (mmol/L)
    'gcs_eye_rand',           # Best eye response (Glasgow Coma Scale) at randomisation GCSEYE
    'gcs_motor_rand',         # Best motor response (Glasgow Coma Scale) at randomisation GCSMOTOR
    'gcs_verbal_rand',        # Best verbal response (Glasgow Coma Scale) at randomisation GCSVERBAL
    'gcs_score_rand',         # Total Glasgow Coma Scale score at randomisation
    'nihss',                  # Total NIH Stroke Score at randomisation
    'liftarms_rand',          # Able to lift both arms off bed at randomisation YNDQ
    'ablewalk_rand',          # Able to walk without help at randomisation YNDQ
    'weakface_rand',          # Unilateral weakness affecting face at randomisation YNDQ
    'weakarm_rand',           # Unilateral weakness affecting arm or hand at randomisation YNDQ
    'weakleg_rand',           # Unilateral weakness affecting leg or foot at randomisation YNDQ
    'dysphasia_rand',         # Dysphasia at randomisation YNDQ
    'hemianopia_rand',        # Homonymous hemianopia at randomisation YNDQ
    'visuospat_rand',         # Visuospatial disorder at randomisation YNDQ
    'brainstemsigns_rand',    # Brainstem or cerebellar signs at randomisation YNDQ
    'otherdeficit_rand',      # Other neurological deficit at randomisation YNDQ
    'stroketype',             # Stroke subtype STROKETYPE
    'pred_nihss',             # NIHSS predicted? Y01N
    'konprob',                # Probability of good outcome based on Konig model
]


RAND_FORM_FORMATS = {
    'livealone_rand': 'YNDQ',           # Lived alone before stroke? YNDQ
    'indepinadl_rand': 'YNDQ',          # Independent in ADL before stroke? YNDQ
    'infarct': 'INFARCT',               # Recent ischaemic change likely cause of this stroke? INFARCT
    'antiplat_rand': 'YNDQ',            # Received antiplatelet drugs in last 48 hours? YNDQ
    'atrialfib_rand': 'YNDQ',           # Patient in atrial fibrillation at randomisation? YNDQ
    'sbprand': float,                   # Systolic BP at randomisation (mm Hg)
    'dbprand': float,                   # Diastolic BP at randomisation (mm Hg)
    'weight': float,                    # Estimated weight (kg)
    'glucose': float,                   # Blood glucose (mmol/L)
    'gcs_eye_rand': 'GCSEYE',           # Best eye response (Glasgow Coma Scale) at randomisation GCSEYE
    'gcs_motor_rand': 'GCSMOTOR',       # Best motor response (Glasgow Coma Scale) at randomisation GCSMOTOR
    'gcs_verbal_rand': 'GCSVERBAL',     # Best verbal response (Glasgow Coma Scale) at randomisation GCSVERBAL
    'gcs_score_rand': float,            # Total Glasgow Coma Scale score at randomisation
    'nihss': float,                     # Total NIH Stroke Score at randomisation
    'liftarms_rand': 'YNDQ',            # Able to lift both arms off bed at randomisation YNDQ
    'ablewalk_rand': 'YNDQ',            # Able to walk without help at randomisation YNDQ
    'weakface_rand': 'YNDQ',            # Unilateral weakness affecting face at randomisation YNDQ
    'weakarm_rand': 'YNDQ',             # Unilateral weakness affecting arm or hand at randomisation YNDQ
    'weakleg_rand': 'YNDQ',             # Unilateral weakness affecting leg or foot at randomisation YNDQ
    'dysphasia_rand': 'YNDQ',           # Dysphasia at randomisation YNDQ
    'hemianopia_rand': 'YNDQ',          # Homonymous hemianopia at randomisation YNDQ
    'visuospat_rand': 'YNDQ',           # Visuospatial disorder at randomisation YNDQ
    'brainstemsigns_rand': 'YNDQ',      # Brainstem or cerebellar signs at randomisation YNDQ
    'otherdeficit_rand': 'YNDQ',        # Other neurological deficit at randomisation YNDQ
    'stroketype': 'STROKETYPE',         # Stroke subtype STROKETYPE
    'pred_nihss': bool,                 # NIHSS predicted? Y01N
    'konprob': float,                   # Probability of good outcome based on Konig model
}


SEVEN_DAY_FORM = [
    'aspirin_pre',                # Aspirin before admission
    'dipyridamole_pre',           # Dipyridamole before admission
    'clopidogrel_pre',            # Clopidogrel before admission
    'lowdose_heparin_pre',        # Low dose heparin before admission
    'fulldose_heparin_pre',       # Full dose heparin before admission
    'warfarin_pre',               # Warfarin before admission
    'antithromb_pre',             # Other thrombotic agents before admission
    'hypertension_pre',           # Treatment for hypertension before admission
    'anticoag_pre',               # Pre-trial treatment with anti-coagulants
    'diabetes_pre',               # Treatment for diabetes before admission
    'stroke_pre',                 # History of previous stroke or TIA
    'aspirin_day1',               # Aspirin in first 24 hours
    'antiplatelet_day1',          # Other antiplatelets in first 24 hours
    'lowdose_heparin_day1',       # Low dose heparin or low molecular weight heparin in first 24 hours
    'full_anticoag_day1',         # Full anti-coagulation in first 24 hours
    'lowerBP_day1',               # Treatment to lower blood pressure in first 24 hours
    'nontrial_thromb_day1',       # Non-trial thrombolysis in first 24 hours
    'iv_fluids_day1',             # Intravenous fluids in first 24 hours
    'insulin_day1',               # Insulin in first 24 hours
    'aspirin_days2to7',           # Aspirin between 24 hours & 7 days
    'antiplatelet_days2to7',      # Other antiplatelets between 24 hours & 7 days
    'lowdose_heparin_days2to7',   # Low dose heparin or low molecular weight heparin between 24 hours & 7 days
    'full_anticoag_days2to7',     # Full anti-coagulation between 24 hours & 7 days
    'lowerBP_days2to7',           # Treatment to lower blood pressure between 24 hours & 7 days
    'nontrial_thromb_days2to7',   # Non-trial thrombolysis between 24 hours & 7 days
    'nasogastric_days2to7',       # Nasogastric tube or percutaneous gastrostomy between 24 hours & 7 days
    'antibiotics_days2to7',       # Antibiotics between 24 hours & 7 days
    'findiag7',                  # Final diagnosis of initial randomising event (7 day form)
    'brainsite7',                # Location of initial ischaemic stroke (7 day form)
    'haem_type7',                # Type of initial haemorrhagic stroke (7 day form)
    'med_adno',                  # Number of nights in Medical Admissions Unit in first 7 days
    'critcareno',                # Number of nights in Critical Care Unit in first 7 days
    'strk_unitno',               # Number of nights in Stroke Unit in first 7 days
    'genwardno',                 # Number of nights in General Ward in first 7 days
    'myocard_infarct',            # Myocardial infarction in first 7 days
    'extracranial_bleed',         # Major extracranial bleed in first 7 days
    'allergic_reaction',          # Major allergic reaction in first 7 days
    'other_effect',               # Other possible side effect in first 7 days
    'adverse_reaction',           # Other adverse reaction in first 7 days
    'other_effect_code',          # D code for other side effects on 7 day form
    'gcs_eye_7',
    'gcs_verbal_7',
    'gcs_motor_7',
    'liftarms_7',
    'ablewalk_7',
    'indepinadl_7'
    ]

SEVEN_DAY_FORM_FORMATS = {
    'aspirin_pre': 'YNDQ',
    'dipyridamole_pre': 'YNDQ',
    'clopidogrel_pre': 'YNDQ',
    'lowdose_heparin_pre': 'YNDQ',
    'fulldose_heparin_pre': 'YNDQ',
    'warfarin_pre': 'YNDQ',
    'antithromb_pre': 'YNDQ',
    'hypertension_pre': 'YNDQ',
    'anticoag_pre': 'YNDQ',
    'diabetes_pre': 'YNDQ',
    'stroke_pre': 'YNDQ',
    'aspirin_day1': 'YNDQ',
    'antiplatelet_day1': 'YNDQ',
    'lowdose_heparin_day1': 'YNDQ',
    'full_anticoag_day1': 'YNDQ',
    'lowerBP_day1': 'YNDQ',
    'nontrial_thromb_day1': 'YNDQ',
    'iv_fluids_day1': 'YNDQ',
    'insulin_day1': 'YNDQ',
    'aspirin_days2to7': 'YNDQ',
    'antiplatelet_days2to7': 'YNDQ',
    'lowdose_heparin_days2to7': 'YNDQ',
    'full_anticoag_days2to7': 'YNDQ',
    'lowerBP_days2to7': 'YNDQ',
    'nontrial_thromb_days2to7': 'YNDQ',
    'nasogastric_days2to7': 'YNDQ',
    'antibiotics_days2to7': 'YNDQ',
    'findiag7': 'FINDIAG',
    'brainsite7': 'BRAINSITE',
    'haem_type7': 'HAEMTYPE',
    'med_adno': float,
    'critcareno': float,
    'strk_unitno': float,
    'genwardno': float,
    'myocard_infarct': 'YNM',
    'extracranial_bleed': 'YNM',
    'allergic_reaction': 'YNM',
    'other_effect': 'YNM',
    'adverse_reaction': 'YNM',
    'other_effect_code': str,
    'gcs_eye_7': 'GCSEYE',
    'gcs_verbal_7': 'GCSMOTOR',
    'gcs_motor_7': 'GCSVERBAL',
    'liftarms_7': 'YNDQ',
    'ablewalk_7': 'YNDQ',
    'indepinadl_7': 'YNDQ',
}