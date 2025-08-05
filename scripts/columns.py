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