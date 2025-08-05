import pandas as pd

def encode_yn(df, col):
    return df[col].map({1: 1, 2: 0})

    """YNDQ
    1='Yes'
    2='No'
    3="Don't Know"
    4= 'Cannot assess'
    10='Died so question not relevant'
    20='Question not answered'
    30="Form not returned"
    0="Question not asked";
    """

def encode_yndq(df, col):
    return df[col].map({
        1: 'Yes',
        2: 'No',
        3: 'Unknown',
        4: 'Unknown',
        10: 'Deceased',
        20: 'Unknown',
        30: 'Unknown',
        0: 'Unknown'
    })
    

def encode_ynm(df, col):
    df[col] = df[col].map({
        1: 1.0,
        2: 0.0,
        3: pd.NA
    })
    return df[col].astype('float')


def encode_GCSEYE(df, col):
    return df[col].map({
        0: pd.NA,         # Missing
        1: 'Never',       # Never
        2: 'To Pain',     # To Pain
        3: 'To command',  # To command
        4: 'Spontaneously', # Spontaneously
        10: 'Deceased',     # Died so question not relevant
        20: 'Unknown',      # Question not answered
        30: 'Unknown',      # Form not returned
        40: 'Unknown'       # Question not asked
    })


def encode_GCSMOTOR(df, col):
    return df[col].map({
        0: pd.NA,                        # Missing
        1: 'None',                       # None
        2: 'Extend to pain',             # Extend to pain
        3: 'Abnormal flex to pain',      # Abnormal flex to pain
        4: 'Normal flex to pain',        # Normal flex to pain
        5: 'Localises movements to pain',# Localises movements to pain
        6: 'Normal',                     # Normal
        10: 'Deceased',                  # Died so question not relevant
        20: 'Unknown',                   # Question not answered
        30: 'Unknown',                   # Form not returned
        40: 'Unknown'                    # Question not asked
    })
    
    
def encode_GCSVERBAL(df, col):
    return df[col].map({
        0: pd.NA,                                   # Missing
        1: 'None',                                  # None
        2: 'Noises only',                           # Noises only
        3: 'Inappropriate words',                   # Inappropriate words
        4: 'Confused in time, place or person',     # Confused in time, place or person
        5: 'Orientated in time, place and person',  # Orientated in time, place and person
        10: 'Deceased',                             # Died so question not relevant
        20: 'Unknown',                              # Question not answered
        30: 'Unknown',                              # Form not returned
        40: 'Unknown'                               # Question not asked
    })
    

def encode_gender(df, col):
    return df[col].map({
        1: 'Male',
        2: 'Female'
    })


def encode_findiag(df, col):
    return df[col].map({
        1: 'Definite ischaemic stroke',
        2: 'Definite or probable haemorrhagic stroke',
        3: 'Non-stroke cause'
    })
    
    
def encode_brainsite(df, col):
    return df[col].map({
        1: 'Unknown',
        2: 'Cerebral hemisphere',
        3: 'Posterior circulation'
    })
    
def encode_haemtype(df, col):
    return df[col].map({
        1: 'Primary intracranial haemorrhage',
        2: 'Subdural haemorrhage',
        3: 'Subarachnoid haemorrhage'
    })
    
