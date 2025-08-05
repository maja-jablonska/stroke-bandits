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
        1: 1,
        2: 0,
        3: pd.NA,
        4: pd.NA,
        10: pd.NA,
        20: pd.NA,
        30: pd.NA,
        0: pd.NA,
    })
    
def encode_GCSEYE(df, col):
    return df[col].map({
        0: pd.NA,         # Missing
        1: 'Never',       # Never
        2: 'To Pain',     # To Pain
        3: 'To command',  # To command
        4: 'Spontaneously', # Spontaneously
        10: pd.NA,        # Died so question not relevant
        20: pd.NA,        # Question not answered
        30: pd.NA,        # Form not returned
        40: pd.NA         # Question not asked
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
        10: pd.NA,                       # Died so question not relevant
        20: pd.NA,                       # Question not answered
        30: pd.NA,                       # Form not returned
        40: pd.NA                        # Question not asked
    })
    
    
def encode_GCSVERBAL(df, col):
    return df[col].map({
        0: pd.NA,                                   # Missing
        1: 'None',                                  # None
        2: 'Noises only',                           # Noises only
        3: 'Inappropriate words',                   # Inappropriate words
        4: 'Confused in time, place or person',     # Confused in time, place or person
        5: 'Orientated in time, place and person',  # Orientated in time, place and person
        10: pd.NA,                                  # Died so question not relevant
        20: pd.NA,                                  # Question not answered
        30: pd.NA,                                  # Form not returned
        40: pd.NA                                   # Question not asked
    })
    

def encode_gender(df, col):
    return df[col].map({
        1: 'Male',
        2: 'Female'
    })
