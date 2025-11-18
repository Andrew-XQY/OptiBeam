import requests
import sseclient
from typing import List
import numpy as np


server = 'cwe-513-vpl746:8765'

def ensure_list(params):
    if isinstance(params, list):
        return params
    else:
        return [params]
    
def getJapcParameter(server:str, parameters:str|List[str]) -> dict:
    url = f'http://{server}/get'
    response = requests.post(url, json={'parameters': ensure_list(parameters)})
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching parameter {parameters}: {response.status_code} - {response.text}")

def setJapcParameters(server:str, parameter_values:dict) -> None:
    url = f'http://{server}/set'
    payload = {
        "parameters": parameter_values,
        "checkDims": False
    }
    try:    
        response = requests.post(url, json=payload)
        response.raise_for_status()        
    except Exception as e:
        print(f"Exception occurred while setting parameters: {e}")
    
    return

def getMagnetCurrent(magnet_name:str, timing_user:str='SCT.USER.SETUP')-> float:
    par_name = f"{magnet_name}/Acquisition"
    return getJapcParameter(server, f"{par_name}@{timing_user}")[f"{timing_user}@{par_name}"]['value']['currentAverage']


def getMagnetsCurrent(magnets_name:List[str], timing_user:str='SCT.USER.SETUP')-> dict:
    par_names = {magnet_name: f"{magnet_name}/Acquisition" for magnet_name in magnets_name}
    data = getJapcParameter(server, [f"{par_name}@{timing_user}" for par_name in par_names.values()])            
    return {magnet_name:data[f"{timing_user}@{par_names[magnet_name]}"]['value']['currentAverage'] for magnet_name in magnets_name}

def setMagnetCurrent(magnet_name:str, current:float, timing_user:str='')-> None:    
    url = f'http://{server}/set'
    par_name = f"{magnet_name}/SettingPPM"        
    parameter_values= {f"{par_name}@{timing_user}": {'current':{'value': current, 'type':'float32'}}}
    setJapcParameters(server, parameter_values)
    
def setMagnetsCurrent(magnets_current:dict, timing_user:str='')-> None:    
    url = f'http://{server}/set'        
    parameter_values= {f"{magnet_name}/SettingPPM@{timing_user}": {'current':{'value': current, 'type':'float32'}} 
                       for magnet_name, current in magnets_current.items()}
    setJapcParameters(server, parameter_values)
    return

def getBooleanValueAtBit(integer_number, bit_position) -> bool:
    mask = 1 << bit_position
    return (integer_number & mask) != 0

def is_busy(state: int) -> bool:
    return getBooleanValueAtBit(state, 4)

def getMagnetsStatus(magnets_name:List[str], timing_user:str='SCT.USER.SETUP')-> dict:
    par_names = {magnet_name: f"{magnet_name}/Acquisition" for magnet_name in magnets_name}
    data = getJapcParameter(server, [f"{par_name}@{timing_user}" for par_name in par_names.values()])            
    return {magnet_name:is_busy(data[f"{timing_user}@{par_names[magnet_name]}"]['value']['current_status']) for magnet_name in magnets_name}
    
    
print('before', getMagnetCurrent('CA.QFD0880'))
setMagnetCurrent('CA.QFD0880', 0)
print('after', getMagnetCurrent('CA.QFD0880'))

    



