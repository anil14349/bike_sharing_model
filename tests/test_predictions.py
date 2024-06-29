import sys
import pytest
from pathlib import Path
import pandas as pd

# Ensure the project path is in the sys.path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Import the necessary modules and functions
from bike_sharing_model import __version__ as _version
from bike_sharing_model.config.core import config
from bike_sharing_model.processing.data_manager import load_pipeline
from bike_sharing_model.predict import make_prediction

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
bike_sharing_pipe = load_pipeline(file_name=pipeline_file_name)


def test_make_prediction(sample_input_data):
    # Convert the sample input data to a DataFrame
    #input_df = pd.DataFrame(sample_input_data)
    
    # Ensure consistent data types
    '''input_df['weekday'] = input_df['weekday'].astype(str)
    input_df['workingday'] = input_df['workingday'].astype(str)
    input_df['weathersit'] = input_df['weathersit'].astype(str)
    input_df['temp'] = input_df['temp'].astype('float64')
    input_df['atemp'] = input_df['atemp'].astype('float64')
    input_df['hum'] = input_df['hum'].astype('float64')
    input_df['windspeed'] = input_df['windspeed'].astype('float64')
    input_df['casual'] = input_df['casual'].astype('float64')
    input_df['registered'] = input_df['registered'].astype('float64')
    input_df['cnt'] = input_df['cnt'].astype('float64')'''
    
    # Make a prediction
    result = make_prediction(input_data=sample_input_data[0])
    
    # Check that the result is a dictionary
    assert isinstance(result, dict)
    
    # Check that the result contains 'predictions', 'version', and 'errors'
    assert 'predictions' in result
    assert 'version' in result
    assert 'errors' in result
    
    # Check that the 'version' is correct
    assert result['version'] == _version
    
    # Check that there are no errors
    assert result['errors'] is None or len(result['errors']) == 0
    
    # Check that the predictions are not None
    assert result['predictions'] is not None
    
    # Check that the predictions have the same length as the input data
    assert len(result['predictions']) == len(sample_input_data[0])

if __name__ == "__main__":
    pytest.main([__file__])
