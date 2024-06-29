import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

# Import the classes from your module
from bike_sharing_model.processing.features import WeekdayImputer, WeathersitImputer, OutlierHandler, Mapper, WeekdayOneHotEncoder

# Test WeekdayImputer
def test_weekday_imputer(sample_input_data):

    
    imputer = WeekdayImputer(variables='weekday')
    result = imputer.fit(sample_input_data[0]).transform(sample_input_data[0])
    
    assert 'dteday' not in result.columns
    #assert result['weekday'].tolist() == ['Mon', 'Tue', 'Wed']
    assert 'Sun' in result['weekday'].tolist()
    assert 'Tue' in result['weekday'].tolist()

# Test WeathersitImputer
def test_weathersit_imputer(sample_input_data):
    df = pd.DataFrame({
        'weathersit': ['Cloudy', np.nan, 'Rainy']
    })
    imputer = WeathersitImputer(variables='weathersit')
    result = imputer.fit_transform(sample_input_data[0])
    
    assert 'Clear' in result['weathersit'].tolist()
    assert 'Mist' in result['weathersit'].tolist()
    assert 'Light Rain' in result['weathersit'].tolist()


# Test OutlierHandler
def test_outlier_handler(sample_input_data):

    handler = OutlierHandler(columns=['windspeed'], method='iqr', factor=1.5)
    result = handler.fit_transform(sample_input_data[0])
    
    assert result['windspeed'].max() < 100
    assert result['windspeed'].min() > -50
    

# Test Mapper
def test_mapper(sample_input_data):

    mappings = {'spring': 1, 'summer': 2, 'fall': 3, 'winter': 4}
    mapper = Mapper(variables='season', mappings=mappings)
    result = mapper.fit_transform(sample_input_data[0])
    
    assert 2 in result['season'].tolist()

# Test WeekdayOneHotEncoder
def test_weekday_one_hot_encoder(sample_input_data):

    encoder = WeekdayOneHotEncoder(column_name='weekday')
    result = encoder.fit_transform(sample_input_data[0])
    
    assert 'weekday' not in result.columns
    assert 'weekday_Mon' in set(result.columns)


# Test pipeline with all transformers
def test_pipeline(sample_input_data):
    # Use a larger dataset with a clear outlier


    pipeline = Pipeline([
        ('weekday_imputer', WeekdayImputer(variables='weekday')),
        ('weathersit_imputer', WeathersitImputer(variables='weathersit')),
        ('outlier_handler', OutlierHandler(columns=['temp'])),
        ('season_mapper', Mapper(variables='season', mappings={'spring': 1, 'summer': 2, 'fall': 3, 'winter': 4})),
        ('weekday_encoder', WeekdayOneHotEncoder(column_name='weekday'))
    ])

    result = pipeline.fit_transform(sample_input_data[0])

    assert 'dteday' not in result.columns
    assert 'weekday' not in result.columns
    assert set(result.columns) >= {'weathersit', 'temp', 'season', 'weekday_Mon', 'weekday_Tue', 'weekday_Wed', 'weekday_Thu', 'weekday_Fri'}
    assert set(result['weathersit'].tolist()) == {'Mist', 'Clear', 'Light Rain'}
    
    # Check if the outlier (100) has been handled
    assert result['temp'].max() < 100
    assert result['temp'].max() > 25  # The upper bound should be higher than the highest non-outlier value
    
    assert set(result['season'].tolist()) == {1, 2, 3, 4}

    # Print the temperature values for debugging
    print(f"Final temp column: {result['temp'].tolist()}")
