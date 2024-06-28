import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

# Import the classes from your module
from bike_sharing_model.processing.features import WeekdayImputer, WeathersitImputer, OutlierHandler, Mapper, WeekdayOneHotEncoder

# Test WeekdayImputer
def test_weekday_imputer():
    df = pd.DataFrame({
        'dteday': pd.to_datetime(['2023-06-26', '2023-06-27', '2023-06-28']),
        'weekday': ['Mon', np.nan, 'Wed']
    })
    imputer = WeekdayImputer(variables='weekday')
    result = imputer.fit_transform(df)
    
    assert 'dteday' not in result.columns
    assert result['weekday'].tolist() == ['Mon', 'Tue', 'Wed']

# Test WeathersitImputer
def test_weathersit_imputer():
    df = pd.DataFrame({
        'weathersit': ['Cloudy', np.nan, 'Rainy']
    })
    imputer = WeathersitImputer(variables='weathersit')
    result = imputer.fit_transform(df)
    
    assert result['weathersit'].tolist() == ['Cloudy', 'Clear', 'Rainy']

# Test OutlierHandler
def test_outlier_handler():
    df = pd.DataFrame({
        'A': [1, 2, 3, 100, -50],
        'B': [10, 20, 30, 40, 50]
    })
    handler = OutlierHandler(columns=['A'], method='iqr', factor=1.5)
    result = handler.fit_transform(df)
    
    assert result['A'].max() < 100
    assert result['A'].min() > -50
    assert result['B'].tolist() == [10, 20, 30, 40, 50]

# Test Mapper
def test_mapper():
    df = pd.DataFrame({
        'season': ['spring', 'summer', 'fall', 'winter']
    })
    mappings = {'spring': 1, 'summer': 2, 'fall': 3, 'winter': 4}
    mapper = Mapper(variables='season', mappings=mappings)
    result = mapper.fit_transform(df)
    
    assert result['season'].tolist() == [1, 2, 3, 4]

# Test WeekdayOneHotEncoder
def test_weekday_one_hot_encoder():
    df = pd.DataFrame({
        'weekday': ['Mon', 'Tue', 'Wed', 'Mon']
    })
    encoder = WeekdayOneHotEncoder(column_name='weekday')
    result = encoder.fit_transform(df)
    
    assert 'weekday' not in result.columns
    assert set(result.columns) == {'weekday_Mon', 'weekday_Tue', 'weekday_Wed'}
    assert result['weekday_Mon'].tolist() == [1, 0, 0, 1]
    assert result['weekday_Tue'].tolist() == [0, 1, 0, 0]
    assert result['weekday_Wed'].tolist() == [0, 0, 1, 0]

# Test pipeline with all transformers
def test_pipeline():
    # Use a larger dataset with a clear outlier
    df = pd.DataFrame({
        'dteday': pd.to_datetime(['2023-06-26', '2023-06-27', '2023-06-28', '2023-06-29', '2023-06-30']),
        'weekday': ['Mon', np.nan, 'Wed', 'Thu', 'Fri'],
        'weathersit': ['Cloudy', np.nan, 'Rainy', 'Sunny', 'Cloudy'],
        'temp': [10, 15, 20, 25, 100],  # 100 is a clear outlier
        'season': ['spring', 'spring', 'summer', 'summer', 'summer']
    })

    pipeline = Pipeline([
        ('weekday_imputer', WeekdayImputer(variables='weekday')),
        ('weathersit_imputer', WeathersitImputer(variables='weathersit')),
        ('outlier_handler', OutlierHandler(columns=['temp'])),
        ('season_mapper', Mapper(variables='season', mappings={'spring': 1, 'summer': 2, 'fall': 3, 'winter': 4})),
        ('weekday_encoder', WeekdayOneHotEncoder(column_name='weekday'))
    ])

    result = pipeline.fit_transform(df)

    assert 'dteday' not in result.columns
    assert 'weekday' not in result.columns
    assert set(result.columns) >= {'weathersit', 'temp', 'season', 'weekday_Mon', 'weekday_Tue', 'weekday_Wed', 'weekday_Thu', 'weekday_Fri'}
    assert result['weathersit'].tolist() == ['Cloudy', 'Clear', 'Rainy', 'Sunny', 'Cloudy']
    
    # Check if the outlier (100) has been handled
    assert result['temp'].max() < 100
    assert result['temp'].max() > 25  # The upper bound should be higher than the highest non-outlier value
    
    assert result['season'].tolist() == [1, 1, 2, 2, 2]

    # Print the temperature values for debugging
    print(f"Final temp column: {result['temp'].tolist()}")

# Test error cases
def test_error_cases():
    with pytest.raises(ValueError):
        WeekdayImputer(variables=['weekday'])  # should be a string, not a list
    
    with pytest.raises(ValueError):
        Mapper(variables=['season'], mappings={})  # variables should be a string
    
    with pytest.raises(ValueError):
        OutlierHandler(method='invalid_method')  # invalid method