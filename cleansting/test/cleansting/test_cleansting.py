import pytest
from pandas import DataFrame
from cleansting import cleansting
# Math
import numpy as np
import pandas as pd


class TestCleanSting(object):

    @pytest.fixture
    def setup(self):
        # List of paths on webscraper_lamudi.
        d1 = r'C:\Users\albin\PycharmProjects\webscraper_lamudi\lamudi\json_raw\enero\lamudi_22_01_2022.json'
        d2 = r'C:\Users\albin\PycharmProjects\webscraper_lamudi\lamudi\json_raw\febrero\01_02_2022.json'
        d3 = r'C:\Users\albin\PycharmProjects\webscraper_lamudi\lamudi\json_raw\febrero\lamudi_15_02_22.json'
        d4 = r'C:\Users\albin\PycharmProjects\webscraper_lamudi\lamudi\json_raw\marzo\lamudi_28_02_22.json'
        d5 = r'C:\Users\albin\PycharmProjects\webscraper_lamudi\lamudi\json_raw\marzo\lamudi_12_03_22.json'
        d6 = r'C:\Users\albin\PycharmProjects\webscraper_lamudi\lamudi\json_raw\marzo\lamudi_18_03_22.json'
        d7 = r'C:\Users\albin\PycharmProjects\webscraper_lamudi\lamudi\json_raw\marzo\lamudi_26_03_2022.json'
        # Create new object
        cl = cleansting.CleanSting()
        # Add the json files.
        cl.fit_json(d1, d2, d3, d4, d5, d6, d7)

        yield cl

    def test_convert_json_to_pandas(self, setup):
        """Test the type of object that returns the json to pd method."""
        cl = setup
        cl.json_to_pd()
        actual = cl.df['main']
        msg = f'We expected {True}, but We get {actual}'

        assert isinstance(actual, DataFrame), msg

    def test_nan_resumen_length(self, setup):
        """Test the resume print the correct information of the DataFrame"""
        cl = setup
        cl.json_to_pd()
        null_cols_resume = cl.nan_resume(normalize=True)
        actual = len(null_cols_resume.index)
        expected = 0
        msg = f'We expected {expected}, but We get {actual}'

        assert actual > expected, msg

    def test_nan_resumen_count(self, setup):
        """Test the resume print the correct Value Counts information of the DataFrame"""
        cl = setup
        cl.json_to_pd()
        null_cols_resume = cl.nan_resume(normalize=False)
        actual = null_cols_resume.apply(lambda feature: isinstance(feature, int)).all()
        expected = True
        msg = f'We expected {expected}, but We get {actual}'

        assert actual, msg

    def test_nan_resumen_normalize(self, setup):
        """Test the resume print the correct Value Counts Normalized information of the DataFrame"""
        cl = setup
        cl.json_to_pd()
        null_cols_resume = cl.nan_resume(normalize=True)
        actual = null_cols_resume.apply(lambda feature: isinstance(feature, float)).all()
        expected = True
        msg = f'We expected {expected}, but We get {actual}'

        assert actual, msg

    def test_drop_nan(self, setup):
        """Test the Drop Nan value function."""
        cl = setup
        cl.json_to_pd()
        before_null_cols_resume = len(cl.nan_resume(normalize=True).index)
        actual = cl.drop_nan('main', 'habitaciones', 'banos', 'autos', 'colonia').nan_resume().index
        after_null_cols_resume = len(cl.nan_resume(normalize=True).index)
        expected = ['habitaciones', 'banos', 'autos', 'colonia']
        msg = f'We expected not True inside the list, but there still items.'
        msg1 = f'We expected this data {after_null_cols_resume}, was diferente of this {before_null_cols_resume}'
        result = actual.isin(expected).tolist()
        inf_result = (cl.df['main'] == np.inf).sum().sum()
        msg2 = f'We expected {0}, but We get {inf_result}'

        assert before_null_cols_resume != after_null_cols_resume, msg1
        assert not True in result, msg
        assert inf_result == 0, msg2

    def test_get_m2_price(self, setup):
        """Test the correct creation of the Series m2 price"""
        columns = ('m2_terreno', 'm2_const', 'habitaciones', 'banos', 'autos')
        cl = setup
        cl.json_to_pd()
        cl.drop_nan('main', 'habitaciones', 'banos', 'autos', 'colonia')
        cl.filter_by(tb_name='main', tipo_inmueble='Casa', tipo_oferta='Venta', colonia='Puerta de Hierro')
        cl.drop_outliers(tb_name='filtered', columns=columns)
        actual = cl.get_m2_price(tb_name='filtered').mean()
        print(actual)
        expected = 40000.0
        msg = f'We expected {expected}, but We get {actual}'
        assert actual > expected, msg

    def test_get_m2_price_error_msg(self, setup):
        """Test the correct creation of the Series m2 price"""
        cl = setup
        cl.json_to_pd()
        cl.filter_by(tb_name='main', tipo_inmueble='Casa', tipo_oferta='Venta', colonia='Puerta de Hierro')
        with pytest.raises(ValueError):
            cl.get_m2_price(tb_name='filtered')

    def test_get_m2_price_error_msg_departamento(self, setup):
        """Test the correct creation of the Series m2 price"""
        cl = setup
        cl.json_to_pd()
        cl.filter_by(tb_name='main', tipo_inmueble='Departamento', tipo_oferta='Venta', colonia='Puerta de Hierro')
        print(cl.df['filtered'])
        with pytest.raises(ZeroDivisionError):
            cl.get_m2_price(tb_name='filtered')

    def test_get_abs_price(self, setup):
        """Test the correct creation of the Series m2 price"""
        columns = ('m2_terreno', 'm2_const', 'habitaciones', 'banos', 'autos')
        cl = setup
        cl.json_to_pd()
        cl.drop_nan('main', 'habitaciones', 'banos', 'autos', 'colonia')
        cl.filter_by(tb_name='main', tipo_inmueble='Casa', tipo_oferta='Venta', colonia='Puerta de Hierro')
        cl.drop_outliers(tb_name='filtered', columns=columns)
        actual = cl.get_abs_price(tb_name='filtered').mean()
        expected = 40000.0
        msg = f'We expected {expected}, but We get {actual}'
        assert actual < expected, msg

    def test_get_abs_price_error_msg(self, setup):
        """Test the correct creation of the Series m2 price"""
        cl = setup
        cl.json_to_pd()
        cl.filter_by(tb_name='main', tipo_inmueble='Casa', tipo_oferta='Venta', colonia='Puerta de Hierro')
        with pytest.raises(ValueError):
            cl.get_abs_price(tb_name='filtered')

    def test_get_abs_price_error_msg_departamento(self, setup):
        """Test the correct creation of the Series m2 price"""
        cl = setup
        cl.json_to_pd()
        cl.filter_by(tb_name='main', tipo_inmueble='Departamento', tipo_oferta='Venta', colonia='Puerta de Hierro')
        print(cl.df['filtered'])
        with pytest.raises(ZeroDivisionError):
            cl.get_abs_price(tb_name='filtered')

    def test_get_m2_const_price(self, setup):
        """Test the correct creation of the Series m2 price"""
        columns = ('m2_terreno', 'm2_const', 'habitaciones', 'banos', 'autos')
        cl = setup
        cl.json_to_pd()
        cl.drop_nan('main', 'habitaciones', 'banos', 'autos', 'colonia').drop_outliers('main', columns=columns)
        cl.filter_by(tb_name='main', tipo_inmueble='Casa', tipo_oferta='Venta', colonia='Puerta de Hierro')
        actual = cl.get_m2_const_price(tb_name='filtered').mean()
        expected = 40000.0
        msg = f'We expected {expected}, but We get {actual}'
        assert actual > expected, msg

    def test_get_m2_const_price_error_msg(self, setup):
        """Test the correct creation of the Series m2 price"""
        cl = setup
        cl.json_to_pd()
        cl.filter_by(tb_name='main', tipo_inmueble='Casa', tipo_oferta='Venta', colonia='Puerta de Hierro')
        with pytest.raises(ValueError):
            cl.get_m2_const_price(tb_name='filtered')

    def test_change_dtypes(self, setup):
        columns = ('colonia', 'tipo_oferta')
        cl = setup
        cl.json_to_pd()
        cl.filter_by(tb_name='main', tipo_inmueble='Casa', tipo_oferta='Venta', colonia='Puerta de Hierro')
        df = cl.change_dtype('main', columns=columns, astype='category')
        actual = df[['colonia', 'tipo_oferta']].dtypes.all()
        expected = 'category'
        msg = f'We expected {expected}, but We get {actual}'
        assert actual == expected, msg

    def test_change_dtypes_erro_msg(self, setup):
        columns = ('coloniaa', 'tipo_oferta')
        cl = setup
        cl.json_to_pd()
        cl.filter_by(tb_name='main', tipo_inmueble='Casa', tipo_oferta='Venta', colonia='Puerta de Hierro')
        with pytest.raises(KeyError):
            cl.change_dtype('main', columns=columns, astype='category')

    def test_filter_by(self, setup):
        """Test the Creation of the Subset Function."""

        cl = setup
        cl.json_to_pd()
        df = cl.filter_by(tipo_inmueble='Casa', tipo_oferta='Venta', colonia='Puerta de Hierro')
        # Result
        actual_type = df.tipo_inmueble.unique()[0]
        actual_offer = df.tipo_oferta.unique()[0]
        actual_colonia = df.colonia.unique()[0]
        # Expected
        expected_type = 'Casa'
        expected_offer = 'Venta'
        expected_colonia = 'Puerta de Hierro'
        msg1 = f'We expected {expected_type}, but We get {actual_type}'
        msg2 = f'We expected {expected_offer}, but We get {actual_offer}'
        msg3 = f'We expected {expected_colonia}, but We get {actual_colonia}'
        # Assert
        assert actual_type == expected_type, msg1
        assert actual_offer == expected_offer, msg2
        assert actual_colonia == expected_colonia, msg3

    def test_filter_by_error_msg(self, setup):
        """Test the correct ValueError prompt of th autosementation"""

        cl = setup
        cl.json_to_pd()
        with pytest.raises(ValueError):
            cl.filter_by(tipo_inmueble='Casa', tipo_oferta='Venta', colonia='Puerta de Hierr00')

    def test_drop_outliers(self, setup):
        """Test the resume print the correct Value Counts information of the DataFrame"""
        columns = ('m2_terreno', 'm2_const', 'habitaciones', 'banos', 'autos')
        cl = setup
        cl.json_to_pd()
        cl.drop_nan('main', 'habitaciones', 'banos', 'autos', 'colonia')
        cl.filter_by('main', tipo_inmueble='Casa', tipo_oferta='Venta', colonia='Puerta de Hierro')
        expected = len(cl.df['filtered'])
        actual = len(cl.drop_outliers('filtered', columns=columns))
        msg = f'We expected {expected}, but We get {actual}'
        assert actual < expected, msg

    def test_drop_outliers_result(self, setup):
        """Test the result values skew are less than one"""
        columns = ('m2_terreno', 'm2_const', 'habitaciones', 'banos', 'autos')
        cl = setup
        cl.json_to_pd()
        cl.drop_nan('main', 'habitaciones', 'banos', 'autos', 'colonia')
        cl.filter_by('main', tipo_inmueble='Casa', tipo_oferta='Venta', colonia='Puerta de Hierro')
        expected = 1
        actual = cl.drop_outliers('filtered', columns=columns) \
            [['m2_terreno', 'm2_const', 'habitaciones', 'banos', 'autos']].skew().abs().gt(expected).all()
        msg = f'We expected {expected}, but We get {actual}'
        assert not actual, msg

    def test_set_sector_inmo_venta(self, setup):
        cl = setup
        cl.json_to_pd()
        cl.drop_nan('main', 'habitaciones', 'banos', 'autos', 'colonia')
        cl.change_dtype('main', columns=('tipo_oferta', 'tipo_inmueble'), astype='category')
        df = cl.set_sector_inmo().df['main']
        # type
        venta = (df.tipo_oferta == 'Venta')
        # Actual
        actual_interes_social = df.loc[(df.price.lt(1000001) & venta), 'sector_inmo'].unique()[0]
        actual_interes_medio = df.loc[(df.price.between(1000001, 3000000) & venta), 'sector_inmo'].unique()[0]
        actual_residencial = df.loc[(df.price.between(3000001, 7000000) & venta), 'sector_inmo'].unique()[0]
        actual_residencial_plus = df.loc[(df.price.between(7000001, 15000000) & venta), 'sector_inmo'].unique()[0]
        actual_premium = df.loc[(df.price.gt(15000000) & venta), 'sector_inmo'].unique()[0]
        actual = [
            actual_interes_social, actual_interes_medio, actual_residencial,
            actual_residencial_plus, actual_premium
        ]

        expected = ['Interés Social', 'Interés Medio', 'Residencial', 'Residencial Plus', 'Premium']
        msg = f'We expected {expected}, but We get {actual}'

        assert actual == expected, msg

    def test_set_sector_inmo_renta(self, setup):
        cl = setup
        cl.json_to_pd()
        cl.drop_nan('main', 'habitaciones', 'banos', 'autos', 'colonia')
        cl.change_dtype('main', columns=('tipo_oferta', 'tipo_inmueble'), astype='category')
        df = cl.set_sector_inmo().df['main']
        # type
        venta = (df.tipo_oferta == 'Renta')
        # Actual
        actual_interes_social = df.loc[(df.price.lt(5001) & venta), 'sector_inmo'].unique()[0]
        actual_interes_medio = df.loc[(df.price.between(5001, 10000) & venta), 'sector_inmo'].unique()[0]
        actual_residencial = df.loc[(df.price.between(10001, 15000) & venta), 'sector_inmo'].unique()[0]
        actual_residencial_plus = df.loc[(df.price.between(15001, 30000) & venta), 'sector_inmo'].unique()[0]
        actual_premium = df.loc[(df.price.gt(30000) & venta), 'sector_inmo'].unique()[0]
        actual = [
            actual_interes_social, actual_interes_medio, actual_residencial,
            actual_residencial_plus, actual_premium
        ]

        expected = ['Interés Social', 'Interés Medio', 'Residencial', 'Residencial Plus', 'Premium']
        msg = f'We expected {expected}, but We get {actual}'

        assert actual == expected, msg

    def test_set_amenidades(self, setup):
        cl = setup
        cl.json_to_pd()
        cl.drop_nan('main', 'habitaciones', 'banos', 'autos', 'colonia')
        cl.change_dtype('main', columns=('tipo_oferta', 'tipo_inmueble'), astype='category')
        _, list_amenidades = cl.set_amenidades('main', 'amenidades')
        df = cl.df['main']
        columns = df.columns
        actual = len(columns[columns.isin(list_amenidades)])
        expected = len(list_amenidades)

        msg = f'We expected {expected}, but We get {actual}'
        assert actual == expected, msg

    def test_get_numeric_columns(self, setup):
        """Test the correct creation of list of columns that contain only the numeric features. """
        cl = setup
        cl.json_to_pd()
        cl.drop_nan('main', 'habitaciones', 'banos', 'autos', 'colonia')
        cl.change_dtype('main', columns=('m2_terreno', 'm2_const', 'total_habitaciones'), astype='float16')
        cl.change_dtype('main', columns=('habitaciones', 'banos', 'autos'), astype='int8')
        numeric_columns = cl.get_numeric_columns()
        df = cl.df['main'].loc[:, numeric_columns]
        actual = df.dtypes.apply(lambda x: x == np.float16 or x == np.float64 or x == np.int8).all()
        msg = f'We expected {True}, but We get {actual}'
        assert actual, msg

    def test_get_categories_columns(self, setup):
        """Test the correct creation of list of columns that contain only the numeric features. """
        cl = setup
        cl.json_to_pd()
        cl.drop_nan('main', 'habitaciones', 'banos', 'autos', 'colonia')
        cl.change_dtype('main',
                        columns=(
                            'colonia', 'tipo_oferta', 'tipo_inmueble', 'sub_category', 'ciudad', 'estado', 'agent'),
                        astype='category')
        categories_columns = cl.get_categories_columns()
        df = cl.df['main'].loc[:, categories_columns]
        actual = df.dtypes.apply(lambda x: x == 'category').all()
        msg = f'We expected {True}, but We get {actual}'
        assert actual, msg
