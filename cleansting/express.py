import joblib
import pandas as pd
import numpy as np
from numpy import array
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder


def express_work_setup(data: DataFrame) -> DataFrame:
    """Generate all the minimal transformation to work with the Data

    Parameters
    ----------
    data : DataFrame:

    Returns
    -------
    DataFrame
    """
    df = data
    df = (df
          .dropna(subset=['colonia', 'm2_const', 'habitaciones', 'banos', 'autos', 'price', 'link_img', 'agent'])
          .pipe(lambda df_: df_[~((df_.tipo_inmueble == 'Casa') & (df_.m2_terreno.isna()))])
          .pipe(lambda df_: df_[(df_.m2_const > 20) & (df_.m2_const < 5000)])  # Remove values less than 10
          .pipe(lambda df_: df_[df_.price > 100])  # Remove listings below 100
          .pipe(lambda df_: df_[~((df_.tipo_oferta == 'Venta') & (df_.price < 100000))])
          .pipe(lambda df_: df_[df_.price < 100000000])
          .pipe(lambda df_: df_[~((df_.tipo_inmueble == 'Casa') & (df_.m2_terreno <= 50))])  # rm houses with less 50m2
          .pipe(lambda df_: df_[~((df_.tipo_inmueble == 'Casa') & (df_.m2_terreno >= 5000))])
          .pipe(lambda df_: df_[~((df_.habitaciones == 0) | (df_.banos == 0))])
          .pipe(lambda df_: df_[df_.banos < 10])
          .pipe(lambda df_: df_[~((df_.tipo_inmueble == 'Departamento') & (df_.habitaciones > 7))])
          .pipe(lambda df_: df_[df_.habitaciones < 10])
          .pipe(lambda df_: df_[df_.autos < 30])
          .pipe(lambda df_: df_[~((df_.autos == 0) & (df_.colonia.str.contains('Vallarta') == False))])
          # Renta
          .pipe(lambda df_: df_[~((df_.tipo_oferta == 'Renta') & (df_.price > 150000))])
          .assign(
        m2_price=lambda df_: df_.price / df_.m2_terreno,  # price / m2_terreno
        m2_const_price=lambda df_: df_.price / df_.m2_const,  # price / m2_const
        m2_abs_price=lambda df_: df_.price / (df_.m2_terreno + df_.m2_const),  # price / m2_terreno + m2_const
        fecha_pub=lambda df_: pd.to_datetime(df_.fecha_pub),
        descripcion=lambda df_: df_.descripcion.str.replace('"text":', '', regex=True))
          .pipe(lambda df_: df_[~(df_.m2_const_price > 100000)])
          .pipe(lambda df_: df_[~((df_.tipo_inmueble == 'Casa') & (df_.m2_price > 90000))])
          .pipe(create_sector_inmo)
          .astype(
        {'tipo_inmueble': 'category', 'sub_category': 'category', 'tipo_oferta': 'category', 'colonia': 'category',
         'estado': 'category', 'ciudad': 'category',
         'price': 'float32', 'm2_price': 'float32', 'm2_const_price': 'float32', 'm2_abs_price': 'float32',
         'm2_terreno': 'float32', 'm2_const': 'float32',
         'habitaciones': 'int16', 'banos': 'int16', 'autos': 'int16', 'antiguedad': 'float32'})
          .drop(columns=['divisa', 'total_habitaciones', 'link', 'link_img', 'agent'])
          ).reset_index().drop(columns='index')
    return df


def rm_outliers_by_colonia(data: DataFrame, columns: list = None) -> DataFrame:
    """ Remove the Outliers by Colonia in the DataFrame Selected.

    ## Use this list of colonies to Test: ['Puerta de Hierro', 'Valle Real', 'Americana', 'Providencia 1a Secc']

    Parameters
    ----------
    data : DataFrame:
        DataFrame that will be clean for its outliers.
    columns : list:
        List of the name of the features/columns that would remove the outliers.
         (Default value = None)

    Returns
    -------
    DataFrame
    """
    if columns is None:
        columns = ['price', 'm2_const', 'm2_terreno', 'habitaciones', 'banos', 'autos']

    df = data

    # Get Data Skew information Table
    skew_features_tb = identify_skew_features(df, columns)
    # Get Index Outliers
    skew_colonias_name = get_colonies_outliers_name(skew_features_tb)
    # Remove outliers by Colonia
    clean_data = rm_skew_outliers(df, skew_colonias_name, columns)

    return clean_data


def identify_skew_features(data: DataFrame, columns: list) -> DataFrame:
    """Create a table with the Skew Resume to identify the Observation that have one or more Outliers.

    Parameters
    ----------
    data :  DataFrame:
        
    columns : list:
        List of the name of the features/columns that would remove the outliers.
        
    groupby :
         (Default value = 'colonia')

    Returns
    -------

    """
    return (data
            .groupby('colonia', observed=True)[columns]
            .skew())


def get_colonies_outliers_name(skew_features_tb: DataFrame) -> array:
    """Create an Array of the Name of the Colonies

    Parameters
    ----------
    skew_features_tb : DataFrame:
        Array of the features name it would be removed the outliers.

    Returns
    -------
    array
    """
    positive_skew_features = skew_features_tb > 1
    negative_skew_features = skew_features_tb < -1
    # Select Positive or Negative Skew data
    skew_features = (
        skew_features_tb[(positive_skew_features) | (negative_skew_features)]
            .notna()  # Identify Only the Skew Observation
        # .any(axis=1)
    ).pipe(lambda df_: df_[df_.any(axis=1)]).index  # Select if any of the row have Outliers

    return skew_features


def rm_skew_outliers(data: DataFrame, skew_colonias_name: array, columns: list) -> DataFrame:
    """Remove the Outliers

    Parameters
    ----------
    data : DataFrame:
        DataFrame that will be clean for its outliers.
        
    skew_colonias_name : list:
        Array with the name of the Colonies with outliers.
    columns : list:
        List of the name of the features/columns that would remove the outliers.

    Returns
    -------
    DataFrame
    """
    df = data
    # iterate colonia names list
    for colonia in skew_colonias_name:
        # select the colonia subset DataFrame
        colonia_subset = df[df.colonia == colonia]
        # Iterate over the Colums with outliers
        for col in columns:
            # Second outliers Check
            # Positive +
            if colonia_subset[col].skew() > 1:
                # select low and hig quantiles value of the feautre
                low = colonia_subset[col].quantile(0.01)
                hig = colonia_subset[col].quantile(0.95)
                # Select only the outliers
                listings_outliers = colonia_subset.loc[(colonia_subset[col] <= low) | (colonia_subset[col] >= hig)]
                # Subset only not outliers
                df = df[~df.index.isin(listings_outliers.index)]

            # Negative -
            if colonia_subset[col].skew() < -1:
                # select low and hig quantiles value of the feautre
                low = colonia_subset[col].quantile(0.05)
                hig = colonia_subset[col].quantile(0.99)
                # Select only the outliers
                listings_outliers = colonia_subset.loc[(colonia_subset[col] <= low) | (colonia_subset[col] >= hig)]
                # Subset only not outliers
                df = df[~df.index.isin(listings_outliers.index)]

    return df.drop(columns=['sku'])


def set_dummies_features(data):
    df = data

    # Creating Dummi Variables
    offer_dummies = pd.get_dummies(df.tipo_oferta, prefix='dummie', drop_first=True)
    type_dummies = (pd.get_dummies(df.tipo_inmueble, prefix='dummie')
                    .drop(columns='dummie_Departamento'))
    # Concatenate the Tables
    df_result = (pd.concat([df, offer_dummies, type_dummies], axis=1)
                 .rename(columns={'dummie_Venta': 'venta', 'dummie_Casa': 'casa'}))

    return df_result


def colonies_encoder(data, save_file: str = None):
    df = data
    # Create Encoder
    encoder = LabelEncoder().fit(df['colonia'])
    df_result = df.assign(endocer_colonia=encoder.transform(df['colonia']))

    if save_file is not None:
        joblib.dump(encoder, save_file)

    return df_result


def create_sector_inmo(data: DataFrame) -> DataFrame:
    """Create a new Column call it 'sector_inmo' that contains the categories of the listings
            economical sectors.
    
    The categories are: 'Interés Social', 'Interés Medio', 'Residencial', 'Residencial Plus', 'Premium'.
    Depends on the type of offer (sell or rent) and the price that the label would be selected.

    Parameters
    ----------
    data : DataFrame:

    Returns
    -------

    
    """
    df = data
    # Create Sector Inmobiliario Column
    categories = ['Interés Social', 'Interés Medio', 'Residencial', 'Residencial Plus', 'Premium']
    df['sector_inmo'] = pd.Series(categories).astype('category')
    # Select Offer Types
    venta = (df.tipo_oferta == 'Venta')
    renta = (df.tipo_oferta == 'Renta')
    # Use a dict to iterate over the strings and subsets.
    offers = {'venta': venta, 'renta': renta}

    for offer_txt, offer_df in offers.items():

        if offer_txt == 'venta':
            interes_social = df.price.lt(1000001)
            interes_medio = df.price.between(1000001, 3000000)
            residencial = df.price.between(3000001, 7000000)
            residencial_plus = df.price.between(7000001, 15000000)
            premium = df.price.gt(15000000)
            sectors = [interes_social, interes_medio, residencial, residencial_plus, premium]

        else:  # renta
            interes_social = df.price.lt(5001)
            interes_medio = df.price.between(5001, 10000)
            residencial = df.price.between(10001, 15000)
            residencial_plus = df.price.between(15001, 30000)
            premium = df.price.gt(30000)
            sectors = [interes_social, interes_medio, residencial, residencial_plus, premium]

        for sector, category in zip(sectors, categories):
            df.loc[sector & offer_df, 'sector_inmo'] = category

    return df


def set_amenidades(data: DataFrame, column: str = 'amenidades') -> DataFrame:
    """Create N number of new Columns of the total amenities in the column 'amenidades' lists.
    
    This column contains list of amenities, and it is variable. The program don't know how many new columns
    will be created. So this proces is O(n2).

    Parameters
    ----------
    data : DataFrame:
        
    column : str : (Default value = 'amenidades')
        The column Name that contain the Raw List of amenities.

    Returns
    -------

    
    """
    df = data.reset_index()

    list_amenidades = list()
    # Unpacking the row amenities
    for i, row_list in enumerate(df[column]):
        if row_list is not None:  # Validate is not empty
            # Create/Add columns amenity value.
            for amenidad in row_list:
                df.loc[i, amenidad] = True
                # Add empty list
                if amenidad not in list_amenidades:
                    list_amenidades.append(amenidad)
    # Add Nan to the empty values to prevente errors with SQL DataBase
    for amenidad in list_amenidades:
        df[amenidad] = df[amenidad].replace(np.nan, False)

    return df
