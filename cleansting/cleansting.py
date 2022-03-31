# Type imports
from typing import Union

import sqlalchemy
from pandas import DataFrame
from pandas import Series
from dataclasses import dataclass, field
# Libraries
import pandas as pd
import numpy as np


@dataclass
class CleanSting:
    """This class clean and manipulate the data of the Listings in Perfect Deals app"""

    json_files: tuple[str] = field(init=False, default_factory=list, repr=False)
    db: sqlalchemy = field(init=False, repr=False)
    df: dict[str:DataFrame] = field(init=False, repr=True, default_factory=dict)

    def fit_json(self, *args: str) -> None:
        """Fit the file path(s) of the json files.

        Parameters
        ----------
        *args: str :
            Json location file path. This could be variable.

        Returns
        -------
        None
        """
        self.json_files = args

    def json_to_pd(self) -> None:
        """Create a new DataFrame call it 'main'.

        This is the result of the concatenation of a list of Json Paths. The result will drop all the
        duplicate SKUs of the DataFrame and stored inside the self.df dict.

        Returns
        -------
        None
        """
        lst_df = [pd.read_json(file) for file in self.json_files]
        self.df['main'] = pd.concat(lst_df).drop_duplicates(subset=['sku'])

    def nan_resume(self, tb_name: str = 'main', normalize: bool = True) -> Series:
        """Create a Resumen Serie of the NaN Values in all the columns of the DataFrame.

        Parameters
        ----------
        tb_name : str: (Default value = 'main')
          This is Key Value Name in the self.df dictionary that contains the DataFrame you want to apply
          the method.

        normalize : bool:  (Default value = True)
            If this Value is True it will return a Resume Table with percentage. If is False it will
            return the count of the NaN values.

        Returns
        -------
        Series
        """
        df = self.df[tb_name]
        if not normalize:
            null_columns = df.isnull().sum()  # Count Values
        else:
            null_columns = df.isnull().mean().round(2)  # Percentage Values

        null_cols_resume = null_columns.loc[null_columns.gt(0)].sort_values(ascending=False)

        return null_cols_resume

    def drop_nan(self, tb_name: str = 'main', *args: str):
        """Remove the NaN and Inf Values of the Columns Selected of the DataFrame.

        Parameters
        ----------
        tb_name : str:  (Default value = 'main')
            This is Key Value Name in the self.df dictionary that contains the DataFrame you want to apply
            the method.

        *args: str :
            The name of the columns you want to drop the NaN and Inf Values.

        Returns
        -------
        self
        """
        df = self.df[tb_name].dropna(subset=args)
        self.df[tb_name] = df[df != np.inf]
        return self

    def get_m2_price(self, tb_name: str = 'main') -> Series:
        """Create a Serie with the value of the m2 price

        Parameters
        ----------
        tb_name : str:  (Default value = 'main')
            This is Key Value Name in the self.df dictionary that contains the DataFrame you want to apply
            the method.

        Returns
        -------
        Series
        """

        df = self.df[tb_name]

        listing_type = df.tipo_inmueble.unique()[0]
        if listing_type == 'Casa':
            result = df.price / df.m2_terreno
        else:
            raise ZeroDivisionError(
                f"The Method '{self.get_m2_price.__name__}' only works with 'Houses'. Please filter by 'Casas'")
        # Validate if is not inf Results.
        if result.mean() != np.inf:
            return result
        else:
            raise ValueError(
                f'Series mean is inf. Please clean inf values before use {self.get_m2_price.__name__}.')

    def get_m2_const_price(self, tb_name: str = 'main') -> Series:
        """Create a Serie with the value of the m2 construction price

        Parameters
        ----------
        tb_name : str:  (Default value = 'main')
            This is Key Value Name in the self.df dictionary that contains the DataFrame you want to apply
            the method.

        Returns
        -------
        Series
        """

        df = self.df[tb_name]
        result = df.price / df.m2_const
        # Validate if is not inf Results.
        if result.mean() != np.inf:
            return result
        else:
            raise ValueError(
                f'Series mean is inf. Please clean inf values before use {self.get_m2_const_price.__name__}.')

    def get_abs_price(self, tb_name: str = 'main') -> Series:
        """Create a Serie with the value of the Absolute price (m2 + m2 const).

        This Serie Only apply for 'House', because it uses the land size and the Departments don't have
        that feature value. In the Case of apartments is better to compered by the m2_const_price.

        Parameters
        ----------
        tb_name : str:  (Default value = 'main')
            This is Key Value Name in the self.df dictionary that contains the DataFrame you want to apply
            the method.

        Returns
        -------
        Series
        """

        df = self.df[tb_name]
        listing_type = df.tipo_inmueble.unique()[0]
        if listing_type == 'Casa':
            result = df.price / (df.m2_terreno + df.m2_const)
        else:
            raise ZeroDivisionError(
                f"The Method '{self.get_abs_price.__name__}' only works with 'Houses'. Please filter by 'Casas'")
        # Validate if is not inf Results.
        if result.mean() != np.inf:
            return result
        else:
            raise ValueError(
                f'Series mean is inf. Please clean inf values before use {self.get_m2_price.__name__}.')

    def set_sector_inmo(self, tb_name: str = 'main'):
        """Create a new Column call it 'sector_inmo' that contains the categories of the listings
        economical sectors.

        The categories are: 'Interés Social', 'Interés Medio', 'Residencial', 'Residencial Plus', 'Premium'.
        Depends on the type of offer (sell or rent) and the price that the label would be selected.

        Parameters
        ----------
        tb_name : str:  (Default value = 'main')
            This is Key Value Name in the self.df dictionary that contains the DataFrame you want to apply
            the method.

        Returns
        -------
        self
        """
        df = self.df[tb_name]
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

        return self

    def set_amenidades(self, tb_name: str = 'main', column: str = 'amenidades' ) -> tuple[DataFrame, list[str]]:
        """Create N number of new Columns of the total amenities in the column 'amenidades' lists.

        This column contains list of amenities, and it is variable. The program don't know how many new columns
        will be created. So this proces is O(n2).

        Parameters
        ----------
        tb_name : str:  (Default value = 'main')
            This is Key Value Name in the self.df dictionary that contains the DataFrame you want to apply
            the method.
        column: str : (Default value = 'amenidades')
            The column Name that contain the Raw List of amenities.

        Returns
        -------
        tuple[DataFrame, list[str]:
            The result is a DataFrame with all the new columns and a list of the name of the columns.
        """
        df = self.df[tb_name].reset_index()
        df.index += 1

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

        self.df[tb_name] = df

        return self.df[tb_name], list_amenidades

    def change_dtype(self, tb_name: str = 'main', **kwargs: Union[tuple, str]) -> DataFrame:
        """Change the dtypes from columns DataFrame.

        Parameters
        ----------
        tb_name : str:  (Default value = 'main')
            This is Key Value Name in the self.df dictionary that contains the DataFrame you want to apply
            the method.
        **kwargs: Union[tuple, str] :
            Expect a parameter call it 'columns': This would contain a tuple of the name of the columns that
            would be change it the dtype. Also expect the parameter 'astype' that contain the dtype conversion.

        Returns
        -------
        DataFrame
        """
        try:
            # Change Type
            dtypes = self.df[tb_name].loc[:, kwargs.get('columns')].astype(kwargs.get('astype', 'category'))
            # Adding Values
            self.df[tb_name].loc[:, kwargs.get('columns')] = dtypes
            return self.df[tb_name]
        except KeyError:
            raise KeyError(
                f'Error on method "{self.change_dtype.__name__}".'
                f'Pleas check the Values on parameters "columns" and/or "astype"')

    def filter_by(self, tb_name: str = 'main', **kwargs: str) -> DataFrame:
        """Segment the DataFrame by a Single Listing Characteristics.
        
        This subset would be filtered by the type of the listing and offer. E.g: 'Sell' and 'House'.

        Parameters
        ----------
        tb_name : str:  (Default value = 'main')
            This is Key Value Name in the self.df dictionary that contains the DataFrame you want to apply
            the method.

        **kwargs: str:
            Expect as parameter the name of the column and as argument the value inside the row you want to
            filter the data. This method only return equal results.

        Returns
        -------
        DataFrame
        """
        df = self.df[tb_name]
        # Select listing type and Offer.
        for col, val in kwargs.items():
            df = df.loc[df[col] == val]

        # Create a subset DataFrame if is not Empty  -> Filtered
        if not df.empty:
            self.df['filtered'] = df
            return self.df['filtered']
        else:
            raise ValueError(
                f'The DataFrame result is Empty,'
                f' please check the name of the columns on "{self.filter_by.__name__}" method')

    def drop_outliers(self, tb_name: str = 'main', **kwargs: tuple) -> DataFrame:
        """Remove the outliers from the DataFrame.
        
        This normally is used when you want to compare a listing whit others. The logic inside this function is
        that when the Skew (outliers) data is positive (right), it would remove principally the positive skew dataset
        above the 95% quantile. And for the negative skew data, it would remove principally the negative dataset
        below the 5% quantile.

        Parameters
        ----------
        tb_name : str:  (Default value = 'main')
            This is Key Value Name in the self.df dictionary that contains the DataFrame you want to apply
            the method.
        **kwargs: tuple :
            Expected as parameter 'columns' and as argument a 'tuple' with the name of the columns names.

        Returns
        -------
        DataFrame
        """
        df = self.df[tb_name]
        columns = kwargs.get('columns')
        if columns is None:
            raise ValueError(f'There are empty the parameters "columns" on the method "{self.drop_outliers.__name__}"')

        # Create Skew Series of the DataFrame
        skew_serie = df.loc[:, columns].skew()
        # Split Positive and Negative Skew Columns.
        positive_skew_cols = skew_serie[skew_serie.gt(1)].index
        negative_skew_cols = skew_serie[skew_serie.lt(-1)].index

        # Positive +
        if positive_skew_cols.size > 0:

            for col in positive_skew_cols:
                # Remove principal Outliers of the Right Side of the Histogram.
                df = df.loc[(df[col] > df[col].quantile(0.01)) & (df[col] < df[col].quantile(0.95))]

        # Negative -
        if negative_skew_cols.size < 0:

            for col in negative_skew_cols:
                # Remove principal Outliers of the Left Side of the Histogram.
                df = df.loc[(df[col] > df[col].quantile(0.05)) & (df[col] < df[col].quantile(0.99))]

        self.df[tb_name] = df
        return self.df[tb_name]

    def get_numeric_columns(self, tb_name: str = 'main') -> list:
        """Return a List of Columns that have numeric data.

        tb_name : str:  (Default value = 'main')
            This is Key Value Name in the self.df dictionary that contains the DataFrame you want to apply
            the method.

        Returns
        -------
        list
        """
        df = self.df[tb_name]
        numeric_columns = df.select_dtypes(include = [np.number]).columns
        return numeric_columns

    def get_categories_columns(self, tb_name: str = 'main') -> list:
        """Return a List of Columns that have Category data.

        tb_name : str:  (Default value = 'main')
            This is Key Value Name in the self.df dictionary that contains the DataFrame you want to apply
            the method.

        Returns
        -------
        list
        """
        df = self.df[tb_name]
        categories_columns = df.select_dtypes(include = 'category').columns
        return categories_columns
