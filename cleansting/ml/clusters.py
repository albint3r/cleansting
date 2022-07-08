from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from cleansting.cleansting import CleanSting


def set_dbscan_clusters(obj: CleanSting, tb_name: str = 'main', eps: float=1.5 / 6371.0088,
                  min_samples: int = 5, scaler: bool = False, **kwargs: str) -> CleanSting:
    """Create a new Column Call it 'clusters' inside the object DataFrame specify.

    Parameters
    ----------
    obj: CleanSting :
        Is an object of the Class CleanSting. This object has an attribute that contains a dictionary with DataFrames.
        
    tb_name : str:  (Default value = 'main')
            This is Key Value Name in the self.df dictionary that contains the DataFrame you want to apply
            the method.
    eps: float : (Default value = 1.5 / 6371.0088)

    min_samples: int : (Default value = 5)

    scaler: bool : (Default value = False)
        Scale X data applying the StandardScaler. This has a min of -1 a max of +1 and a median of 0.
    **kwargs: str:
        The name of the columns that contains the Latitud and Longitude of the Listings.

    Returns
    -------
    CleanSting

    """
    cl = obj
    model = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree', metric='haversine')
    # Set X Values.
    lat, long = kwargs.get('lat', 'lat'), kwargs.get('long', 'long')
    X = cl.df[tb_name].loc[:, [lat, long]].values
    if scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    # Create New Column: Cluster
    cl.df[tb_name]['cluster'] = model.fit_predict(X)
    return cl


def set_kclusters(obj: CleanSting, tb_name: str = 'main',
                  n_clusters: int = 100, scaler: bool = False, **kwargs: str) -> CleanSting:
    """Create a new Column Call it 'clusters' inside the object DataFrame specify.

    Parameters
    ----------
    obj: CleanSting :
        Is an object of the Class CleanSting. This object has an attribute that contains a dictionary with DataFrames.

    tb_name : str:  (Default value = 'main')
        This is Key Value Name in the self.df dictionary that contains the DataFrame you want to apply
        the method.
    n_clusters: int : (Default value = 100)
        Integer of the number of clusters will be created.
    scaler: bool : (Default value = False)
        Scale X data applying the StandardScaler. This has a min of -1 a max of +1 and a median of 0.
    **kwargs: str:
        The name of the columns that contains the Latitud and Longitude of the Listings.

    Returns
    -------
    CleanSting

    """
    cl = obj
    model = KMeans(n_clusters=n_clusters, random_state=42)
    # Set X Values.
    lat, long = kwargs.get('lat', 'lat'), kwargs.get('long', 'long')
    X = cl.df[tb_name].loc[:, [lat, long]].values
    if scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    # Create New Column: Cluster
    cl.df[tb_name]['cluster'] = model.fit_predict(X)
    return cl
