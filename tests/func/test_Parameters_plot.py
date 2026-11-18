import warnings

import geopandas
import matplotlib
import numpy as np
import pytest
from shapely.geometry import box, LineString

from pyPRMS import MetaData, Parameters

matplotlib.use('Agg')


@pytest.fixture(scope='class')
def plot_pdb(tmp_path_factory):
    """Parameters instance with 2 HRUs, 2 segments, and matching geometry (EPSG:4326)"""
    tmp_dir = tmp_path_factory.mktemp('plot_geom')

    hru_file = tmp_dir / 'hrus.geojson'
    seg_file = tmp_dir / 'segs.geojson'

    hru_gdf = geopandas.GeoDataFrame({'hru_id': [1, 2]},
                                     geometry=[box(-100.0, 40.0, -99.0, 41.0),
                                               box(-99.0, 40.0, -98.0, 41.0)],
                                     crs='EPSG:4326')
    seg_gdf = geopandas.GeoDataFrame({'seg_id': [1, 2]},
                                     geometry=[LineString([(-99.8, 40.2), (-99.0, 40.5)]),
                                               LineString([(-99.0, 40.5), (-98.2, 40.8)])],
                                     crs='EPSG:4326')
    hru_gdf.to_file(hru_file, driver='GeoJSON')
    seg_gdf.to_file(seg_file, driver='GeoJSON')

    pdb = Parameters(metadata=MetaData(verbose=False).metadata)
    pdb.dimensions.add(name='nhru', size=2)
    pdb.dimensions.add(name='nsegment', size=2)

    pdb.add(name='cov_type')
    pdb['cov_type'].data = np.array([1, 2], dtype=np.int32)

    pdb.add(name='seg_length')
    pdb['seg_length'].data = np.array([100.0, 200.0], dtype=np.float32)

    pdb.shapefile_hrus(str(hru_file), shape_key='hru_id')
    pdb.shapefile_segments(str(seg_file), shape_key='seg_id')
    return pdb


class TestParametersPlot:
    @pytest.mark.parametrize('name', ['cov_type',        # HRU plot path
                                      'seg_length'])     # segment plot path
    def test_plot_formatters(self, plot_pdb, name, tmp_path):
        """plot() must not raise NameError or emit cartopy formatter deprecation warnings"""
        import matplotlib.pyplot as plt

        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter('always')
            plot_pdb.plot(name, output_dir=str(tmp_path), limits='absolute')

        plt.close('all')

        assert (tmp_path / f'{name}.png').exists()

        formatter_warnings = [str(ww.message) for ww in wlist
                              if issubclass(ww.category, DeprecationWarning) and 'Formatter' in str(ww.message)]
        assert formatter_warnings == []
