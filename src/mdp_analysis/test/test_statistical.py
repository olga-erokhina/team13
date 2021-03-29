import pytest
import numpy as np
import statistical as st
import pandas as pd
from pandas._testing import assert_frame_equal, assert_series_equal


mydata = pd.DataFrame(data={'col1':[1,2],'col2':[1e-6,2e-6],'col3':[3,4]})
myref = pd.DataFrame(data={'col1':[1,2],'col3':[3,4]})
drop_ref = {(myref.columns[0],myref.columns[0]),(myref.columns[1],myref.columns[1]),(myref.columns[1],myref.columns[0])}
corr2ref = myref.corr().unstack().drop(labels=drop_ref).sort_values(ascending=False, key=lambda col: col.abs())
distref = np.asarray([2*2**(0.5)])


def test_check_if_significant():
    data, ref = st.check_if_significant(mydata)
    assert_frame_equal(data, myref)

def test_get_correlation_measure():
    corr2 = st.get_correlation_measure(myref)
    assert_series_equal(corr2, corr2ref)


def test_euclidean_distance():
    dist = st.euclidean_distance([0],[1],np.asarray(myref).T)
    np.testing.assert_array_equal(dist, distref)
