import dask.array
import sparse
from low_rank_data import XY, XY_incomplete, missing_mask
from common import reconstruction_error

from fancyimpute import SoftImpute, SoftImputeWarmStarts

def test_soft_impute_with_low_rank_random_matrix():
    solver = SoftImpute(shrinkage_value=1)
    XY_completed = solver.fit_transform(XY_incomplete)
    _, missing_mae = reconstruction_error(
        XY,
        XY_completed,
        missing_mask,
        name="SoftImpute")
    assert missing_mae < 0.1, "Error too high!"

def test_soft_impute_warm_starts():
    runner = SoftImputeWarmStarts()
    runner.run(XY_incomplete)

def test_soft_impute_dask():
    solver = SoftImpute()
    dask_incomplete = dask.array.from_array(XY_incomplete)
    XY_completed = solver.fit_transform(dask_incomplete)
    _, missing_mae = reconstruction_error(
        XY,
        XY_completed,
        missing_mask,
        name="SoftImpute")
    assert missing_mae < 0.1, "Error too high!"


if __name__ == "__main__":
    #test_soft_impute_warm_starts()
    test_soft_impute_with_low_rank_random_matrix()
