import pytest
from soluciones import (
    ej_1_statmodels_intervalo_confianza,
    ej_2_sklearn_coeficientes,
)


def test_sol_1():
    prediction, conf_int = ej_1_statmodels_intervalo_confianza()
    
    assert 5.626376425466223 == pytest.approx(prediction)
    assert (4.512823540396073, 6.739929310536373) == pytest.approx(conf_int)
    

def test_sol_2():
    coeficientes = ej_2_sklearn_coeficientes()
    
    assert (
        (
            0.10221285489797609,
            -0.2598213445569269,
            -0.08990593114955843
        ) == pytest.approx(coeficientes)
    )
