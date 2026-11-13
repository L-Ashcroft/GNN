"""Meaningful boundary tests: screening does not conflate failure and missingness."""
from audit import flags,native_number

def main():
    assert native_number('1G')==1e9 and native_number('1')==1
    assert flags(60,1e6,60,1,1e9)==[]
    assert flags(60,1e9,360,1,1e9)==[]
    assert flags(-5,4.1e11,-34210000,1,1e9)==['nonpositive_DC_gain_review','reported_ft_outside_measured_sweep','absolute_reported_phase_over360_review']
    assert flags(60,2e9,60,1,1e9)==['reported_ft_outside_measured_sweep']
    assert flags(60,1e6,-30,1,1e9)==[] # negative phase alone is not deleted as failed
    assert flags(60,1e6,370,1,1e9)==['absolute_reported_phase_over360_review']
    print('PASS: exact-boundary, extrapolation, negative-phase and independent-trigger checks')

if __name__=='__main__':main()
