# A simulation of death toll of COVID-19 by herd immunization
with underlying actual number of infections

*Zizheng Xu*

## The model
![model](https://github.com/ZizhengXu/my_portfoilio/blob/master/COV19simulation/img/model.png)

The infectious rate, $\beta$, controls the rate of spread which represents the probability of transmitting disease between a susceptible and an infectious individual. 

The incubation rate, $\sigma$, is the rate of latent individuals becoming infectious (average duration of incubation is $1/\sigma$). 

Recovery rate, $\gamma$ = 1/D, is determined by the average duration, D, of infection. 

For the SEIRS model, $\xi$ is the rate which recovered individuals return to the susceptible statue due to loss of immunity.

$\mu$ and $\nu$ represent the birth and death rates, respectively, and are assumed to be equal to maintain a constant population

A demo of the solution of the ODE system looks like this:


![benchmark](https://github.com/ZizhengXu/my_portfoilio/blob/master/COV19simulation/img/benchmark.png)
