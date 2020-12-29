
## Differential Cepstrum Zero/Pole Extraction  
This is a python implementation of a new technique for extracting the zeros and poles of a system. As input we take an impulse response. From this we calculate its differential cepstrum and fit it using Prony's Method.  
  
`prony_cep.py` demonstrates this on analytic signals.  
`real_signal.py` demonstrates this on real signals.  
`cepstrum_utils.py` has all of the necessary functions.  
  
## Analytic Example  
![ground truth differential cepstrum](https://github.com/ChristianYost/ZP_cep_est/blob/master/figs_pics/a_orig_diff_cep.png)! [ground truth zeros/poles](https://github.com/ChristianYost/ZP_cep_est/blob/master/figs_pics/a_orig_zp.png)    
![estimated differential cepstrum](https://github.com/ChristianYost/ZP_cep_est/blob/master/figs_pics/a_appx_diff_cep.png) ![estimate zeros/poles](https://github.com/ChristianYost/ZP_cep_est/blob/master/figs_pics/a_est_zp.png)  
 
