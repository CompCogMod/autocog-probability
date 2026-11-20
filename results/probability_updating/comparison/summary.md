# Probability-updating theory comparison

Score: `cv_nll_per_trial` (lower is better). Fits are pooled per study; CV holds out whole subjects. Every theory shares the same reporting stage: fitted rounding to multiples of 5 and 10, and each study's response scale.

## Ranking per study (difference from the best theory, per trial)

| study | n | winner | bayes_noise | bayes_prob_noise | bayes_sampler | bayesian_sampler | compression_logodds_noise | linear_compression | llo_bias | llo_prob_noise | llo_sampler | weak_strong_evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| augenblick2025/1a | 10488 | linear_compression | 0.0820 | 0.0297 | 0.0336 | 0.0029 | 0.0454 | **0** | 0.0748 | 0.0173 | 0.0209 | 0.0748 |
| augenblick2025/1b | 7500 | bayesian_sampler | 0.0355 | 0.0024 | 0.0032 | **0** | 0.0366 | 0.0024 | 0.0360 | 0.0035 | 0.0023 | 0.0142 |
| augenblick2025/2 | 8000 | llo_prob_noise | 0.1612 | 0.0555 | 0.0648 | 0.0047 | 0.0427 | 0.0007 | 0.0443 | **0** | 0.0027 | 0.0314 |
| buser2018/main | 5346 | llo_sampler | 0.3810 | 0.4225 | 0.4038 | 0.3530 | 0.3525 | 0.3576 | 0.0204 | 0.0177 | **0** | 0.0204 |
| enke_graeber2023/A_beliefs | 6000 | weak_strong_evidence | 0.2234 | 0.2279 | 0.2066 | 0.1458 | 0.1161 | 0.1477 | 0.0080 | 0.1080 | 0.0479 | **0** |
| enke_graeber2023/A_beliefs_default | 1296 | weak_strong_evidence | 0.6505 | 0.4838 | 0.4284 | 0.2502 | 0.1166 | 0.3852 | 0.0213 | 0.2114 | 0.1017 | **0** |
| enke_graeber2023/B_beliefs | 5834 | weak_strong_evidence | 0.2357 | 0.1605 | 0.1552 | 0.0771 | 0.0245 | 0.0813 | 0.0250 | 0.0803 | 0.0547 | **0** |
| hill2017/exp1 | 11527 | llo_bias | 0.4595 | 0.8934 | 0.5670 | 0.2566 | 0.4028 | 0.8934 | **0** | 0.0739 | 0.2611 | 0.0000 |
| hill2017/exp2 | 3160 | weak_strong_evidence | 0.3284 | 0.6875 | 0.3757 | 0.0583 | 0.2668 | 0.6875 | 0.0132 | 0.0320 | 0.1808 | **0** |
| howe2022/exp1 | 602 | weak_strong_evidence | 0.1220 | 0.1881 | 0.1866 | 0.0507 | 0.0017 | 0.0516 | 0.0003 | 0.0501 | 0.0269 | **0** |
| howe2022/exp2 | 300 | weak_strong_evidence | 0.2104 | 0.2690 | 0.2683 | 0.0760 | 0.0347 | 0.0772 | 0.0000 | 0.0556 | 0.0305 | **0** |
| howe2022/exp3 | 300 | bayesian_sampler | 0.0552 | 0.0571 | 0.0572 | **0** | 0.0158 | 0.0000 | 0.0158 | 0.0000 | 0.0000 | 0.0158 |
| tappin2020/S1 | 7872 | llo_sampler | 0.0337 | 0.0775 | 0.0216 | 0.0155 | 0.0183 | 0.0776 | 0.0245 | 0.0547 | **0** | 0.0245 |
| tappin2020/S2 | 15872 | llo_sampler | 0.2471 | 0.2712 | 0.2497 | 0.2117 | 0.1967 | 0.2132 | 0.0107 | 0.0381 | **0** | 0.0100 |
| tappin2020/control | 8032 | llo_sampler | 0.0259 | 0.0413 | 0.0079 | 0.0057 | 0.0137 | 0.0406 | 0.0201 | 0.0378 | **0** | 0.0201 |
| xie_hayes2022/expt1 | 1970 | llo_prob_noise | 0.0982 | 0.0186 | 0.0190 | 0.0086 | 0.0174 | 0.0122 | 0.0788 | **0** | 0.0002 | 0.0305 |
| xie_hayes2022/expt2 | 1784 | compression_logodds_noise | 0.6808 | 0.3631 | 0.3024 | 0.0412 | **0** | 0.3229 | 0.1243 | 0.3043 | 0.2582 | 0.0619 |
| xie_hayes2022/expt3 | 2420 | llo_sampler | 0.1238 | 0.0420 | 0.0373 | 0.0324 | 0.0533 | 0.0361 | 0.0546 | 0.0046 | **0** | 0.0411 |

## Wins

- bayes_noise: 0 of 18 studies
- bayes_prob_noise: 0 of 18 studies
- bayes_sampler: 0 of 18 studies
- bayesian_sampler: 2 of 18 studies
- compression_logodds_noise: 1 of 18 studies
- linear_compression: 1 of 18 studies
- llo_bias: 1 of 18 studies
- llo_prob_noise: 2 of 18 studies
- llo_sampler: 5 of 18 studies
- weak_strong_evidence: 6 of 18 studies

## Best theory per `noise` (minimum over `mean_rule`), difference per trial

| study | winner | logodds_gaussian | probability_gaussian | sampling |
|---|---|---|---|---|
| augenblick2025/1a | probability_gaussian | 0.0454 | **0** | 0.0029 |
| augenblick2025/1b | sampling | 0.0142 | 0.0024 | **0** |
| augenblick2025/2 | probability_gaussian | 0.0314 | **0** | 0.0027 |
| buser2018/main | sampling | 0.0204 | 0.0177 | **0** |
| enke_graeber2023/A_beliefs | logodds_gaussian | **0** | 0.1080 | 0.0479 |
| enke_graeber2023/A_beliefs_default | logodds_gaussian | **0** | 0.2114 | 0.1017 |
| enke_graeber2023/B_beliefs | logodds_gaussian | **0** | 0.0803 | 0.0547 |
| hill2017/exp1 | logodds_gaussian | **0** | 0.0739 | 0.2566 |
| hill2017/exp2 | logodds_gaussian | **0** | 0.0320 | 0.0583 |
| howe2022/exp1 | logodds_gaussian | **0** | 0.0501 | 0.0269 |
| howe2022/exp2 | logodds_gaussian | **0** | 0.0556 | 0.0305 |
| howe2022/exp3 | sampling | 0.0158 | 0.0000 | **0** |
| tappin2020/S1 | sampling | 0.0183 | 0.0547 | **0** |
| tappin2020/S2 | sampling | 0.0100 | 0.0381 | **0** |
| tappin2020/control | sampling | 0.0137 | 0.0378 | **0** |
| xie_hayes2022/expt1 | probability_gaussian | 0.0174 | **0** | 0.0002 |
| xie_hayes2022/expt2 | logodds_gaussian | **0** | 0.3043 | 0.0412 |
| xie_hayes2022/expt3 | sampling | 0.0411 | 0.0046 | **0** |

Wins: logodds_gaussian 8, probability_gaussian 3, sampling 7
Total difference per trial summed over studies: logodds_gaussian 0.228, probability_gaussian 1.071, sampling 0.623

## Best theory per `mean_rule` (minimum over `noise`), difference per trial

| study | winner | bayes | compression | logodds_weighting | weak_strong_weighting |
|---|---|---|---|---|---|
| augenblick2025/1a | compression | 0.0297 | **0** | 0.0173 | 0.0748 |
| augenblick2025/1b | compression | 0.0024 | **0** | 0.0023 | 0.0142 |
| augenblick2025/2 | logodds_weighting | 0.0555 | 0.0007 | **0** | 0.0314 |
| buser2018/main | logodds_weighting | 0.3810 | 0.3525 | **0** | 0.0204 |
| enke_graeber2023/A_beliefs | weak_strong_weighting | 0.2066 | 0.1161 | 0.0080 | **0** |
| enke_graeber2023/A_beliefs_default | weak_strong_weighting | 0.4284 | 0.1166 | 0.0213 | **0** |
| enke_graeber2023/B_beliefs | weak_strong_weighting | 0.1552 | 0.0245 | 0.0250 | **0** |
| hill2017/exp1 | logodds_weighting | 0.4595 | 0.2566 | **0** | 0.0000 |
| hill2017/exp2 | weak_strong_weighting | 0.3284 | 0.0583 | 0.0132 | **0** |
| howe2022/exp1 | weak_strong_weighting | 0.1220 | 0.0017 | 0.0003 | **0** |
| howe2022/exp2 | weak_strong_weighting | 0.2104 | 0.0347 | 0.0000 | **0** |
| howe2022/exp3 | compression | 0.0552 | **0** | 0.0000 | 0.0158 |
| tappin2020/S1 | logodds_weighting | 0.0216 | 0.0155 | **0** | 0.0245 |
| tappin2020/S2 | logodds_weighting | 0.2471 | 0.1967 | **0** | 0.0100 |
| tappin2020/control | logodds_weighting | 0.0079 | 0.0057 | **0** | 0.0201 |
| xie_hayes2022/expt1 | logodds_weighting | 0.0186 | 0.0086 | **0** | 0.0305 |
| xie_hayes2022/expt2 | compression | 0.3024 | **0** | 0.1243 | 0.0619 |
| xie_hayes2022/expt3 | logodds_weighting | 0.0373 | 0.0324 | **0** | 0.0411 |

Wins: bayes 0, compression 4, logodds_weighting 8, weak_strong_weighting 6
Total difference per trial summed over studies: bayes 3.069, compression 1.221, logodds_weighting 0.212, weak_strong_weighting 0.345

## Parameter estimates

### bayes_noise

| study | sigma | epsilon | round10 | round5 |
|---|---|---|---|---|
| augenblick2025/1a | 0.242 | 0.359 | 0.22 | 0.477 |
| augenblick2025/1b | 0.15 | 0.333 | 0.298 | 0.35 |
| augenblick2025/2 | 0.226 | 0.209 | 0.0564 | 0.0691 |
| buser2018/main | 0.745 | 0.0983 | 0.316 | 0.687 |
| enke_graeber2023/A_beliefs | 0.787 | 0.53 | 0.237 | 0.729 |
| enke_graeber2023/A_beliefs_default | 1.6 | 0.538 | 0.0873 | 0.474 |
| enke_graeber2023/B_beliefs | 0.916 | 0.6 | 0.434 | 0.737 |
| hill2017/exp1 | 1.23 | 0.215 | 0.188 | 0.14 |
| hill2017/exp2 | 1.23 | 0.265 | 0.231 | 0.216 |
| howe2022/exp1 | 0.579 | 0.113 | 0.0563 | 0.0073 |
| howe2022/exp2 | 0.652 | 0.0531 | 0.04 | 0 |
| howe2022/exp3 | 0.563 | 0.0447 | 0.0405 | 0 |
| tappin2020/S1 | 1.11 | 0.42 | 0.164 | 0.005 |
| tappin2020/S2 | 1.12 | 0.481 | 0.115 | 0.0446 |
| tappin2020/control | 0.521 | 0.239 | 0.142 | 0.0605 |
| xie_hayes2022/expt1 | 1.12 | 0.3 |
| xie_hayes2022/expt2 | 1.13 | 0.55 | 0 | 0.0273 |
| xie_hayes2022/expt3 | 1.26 | 0.382 |

### bayes_prob_noise

| study | tau | epsilon | round10 | round5 |
|---|---|---|---|---|
| augenblick2025/1a | 0.0618 | 0.272 | 0.207 | 0.487 |
| augenblick2025/1b | 0.0328 | 0.3 | 0.281 | 0.365 |
| augenblick2025/2 | 0.051 | 0.163 | 0.0466 | 0.0718 |
| buser2018/main | 0.142 | 0.0765 | 0.288 | 0.693 |
| enke_graeber2023/A_beliefs | 0.097 | 0.496 | 0.234 | 0.729 |
| enke_graeber2023/A_beliefs_default | 0.0489 | 0.464 | 0.109 | 0.461 |
| enke_graeber2023/B_beliefs | 0.103 | 0.501 | 0.428 | 0.739 |
| hill2017/exp1 | 0.0138 | 0.514 | 0.212 | 0.128 |
| hill2017/exp2 | 0.0142 | 0.577 | 0.256 | 0.2 |
| howe2022/exp1 | 0.121 | 0.151 | 0.0574 | 0.0063 |
| howe2022/exp2 | 0.143 | 0.0713 | 0.0408 | 0 |
| howe2022/exp3 | 0.127 | 0.0407 | 0.041 | 0 |
| tappin2020/S1 | 0.146 | 0.448 | 0.119 | 0.007 |
| tappin2020/S2 | 0.152 | 0.446 | 0.102 | 0.0423 |
| tappin2020/control | 0.0778 | 0.241 | 0.134 | 0.0585 |
| xie_hayes2022/expt1 | 0.138 | 0.238 |
| xie_hayes2022/expt2 | 0.0565 | 0.356 | 0 | 0.0174 |
| xie_hayes2022/expt3 | 0.181 | 0.244 |

### bayes_sampler

| study | N | tau | epsilon | round10 | round5 |
|---|---|---|---|---|---|
| augenblick2025/1a | 200 | 0.0527 | 0.276 | 0.207 | 0.487 |
| augenblick2025/1b | 200 | 0.0322 | 0.241 | 0.271 | 0.373 |
| augenblick2025/2 | 200 | 0.0431 | 0.153 | 0.0466 | 0.0716 |
| buser2018/main | 14 | 0.0744 | 0.0845 | 0.294 | 0.693 |
| enke_graeber2023/A_beliefs | 20.9 | 0.0243 | 0.537 | 0.239 | 0.728 |
| enke_graeber2023/A_beliefs_default | 18.5 | 0.035 | 0.404 | 0.114 | 0.459 |
| enke_graeber2023/B_beliefs | 29 | 0.0819 | 0.498 | 0.429 | 0.739 |
| hill2017/exp1 | 12 | 0.0104 | 0.272 | 0.189 | 0.152 |
| hill2017/exp2 | 8.31 | 0.0104 | 0.312 | 0.241 | 0.214 |
| howe2022/exp1 | 45.7 | 0.0982 | 0.158 | 0.0574 | 0.0063 |
| howe2022/exp2 | 48.7 | 0.128 | 0.0727 | 0.0408 | 0.0001 |
| howe2022/exp3 | 135 | 0.12 | 0.0406 | 0.041 | 0 |
| tappin2020/S1 | 8.43 | 0.0357 | 0.424 | 0.124 | 0.0081 |
| tappin2020/S2 | 10 | 0.0279 | 0.527 | 0.0692 | 0.0826 |
| tappin2020/control | 27.9 | 0.0148 | 0.234 | 0.136 | 0.0636 |
| xie_hayes2022/expt1 | 77.3 | 0.132 | 0.239 |
| xie_hayes2022/expt2 | 18.1 | 0.044 | 0.269 | 0 | 0.0184 |
| xie_hayes2022/expt3 | 10 | 0.136 | 0.251 |

### bayesian_sampler

| study | N | b | tau | epsilon | round10 | round5 |
|---|---|---|---|---|---|---|
| augenblick2025/1a | 200 | 12.9 | 0.0638 | 0.219 | 0.204 | 0.489 |
| augenblick2025/1b | 200 | 1.71 | 0.0335 | 0.236 | 0.27 | 0.374 |
| augenblick2025/2 | 200 | 12.8 | 0.0382 | 0.161 | 0.0476 | 0.0717 |
| buser2018/main | 54.5 | 5.8 | 0.121 | 0.0588 | 0.296 | 0.693 |
| enke_graeber2023/A_beliefs | 110 | 13.6 | 0.0983 | 0.381 | 0.233 | 0.73 |
| enke_graeber2023/A_beliefs_default | 16.4 | 0.348 | 0.0115 | 0.404 | 0.0881 | 0.507 |
| enke_graeber2023/B_beliefs | 97.6 | 10.1 | 0.0918 | 0.407 | 0.433 | 0.739 |
| hill2017/exp1 | 15.8 | 0.159 | 0.005 | 0.284 | 0.187 | 0.191 |
| hill2017/exp2 | 8.34 | 0.0825 | 0.005 | 0.311 | 0.277 | 0.204 |
| howe2022/exp1 | 47 | 10.5 | 0.108 | 0.0616 | 0.0582 | 0.0057 |
| howe2022/exp2 | 46.7 | 14.3 | 0.116 | 0.0212 | 0.0413 | 0.0001 |
| howe2022/exp3 | 90.8 | 20 | 0.116 | 0.033 | 0.041 | 0 |
| tappin2020/S1 | 10.3 | 0.0776 | 0.0254 | 0.448 | 0.0998 | 0.0483 |
| tappin2020/S2 | 67 | 11.2 | 0.157 | 0.281 | 0.103 | 0.0422 |
| tappin2020/control | 24.4 | 0.179 | 0.0162 | 0.218 | 0.142 | 0.0642 |
| xie_hayes2022/expt1 | 9.35 | 0.501 | 0.0984 | 0.199 |
| xie_hayes2022/expt2 | 6.54 | 0.35 | 0.0271 | 0.194 | 0 | 0 |
| xie_hayes2022/expt3 | 26.6 | 1.2 | 0.162 | 0.219 |

### compression_logodds_noise

| study | lam | sigma | epsilon | round10 | round5 |
|---|---|---|---|---|---|
| augenblick2025/1a | 0.942 | 0.297 | 0.286 | 0.211 | 0.487 |
| augenblick2025/1b | 0.985 | 0.246 | 0.243 | 0.274 | 0.371 |
| augenblick2025/2 | 0.898 | 0.217 | 0.175 | 0.0468 | 0.075 |
| buser2018/main | 0.953 | 0.71 | 0.0726 | 0.346 | 0.692 |
| enke_graeber2023/A_beliefs | 0.924 | 0.886 | 0.331 | 0.238 | 0.729 |
| enke_graeber2023/A_beliefs_default | 0.944 | 1.1 | 0.213 | 0.115 | 0.473 |
| enke_graeber2023/B_beliefs | 0.859 | 0.938 | 0.275 | 0.458 | 0.734 |
| hill2017/exp1 | 0.993 | 1.03 | 0.23 | 0.193 | 0.145 |
| hill2017/exp2 | 0.992 | 1.06 | 0.271 | 0.236 | 0.222 |
| howe2022/exp1 | 0.728 | 0.545 | 0.0509 | 0.0569 | 0.0067 |
| howe2022/exp2 | 0.659 | 0.554 | 0.0206 | 0.0401 | 0 |
| howe2022/exp3 | 0.732 | 0.532 | 0.0471 | 0.0408 | 0 |
| tappin2020/S1 | 0.995 | 1.13 | 0.391 | 0.167 | 0.0074 |
| tappin2020/S2 | 0.931 | 1.16 | 0.268 | 0.143 | 0.0481 |
| tappin2020/control | 0.995 | 0.527 | 0.225 | 0.144 | 0.062 |
| xie_hayes2022/expt1 | 0.894 | 1.22 | 0.102 |
| xie_hayes2022/expt2 | 0.935 | 1.03 | 0.16 | 0 | 0.0214 |
| xie_hayes2022/expt3 | 0.932 | 1.28 | 0.193 |

### linear_compression

| study | lam | tau | epsilon | round10 | round5 |
|---|---|---|---|---|---|
| augenblick2025/1a | 0.886 | 0.0686 | 0.221 | 0.205 | 0.489 |
| augenblick2025/1b | 0.99 | 0.0346 | 0.289 | 0.278 | 0.367 |
| augenblick2025/2 | 0.882 | 0.0473 | 0.162 | 0.0476 | 0.0716 |
| buser2018/main | 0.814 | 0.132 | 0.0553 | 0.295 | 0.693 |
| enke_graeber2023/A_beliefs | 0.794 | 0.104 | 0.377 | 0.233 | 0.73 |
| enke_graeber2023/A_beliefs_default | 0.939 | 0.0351 | 0.459 | 0.126 | 0.456 |
| enke_graeber2023/B_beliefs | 0.822 | 0.0988 | 0.403 | 0.433 | 0.739 |
| hill2017/exp1 | 1 | 0.0138 | 0.514 | 0.212 | 0.128 |
| hill2017/exp2 | 1 | 0.0142 | 0.577 | 0.256 | 0.2 |
| howe2022/exp1 | 0.69 | 0.117 | 0.0611 | 0.0582 | 0.0057 |
| howe2022/exp2 | 0.619 | 0.123 | 0.0213 | 0.0413 | 0 |
| howe2022/exp3 | 0.694 | 0.121 | 0.0332 | 0.041 | 0 |
| tappin2020/S1 | 0.987 | 0.147 | 0.444 | 0.119 | 0.007 |
| tappin2020/S2 | 0.743 | 0.162 | 0.279 | 0.103 | 0.0423 |
| tappin2020/control | 0.983 | 0.0784 | 0.236 | 0.134 | 0.0585 |
| xie_hayes2022/expt1 | 0.93 | 0.14 | 0.21 |
| xie_hayes2022/expt2 | 0.949 | 0.0518 | 0.356 | 0 | 0.0172 |
| xie_hayes2022/expt3 | 0.912 | 0.178 | 0.213 |

### llo_bias

| study | alpha | beta | sigma | epsilon | round10 | round5 |
|---|---|---|---|---|---|---|
| augenblick2025/1a | 2.11 | 0.939 | 0.255 | 0.339 | 0.215 | 0.481 |
| augenblick2025/1b | 0.998 | 0.989 | 0.157 | 0.326 | 0.296 | 0.351 |
| augenblick2025/2 | 0.81 | 0.711 | 0.229 | 0.161 | 0.0513 | 0.0708 |
| buser2018/main | 1.01 | 0.3 | 0.339 | 0.159 | 0.314 | 0.686 |
| enke_graeber2023/A_beliefs | 0.858 | 0.366 | 0.667 | 0.323 | 0.236 | 0.729 |
| enke_graeber2023/A_beliefs_default | 0.884 | 0.154 | 0.804 | 0.233 | 0.109 | 0.474 |
| enke_graeber2023/B_beliefs | 0.509 | 0.45 | 1.03 | 0.222 | 0.439 | 0.739 |
| hill2017/exp1 | 1 | 0.0048 | 0.05 | 0.428 | 0.147 | 0.109 |
| hill2017/exp2 | 1.04 | 0.0946 | 0.212 | 0.434 | 0.222 | 0.231 |
| howe2022/exp1 | 0.723 | 0.694 | 0.544 | 0.0512 | 0.0569 | 0.0067 |
| howe2022/exp2 | 0.74 | 0.465 | 0.533 | 0.0247 | 0.0406 | 0 |
| howe2022/exp3 | 0.904 | 0.721 | 0.532 | 0.0471 | 0.0408 | 0 |
| tappin2020/S1 | 1 | 0.56 | 0.899 | 0.454 | 0.163 | 0.0027 |
| tappin2020/S2 | 0.945 | 0.16 | 0.723 | 0.286 | 0.124 | 0.0391 |
| tappin2020/control | 0.944 | 0.51 | 0.522 | 0.229 | 0.142 | 0.0606 |
| xie_hayes2022/expt1 | 0.736 | 0.674 | 1.3 | 0.134 |
| xie_hayes2022/expt2 | 0.828 | 0.295 | 1.2 | 0.165 | 0 | 0.0261 |
| xie_hayes2022/expt3 | 0.865 | 0.328 | 1.21 | 0.215 |

### llo_prob_noise

| study | alpha | beta | tau | epsilon | round10 | round5 |
|---|---|---|---|---|---|---|
| augenblick2025/1a | 2.11 | 0.878 | 0.0665 | 0.241 | 0.205 | 0.488 |
| augenblick2025/1b | 0.998 | 0.999 | 0.0328 | 0.299 | 0.281 | 0.365 |
| augenblick2025/2 | 0.828 | 0.713 | 0.0471 | 0.163 | 0.048 | 0.071 |
| buser2018/main | 0.995 | 0.284 | 0.0627 | 0.149 | 0.307 | 0.69 |
| enke_graeber2023/A_beliefs | 0.79 | 0.433 | 0.0925 | 0.383 | 0.234 | 0.73 |
| enke_graeber2023/A_beliefs_default | 1 | 0.125 | 0.0227 | 0.437 | 0.152 | 0.44 |
| enke_graeber2023/B_beliefs | 0.368 | 0.703 | 0.102 | 0.406 | 0.429 | 0.739 |
| hill2017/exp1 | 1 | 0.0012 | 0.005 | 0.422 | 0.143 | 0.0659 |
| hill2017/exp2 | 1 | 0.0009 | 0.005 | 0.492 | 0.181 | 0.147 |
| howe2022/exp1 | 2.63 | 0.656 | 0.117 | 0.0614 | 0.0582 | 0.0057 |
| howe2022/exp2 | 0.701 | 0.442 | 0.119 | 0.024 | 0.0414 | 0 |
| howe2022/exp3 | 2.11 | 0.683 | 0.121 | 0.0332 | 0.041 | 0 |
| tappin2020/S1 | 1.06 | 0.397 | 0.105 | 0.493 | 0.128 | 0.0069 |
| tappin2020/S2 | 0.909 | 0.14 | 0.105 | 0.304 | 0.105 | 0.0422 |
| tappin2020/control | 0.941 | 1.76 | 0.0792 | 0.233 | 0.134 | 0.0585 |
| xie_hayes2022/expt1 | 1.04 | 0.732 | 0.133 | 0.234 |
| xie_hayes2022/expt2 | 1.29 | 0.675 | 0.0428 | 0.391 | 0 | 0.0179 |
| xie_hayes2022/expt3 | 0.851 | 0.474 | 0.153 | 0.237 |

### llo_sampler

| study | alpha | beta | N | tau | epsilon | round10 | round5 |
|---|---|---|---|---|---|---|---|
| augenblick2025/1a | 1.51 | 0.879 | 200 | 0.0584 | 0.243 | 0.205 | 0.488 |
| augenblick2025/1b | 0.997 | 0.982 | 200 | 0.0324 | 0.239 | 0.271 | 0.373 |
| augenblick2025/2 | 0.823 | 0.71 | 200 | 0.0345 | 0.159 | 0.0482 | 0.0709 |
| buser2018/main | 0.993 | 0.29 | 54.5 | 0.0273 | 0.144 | 0.312 | 0.691 |
| enke_graeber2023/A_beliefs | 0.822 | 0.391 | 24.4 | 0.014 | 0.385 | 0.235 | 0.73 |
| enke_graeber2023/A_beliefs_default | 0.964 | 0.125 | 87.2 | 0.005 | 0.39 | 0.118 | 0.463 |
| enke_graeber2023/B_beliefs | 0.487 | 0.522 | 12 | 0.0356 | 0.324 | 0.434 | 0.738 |
| hill2017/exp1 | 1.04 | 0.0983 | 200 | 0.005 | 0.358 | 0.186 | 0.143 |
| hill2017/exp2 | 1.05 | 0.0842 | 200 | 0.005 | 0.428 | 0.226 | 0.223 |
| howe2022/exp1 | 3 | 0.661 | 18.9 | 0.0216 | 0.0641 | 0.0549 | 0.0123 |
| howe2022/exp2 | 0.702 | 0.439 | 25.4 | 0.0761 | 0.0202 | 0.0412 | 0.0001 |
| howe2022/exp3 | 2.97 | 0.683 | 187 | 0.116 | 0.0331 | 0.041 | 0 |
| tappin2020/S1 | 0.954 | 0.589 | 8.47 | 0.0351 | 0.402 | 0.125 | 0.0084 |
| tappin2020/S2 | 0.837 | 0.168 | 10.2 | 0.028 | 0.218 | 0.0537 | 0.097 |
| tappin2020/control | 0.922 | 2.42 | 27.8 | 0.0146 | 0.223 | 0.137 | 0.0639 |
| xie_hayes2022/expt1 | 1.04 | 0.73 | 200 | 0.13 | 0.234 |
| xie_hayes2022/expt2 | 1.15 | 0.304 | 31.2 | 0.0244 | 0.335 | 0 | 0.0223 |
| xie_hayes2022/expt3 | 0.799 | 0.469 | 9.4 | 0.0921 | 0.226 |

### weak_strong_evidence

| study | alpha | beta | gamma | sigma | epsilon | round10 | round5 |
|---|---|---|---|---|---|---|---|
| augenblick2025/1a | 3 | 0.951 | 0.974 | 0.256 | 0.339 | 0.214 | 0.482 |
| augenblick2025/1b | 0.982 | 1.04 | 0.772 | 0.272 | 0.208 | 0.269 | 0.375 |
| augenblick2025/2 | 0.829 | 0.482 | 0.618 | 0.226 | 0.159 | 0.052 | 0.0699 |
| buser2018/main | 1.01 | 0.287 | 0.732 | 0.339 | 0.159 | 0.314 | 0.686 |
| enke_graeber2023/A_beliefs | 0.855 | 0.478 | 0.646 | 0.666 | 0.313 | 0.236 | 0.729 |
| enke_graeber2023/A_beliefs_default | 0.865 | 0.412 | 0.442 | 0.75 | 0.241 | 0.111 | 0.473 |
| enke_graeber2023/B_beliefs | 0.434 | 0.765 | 0.655 | 0.965 | 0.229 | 0.445 | 0.74 |
| hill2017/exp1 | 1 | 0.0052 | 0.101 | 0.05 | 0.428 | 0.147 | 0.109 |
| hill2017/exp2 | 1 | 0.0046 | 0.1 | 0.05 | 0.503 | 0.182 | 0.194 |
| howe2022/exp1 | 1.51 | 0.678 | 1.37 | 0.54 | 0.0531 | 0.0568 | 0.0067 |
| howe2022/exp2 | 0.748 | 0.47 | 1.1 | 0.533 | 0.0247 | 0.0406 | 0 |
| howe2022/exp3 | 1.51 | 0.711 | 0.972 | 0.532 | 0.0471 | 0.0408 | 0 |
| tappin2020/S1 | 1 | 0.672 | 1.5 | 0.899 | 0.454 | 0.163 | 0.0027 |
| tappin2020/S2 | 0.951 | 0.198 | 0.561 | 0.723 | 0.286 | 0.124 | 0.0392 |
| tappin2020/control | 0.944 | 2.42 | 0.318 | 0.522 | 0.229 | 0.142 | 0.0606 |
| xie_hayes2022/expt1 | 0.594 | 0.855 | 0.1 | 1.24 | 0.115 |
| xie_hayes2022/expt2 | 0.698 | 1.12 | 0.432 | 1.09 | 0.168 | 0 | 0.0256 |
| xie_hayes2022/expt3 | 0.799 | 0.662 | 0.687 | 1.22 | 0.192 |

