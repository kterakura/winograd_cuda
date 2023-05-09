# Winograd Algorithmと単純な畳み込みアルゴリズムをGPUで比較
 
Coalesced Accessや和をとる際に衝突を避けるために重みは深さ優先で保持し、さらに重みの入れ替えをおこなっている。


## winogradとの比較 on my GPU(GeForce GTX 1650)
| | time1[ms] | time2(winograd)[ms] | time1/time2 |
| ---- | ---- | ---- | ---- |
| input3232_16 | 0.024704 | 0.018624 | 1.326460 |
| input1616_32 | 0.017568 | 0.013920 | 1.262069 |
| input88_64   | 0.033888 | 0.018464 | 1.835355 |

## winogradとの比較 on JetsonNano 4GB(-arch=sm_50)
| | time1[ms] | time2(winograd)[ms] | time1/time2 |
| ---- | ---- | ---- | ---- |
| input3232_16 | 0.261094 | 0.199844 | 1.306489 |
| input1616_32 | 0.309323 | 0.203073 | 1.523211 |
| input88_64   | 0.240104 | 0.159844 | 1.502115 |

