
# winogradとの比較
| | time1[ms] | time2(winograd)[ms] | elapsed_time_ms1/elapsed_time_ms2 |
| ---- | ---- | ---- | ---- |
| input3232_3_output3232_16 | 0.010112 | 0.012608 | 0.802030 |
| input3232_16_output3232_16 | 0.032768 | 0.026496 | 1.236715 |
| input3232_16_output1616_32 | 0.024032 | 0.046208 | 0.686169 |
| input1616_32_output1616_32 | 0.030720 | 0.020864 | 1.472393 |
| input1616_32_output88_64 | 0.022176 | 0.032768 | 0.676758 |
| input88_64_output88_64 | 0.028480 | 0.024576 | 1.158854 |


# on JetsonNano 2GB
| | time1[ms] | time2(winograd)[ms] | elapsed_time_ms1/elapsed_time_ms2 |
| ---- | ---- | ---- | ---- |
| input3232_3_output3232_16 | 1.206875 | 0.681771 | 0.564906 |
| input3232_16_output3232_16 | 3.077604 | 2.856875 | 1.077262 |
| input3232_16_output1616_32 | 3.323542 | 4.283802 | 0.782452 |
| input1616_32_output1616_32 | 2.856094 | 2.238281 | 1.276021 |
| input1616_32_output88_64 | 2.613073 | 3.570260 | 0.731900 |
| input88_64_output88_64 | 2.629323 | 1.843437 | 1.426316 |

# same tiling
| | time1(all block)[ms] | time2(per block)[ms] |
| ---- | ---- | ---- |
| input3232_16_output1616_16 | 31.323296 | 35.233791 |
| input1616_32_output1616_32 | 30.318592 | 34.079617 |
| input88_64_output88_64 | 26.871489 | 31.176704 |


# how to implement