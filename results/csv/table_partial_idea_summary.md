# Partial Equivariance quick sweep (L=3,p=2, seeds=0,1)

## Test accuracy (mean)
| subgroup | train=30 | train=120 |
| --- | --- | --- |
| C4 | 0.477 | 0.535 |
| D2_V4 | 0.491 | 0.523 |
| D4 | 0.482 | 0.540 |
| Z2_reflection | 0.489 | 0.534 |
| Z2_rot180 | 0.469 | 0.521 |
| none | 0.472 | 0.488 |


## Train accuracy (mean)
| subgroup | train=30 | train=120 |
| --- | --- | --- |
| C4 | 0.783 | 0.617 |
| D2_V4 | 0.783 | 0.600 |
| D4 | 0.783 | 0.654 |
| Z2_reflection | 0.883 | 0.717 |
| Z2_rot180 | 0.867 | 0.642 |
| none | 0.933 | 0.700 |


## Generalization gap (mean)
| subgroup | train=30 | train=120 |
| --- | --- | --- |
| C4 | 0.307 | 0.082 |
| D2_V4 | 0.293 | 0.077 |
| D4 | 0.302 | 0.114 |
| Z2_reflection | 0.394 | 0.182 |
| Z2_rot180 | 0.397 | 0.121 |
| none | 0.462 | 0.212 |

