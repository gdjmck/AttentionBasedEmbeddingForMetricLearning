# AttentionBasedEmbeddingForMetricLearning
Pytorch Implementation of paper Attention-based Ensemble for Deep Metric Learning

**Major difference from the paper:**
attention maps are not followed by a sigmoid activation function and minmax norm are used instead.


**The weighted sampling module code is copied from [suruoxi/DistanceWeightedSampling](https://github.com/suruoxi/DistanceWeightedSampling.git)**


performance on Stanford Cars 196:  71.4% recall@1    86.9% recall@4 (8 attentions and size of each embedding is 64)

## TODO:
transform attention map: ```att_maps = sign(att_maps) * sqrt(abs(att_maps))``` before normalizing. (Motivated by [tau-yihouxiang/WSDAN](https://github.com/tau-yihouxiang/WS_DAN))


**Will update here if I got better validation performance**
