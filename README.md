# LLMAir
Accurate and timely air quality prediction is crucial for cities and individuals to effectively take necessary precautions against potential air pollution. Existing studies typically rely on building prediction models based on large-scale monitoring data, often designed for specific tasks. Recently, pre-trained large language models (LLMs) have achieved significant progress in various time series analysis tasks due to their powerful representation and inference capabilities. However, their application to air quality data with spatio-temporal features remains largely unexplored. In this work, we propose LLMAir, an adaptive reprogramming approach that adapts pre-trained LLMs for air quality prediction. We first construct spatiotemporal tokens based on monitoring stations by integrating value, node, and time embeddings. Next, we design an adaptive semantic-enhanced reprogramming module to compute similarity matching scores between our spatiotemporal tokens and pre-trained word embeddings for alignment. We employ a semantic regulator to generate the optimal length of word prototypes, which serve as prompt prefixes for adaptive reprogramming and guiding the spatiotemporal token embeddings into the frozen LLM. Additionally, we jointly optimize predictive error and alignment loss to train our model. Experimental results demonstrate that LLMAir achieves state-of-the-art performance in air quality prediction and few-shot forecasting across two real-world datasets. 

### Runing

```shell
bash scripts/llmair.sh
```
