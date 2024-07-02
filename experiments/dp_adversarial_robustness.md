# Adversarial robust auction learning using differential privacy

This document outlines results from experiments designed to explore the efficacy of using Differentially Private Stochastic Gradient Descent (DPSGD) in training models for multi-bidder, multi-item auctions. A key focus is on evaluating model robustness under conditions of an online attack, where an attacker aims to find optimal misreports for a specific bidder. These attack simulations are facilitated by the `attack_mode=online` parameter.

## Experiment Setup

The models are trained under various configurations with and without differential privacy, and in some cases with the presence of an online attacker. A noise scale of 0.9 was used for DPSGD. Our primary metric of interest is the **score**, computed as $\sqrt{\text{Revenue}} - \sqrt{\text{Regret}}$, with a higher score indicating better performance. It is important to note that experiments involving only one bidder were omitted, as in such cases, the attacking bidder can completely control the valuation profile of the auction, skewing results.

### Excluded Configurations
- **Single Bidder Scenarios**: The decision to exclude single-bidder experiments was based on the complete control a single bidder would have over the auction, potentially skewing results. It might be interesting to test whether auction learning can still benefit from DPSGD in this setting. 

## Results

Below is a table summarizing the outcomes of our experiments across different configurations:

|   bidders |   items |    score |      regret |   revenue | attack   | differential privacy  |
|----------:|--------:|---------:|------------:|----------:|:---------|:---------------|
|         2 |       2 | 0.807201 | 0.033857    |  0.982485 |          | False          |
|         2 |       2 | 0.58268  | 0.0414263   |  0.618134 | online   | False          |
|         2 |       2 | 0.62097  | 0.0142761   |  0.548271 | online   | true           |
|         2 |       2 | 0.593785 | 0.15223     |  0.96816  |          | true           |
|         3 |      10 | 1.87873  | 7.368e-07   |  3.53286  |          | False          |
|         3 |      10 | 1.84387  | 5.0615e-07  |  3.40248  | online   | False          |
|         3 |      10 | 1.92077  | 0.129666    |  5.20232  | online   | true           |
|         3 |      10 | 2.06579  | 0.0418628   |  5.15471  |          | true           |
|         5 |      10 | 2.38727  | 0.000578843 |  5.81453  |          | False          |
|         5 |      10 | 2.03834  | 0.0380946   |  4.98858  | online   | False          |
|         5 |      10 | 2.09256  | 0.049628    |  5.36078  | online   | true           |
|         5 |      10 | 2.26136  | 0.0120752   |  5.62282  |          | true           |

![barchart for 2x2, 3x10 and 5x10](https://github.com/mechanism-learning-research/two-player-auctions/assets/13840966/6f0e9209-6c3a-4acb-a5d2-346d518d6e3d)

## Interpretation of Results

### Differential Privacy Impact
- **Robustness Under Attack**: Preliminary results suggest that the incorporation of DPSGD may enhance the model's robustness to online attacks, as indicated by score improvements in several configurations under attack conditions.
- **Score Fluctuations and Stochastic Regularization**: Notably, in the 3x10 setting, the score with differential privacy is higher than without, which might suggest that the benefits seen may be a result of stochastic regularization effects.

To investigate this further, we ran several smaller experiments on different configurations and observed model performance in presence or absence of differential privacy and adversarial attacks.

We trained models for the 2x3 and 3x3 setting for 10000 steps and observed that while DPSGD leads to comparable performance as classical SGD in absence of an attacker, the attacker along with DPSGD may even improve the model score with regards to the baseline setting. This is likely due to additional stochastic regularization effects, as the attacker model cannot easily learn a useful valuation distribution after merely 10000 training steps, especially in presence of the gradient noise caused by DPSGD.

![barchart for 2x3 and 3x3](https://github.com/mechanism-learning-research/two-player-auctions/assets/13840966/85e6abf8-1208-42ae-9c56-67b12f97f693)

We further trained models for the 3x4 setting for 240000 steps and observed their performance over time.

![performance over time for 3x4](https://github.com/mechanism-learning-research/two-player-auctions/assets/13840966/9b8ff6b3-e586-444a-aaf3-c02c2e9c77bd)

The results suggest that an online attacker can improve model score even after relatively long training. As the models are trained longer, however, the auction learner's regret increases more and more, as the attacker gets better and better at exploiting the auction rules.

As expected, the effect of differential privacy on the performance metrics is not as pronounced as the effect of the attacker, at least at a moderate noise scale setting.

We use the loss function $-(\sqrt{\text{Revenue}} - \sqrt{\text{Regret}}) + \text{Regret}$, suggested in [Rahme, Jelassi, Weinberg (2021)](https://arxiv.org/abs/2006.05684), for the training of the auction. This loss function induces a bias towards low regret mechanisms, and was claimed to speed up training. This bias seems to be alleviated by differential privacy without hurting model score in early stages of training. It might be worthwhile to introduce noise scale scheduling and basically interpolate between DPSGD and SGD over time to speed up training even further.

## Conclusion

In conclusion, our experiments suggest that incorporating differential privacy through DPSGD holds promise for enhancing the robustness of auction learning models against online attacks. While further research is needed to explore the generalizability of these findings and the impact of various factors like noise scale and attacker sophistication, the observed improvements in score under attack conditions, potentially driven by stochastic regularization effects, warrant further investigation. It is important to note that in this context, differential privacy is primarily leveraged as a tool to improve robustness against adversarial attacks, rather than directly protecting bidder privacy. This research contributes to the growing body of work on robust machine learning and its application to complex real-world problems like auction design. By developing robust auction mechanisms, we can strive towards fairer, more transparent, and trustworthy online marketplaces.

## Limitations

While these experiments provide promising initial insights into the potential of DPSGD for enhancing robustness in auction learning, it's important to acknowledge certain limitations. The study focuses on a specific auction format and a limited range of bidder and item configurations. Further research is needed to explore the generalizability of these findings to other auction types and more complex settings. Additionally, the analysis primarily considers a single noise scale for DPSGD. Investigating the impact of varying noise levels on model performance and robustness could provide a more comprehensive understanding of the trade-offs involved. Finally, the attacker model used in these experiments is relatively simple. Exploring more sophisticated attack strategies could shed light on the limitations of DPSGD in defending against more advanced adversarial tactics.
