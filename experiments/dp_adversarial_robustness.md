# Adversarial robust auction learning using differential privacy

This document outlines preliminary results from experiments designed to explore the efficacy of using Differentially Private Stochastic Gradient Descent (DPSGD) in training models for multi-bidder, multi-item auctions. A key focus is on evaluating model robustness under conditions of an online attack, where an attacker aims to find optimal misreports for a specific bidder. These attack simulations are facilitated by the `attack_mode=online` parameter.

## Experiment Setup

The models are trained under various configurations with and without differential privacy, and in some cases with the presence of an online attacker. A noise scale of 0.9 was used for DPSGD. Our primary metric of interest is the **score**, computed as $\sqrt{\text{Revenue}} - \sqrt{\text{Regret}}$, with a higher score indicating better performance. It is important to note that experiments involving only one bidder were omitted, as in such cases, the attacking bidder can completely control the valuation profile of the auction, skewing results.

## Preliminary Results

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

## Areas for Further Interpretation

### Differential Privacy Impact
- **Robustness Under Attack**: Preliminary results suggest that the incorporation of DPSGD may enhance the model's robustness to online attacks, as indicated by score improvements in several configurations under attack conditions.
- **Score Fluctuations and Stochastic Regularization**: Notably, in the 3x10 setting, the score with differential privacy is higher than without, which might suggest that the benefits seen may not solely be due to enhanced privacy but could also be a result of stochastic regularization effects. This observation warrants a detailed investigation to discern the impact of DPSGD on score performance.

### Excluded Configurations
- **Single Bidder Scenarios**: The decision to exclude single-bidder experiments was based on the complete control a single bidder would have over the auction, potentially skewing results. It might be interesting to test whether auction learning can still benefit from DPSGD in this setting. 

## Future Work
The results obtained call for a detailed analysis to confirm the preliminary theses. Conducting smaller, targeted experiments could help in clarifying the observed effects and in validating the hypothesized benefits of differential privacy and stochastic regularization during training of TPAL.

