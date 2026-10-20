# SC5002 Lab Assignment — Search Algorithms & Regularised Regression

**Course:** SC5002 Artificial Intelligence Fundamentals & Applications, Nanyang Technological University  
**Author:** YU XIYAO (U2440309D)  
**Semester:** AY2026/27 Semester 1

This repository contains the practical component (Part 2) of the SC5002 lab assignment: a comparison of Linear Regression and Ridge Regression on the California Housing dataset using five-fold cross-validation. Linear Regression and weakly regularised Ridge achieve almost identical validation scores, while a large Ridge penalty substantially reduces predictive performance. The accompanying report for submission on NTULearn contains the Part 1 search exercises and the “Learn with AI” prompts, explanations and reflections.

## Repository contents

| File | Description |
|---|---|
| `SC5002_Part2_Regression.ipynb` | Colab notebook: template Cells 1–3 (Tasks 2.1–2.3) and extension Cells 4–8 |
| `model_comparison.png` | Task 2.3 bar chart: mean five-fold CV R² for each model |
| `alpha_sweep.png` | Extension: CV R² across 50 values of α from 0.001 to 10,000 |
| `coefficient_shrinkage.png` | Extension: coefficient paths as α increases |
| `README.md` | Project overview, methods, results and AI-use disclosure |

## How to run

1. Open [Google Colab](https://colab.research.google.com) and upload `SC5002_Part2_Regression.ipynb`, or open the notebook from this repository using Colab’s GitHub option.
2. Select **Runtime → Run all** and execute the cells in order.
3. Review the printed results and inline charts. The plotting cells also save PNG files in the Colab session; download them if you want to retain local copies.

**Environment:** Python 3, scikit-learn, pandas, NumPy and Matplotlib. These libraries are normally available in Colab. Loading the dataset requires internet access on the first run unless it is already cached; runtime depends on the download and session environment.

The random sample and train/test split both use `random_state=42`. All compared models use the same five cross-validation folds. The template’s `cv=5` uses non-shuffled folds for regression; it does not generate five independent random train/test splits. Small numerical differences may occur across library versions.

## Dataset

The [California Housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) is based on the 1990 US Census. Each row represents a census block group, referred to here as a district. The experiment uses a random sample of 1,000 of the dataset’s 20,640 rows.

| Variable | Role | Meaning |
|---|---|---|
| `MedHouseVal` | **Continuous target** | Median house value, in units of $100,000 |
| `MedInc` | Feature | Median income in the district, in units of $10,000 |
| `HouseAge` | Feature | Median age of houses in the district, in years |
| `AveRooms` | Feature | Average number of rooms per household |
| `AveBedrms` | Feature | Average number of bedrooms per household |
| `Population` | Feature | District population |
| `AveOccup` | Feature | Average number of people per household |
| `Latitude` | Feature | District latitude |
| `Longitude` | Feature | District longitude |

Row indices are identifiers, not input features. The target has an upper recording limit of `5.00001` ($500,001), so districts recorded at this ceiling do not reveal the full variation in higher house values.

## Method

1. **Preprocessing:** Standardise all eight input features using `StandardScaler`. Standardisation makes a one-unit change correspond to one standard deviation and prevents measurement units from determining how strongly Ridge penalises a feature’s coefficient. For ordinary least squares with an intercept, consistent scaling does not change predictions in exact arithmetic, although it can improve numerical conditioning.
2. **Train/test split:** Divide the 1,000 sampled rows into 800 training rows and 200 held-out test rows.
3. **Models:** Compare ordinary Linear Regression with Ridge Regression at α = 0.1, 10 and 1000. Ridge minimises the sum of squared prediction errors plus α times the sum of squared coefficients; the intercept is not penalised.
4. **Cross-validation:** Evaluate each model on the same five folds of the 800-row training set. Each fit uses 640 rows for training and 160 for validation. Report the mean R² and standard deviation across the five validation scores.
5. **Extensions:** Examine individual fold scores, sweep α, inspect coefficient changes, compare preprocessing strategies and evaluate the fitted models on the held-out test set.

R² = 1 indicates perfect predictions. R² = 0 means the squared prediction error equals that of always predicting the evaluation subset’s mean target value; negative scores indicate worse performance than that baseline. R² is not the percentage of predictions that are correct.

**Preprocessing caveat:** The supplied template fits the scaler on all 1,000 rows before splitting, so the main results reproduce that procedure. This allows information about the validation and test feature distributions into preprocessing. The appropriate approach is to split the raw data first and place `StandardScaler` and the estimator in a `Pipeline`, fitting the scaler separately within each cross-validation training fold. Cell 7 compares the approaches.

## Results

### Task 2.2 — Five-fold cross-validation

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean CV R² | Std across folds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Linear Regression | 0.7494 | 0.6018 | 0.6308 | 0.6385 | 0.6080 | **0.6457** | 0.0600 |
| Ridge (α = 0.1) | 0.7494 | 0.6019 | 0.6310 | 0.6385 | 0.6079 | **0.6457** | 0.0599 |
| Ridge (α = 10) | 0.7442 | 0.6032 | 0.6419 | 0.6334 | 0.5989 | 0.6443 | 0.0588 |
| Ridge (α = 1000) | 0.4043 | 0.3384 | 0.3775 | 0.3686 | 0.2983 | 0.3574 | 0.0405 |

These values match the results recorded in the assignment report. Means and standard deviations are calculated from the underlying scores, rather than from the rounded values displayed in the table.

![Mean five-fold cross-validation R² for Linear Regression and Ridge Regression](model_comparison.png)

### Key findings

- **Weak regularisation provides no meaningful improvement in this comparison.** Ridge with α = 0.1 matches Linear Regression’s mean score to four decimal places. At α = 10, the mean decreases by 0.0014, with improvements in two folds and decreases in three. With 640 observations and eight features in each cross-validation fit, ordinary regression may already estimate coefficients reasonably reliably. These results do not by themselves establish whether it overfits; that would require comparing training and validation performance.
- **Heavy regularisation causes a substantial loss of predictive performance.** At α = 1000, mean CV R² falls from 0.6457 to 0.3574, with worse performance in every fold. The smaller fold standard deviation reflects more consistently poor predictions, not a better model.
- **The coefficient changes explain the loss.** At α = 1000, the standardised income coefficient decreases from 0.801 to 0.333, latitude from −0.905 to −0.089, and longitude from −0.842 to −0.054. Predictions therefore respond much less strongly to these differences between districts. The sum of absolute coefficients decreases from 3.328 to 0.770, approximately 77%; this is a descriptive measure of shrinkage, while Ridge itself penalises the sum of squared coefficients.
- **Cross-validation exposes variation hidden by a single score.** Linear Regression’s fold scores range from 0.6018 to 0.7494, a spread of 0.1476. Averaging across folds reduces dependence on one favourable or unfavourable partition, while the individual scores show variation between validation subsets. This observed range is not a bound on all possible train/test splits.

### Task 2.3 — Underfitting and overfitting

Ridge balances underfitting and overfitting by controlling how strongly coefficients are penalised. Overfitting occurs when a model fits sample-specific noise and achieves training performance that does not carry over to unseen data. Increasing α can reduce this sensitivity to the training sample, but excessive shrinkage prevents the model from capturing useful relationships and causes underfitting. Here, α = 1000 sharply reduces both coefficient magnitudes and validation performance, whereas the weaker penalties provide no meaningful improvement over ordinary Linear Regression. As α becomes extremely large, coefficients approach zero and predictions approach the training-set mean, making it difficult to distinguish cheaper from more expensive districts. The appropriate α should therefore be chosen using validation performance, with preprocessing fitted within each fold and the test set reserved for final evaluation.

## Extension experiments (Cells 4–8)

| Cell | Question | Finding |
|---|---|---|
| 4 | How much do individual folds vary? | Linear Regression fold scores span 0.1476 R², showing why both the average and fold-level results are useful. |
| 5 | Where is the best α among the tested values? | The sweep peaks near α ≈ 1.9 at CV R² ≈ 0.6459, a negligible gain over Linear Regression. Scores change little at small α and decline substantially at large α. |
| 6 | What does α do to the coefficients? | At α = 1000, the magnitudes of the latitude and longitude coefficients fall from approximately 0.9 to below 0.1. Individual coefficients need not shrink monotonically, and some can change sign. |
| 7 | Does fitting the scaler before splitting leak information? | Yes: preprocessing uses information from held-out feature distributions. Fitting the scaler within each fold changes CV R² by at most approximately 0.003 in this comparison, but remains the appropriate procedure. |
| 8 | Do the held-out test scores support the same broad pattern? | Reported test R² is approximately 0.662 for Linear Regression, 0.658 for Ridge at α = 10, and 0.401 for Ridge at α = 1000, consistent with the CV ordering. |

The extension results summarise the notebook experiments. Under the template’s preprocessing procedure, the test rows are held out from regression fitting, but not from scaler fitting; the resulting test scores are therefore not a fully isolated evaluation of the complete preprocessing-and-model workflow.

![Mean cross-validation R² across the Ridge alpha sweep](alpha_sweep.png)

![Standardised feature coefficient paths as Ridge alpha increases](coefficient_shrinkage.png)

## Changes made to the provided template

Copying the template from the assignment PDF disrupted the code’s formatting. The repairs restored executable Python without changing the intended dataset, model settings or evaluation procedure:

1. Rejoined the Ridge results `print(f"...")` statement, which had been split across lines inside a quoted string and caused `SyntaxError: unterminated f-string literal`.
2. Separated statements that had been merged onto the same line, including imports, feature/target assignments and plotting commands.
3. Restored indentation inside the model-training and bar-annotation loops, and separated executable statements from comments where line breaks had been lost.

I provided the error message and copied code to ChatGPT and asked it to explain the cause in simple terms. Its explanation helped distinguish formatting errors introduced during copying from problems with the regression method. The repaired code produced the dataset preview, model scores and chart used in the report.

Additional notebook conveniences include `pd.set_option` for wider printed tables, retaining per-fold results in `ridge_fold_scores`, and `plt.savefig(...)` for chart export. These additions do not change the template’s model calculations. Cells 4–8 add the extension analyses described above.

## Use of Generative AI

I used **ChatGPT (GPT 6 ASTRA-HIGH)** as a learning partner for the assignment. The prompts covered admissible and consistent heuristics, checking my 8-puzzle calculations, debugging the copied Python template, identifying the target and input features, explaining feature scaling, interpreting cross-validation scores, and understanding Ridge regularisation. The accompanying report records the prompts and my synthesis or reflection for each task.

The AI helped me connect the numerical results to the underlying concepts, particularly why increasing α can suppress useful predictive relationships and why an R² score is not a percentage accuracy. I also used its feedback to correct my description of the A* tie-breaking rule: my recorded selections follow lowest f, then lowest h, and finally FIFO, rather than FIFO alone. For the regression conclusions, I checked the explanations against the reported fold scores and coefficient table, and distinguished the recommendation to fit scaling within each fold from what the supplied template actually does.

Generative AI also assisted with editing and formatting this README using the assignment document and the supplied repository summary.

## Limitations

- **Sample size and coverage:** The experiment uses 1,000 of 20,640 rows. Scores describe this sample and evaluation setup, rather than performance on the entire dataset or on current housing markets.
- **Linear model assumptions:** The models use additive linear effects and cannot directly represent complex nonlinear or interacting relationships, including detailed geographic patterns. Comparing nonlinear models would be a separate experiment.
- **Target ceiling:** Values recorded at `5.00001` conceal variation above the upper recording limit. Scaling the features cannot recover that missing target information.
- **Preprocessing leakage:** The template fits the scaler before splitting. The small observed difference in Cell 7 does not justify this practice; a pipeline fitted within each fold is the appropriate evaluation workflow.
- **Model selection and uncertainty:** The small gain from the α sweep is insufficient to claim a meaningful advantage for Ridge. Fold-score standard deviations describe variation across these folds; they are not confidence intervals. Test scores should be used for final assessment after selection, rather than for further tuning.
