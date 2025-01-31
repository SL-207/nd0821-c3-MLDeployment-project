# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model performs binary classification of a person's income group.  Multiple versions of the model were trained for this problem using Support Vector Machine (SVM) and XGBoost but the final model was trained with XGBoost ensemble and yielded the best overall result and fbeta score.

## Intended Use
This model predicts whether a person's income is above $50k given census-relevant data.

## Training Data
The model was trained on data from the 1994 census database, containing census records of 32,562 individuals. Features in this dataset include census-relevant information like age, workclass, education, marital-status,occupation, relationship, and salary.

## Evaluation Data
The evaluation data was derived from a separate split of the census dataset, comprising 20% of the data.

## Metrics
The final model yielded a precision of 0.851, a recall of 0.740, and a fbeta score of 0.791.

## Ethical Considerations
This model is trained on data extracted from US 1994 census. Thus, it may be partially biased may not necessarily representative of all other present populations when predicting income.

## Caveats and Recommendations
While this model is accurate most the of the time, it may make misclassifications on occassion, so it is best to use it for cases where some error is permissible.
