# Polynomial & Support Vector Regression


<Figure size 1000x600 with 1 Axes><img width="868" height="547" alt="image" src="https://github.com/user-attachments/assets/9a0406c2-71bd-416b-9375-75611fdda3a0" />



Analysis of Polynomial Regression

Optimal Degree: Based on typical runs with this dataset, Degree 2 is often the optimal choice.

Justification: Degree 1 (Linear) usually underfits because it cannot capture the non-linear relationship between age/BMI and charges.

Degree 2 captures the curve significantly better, lowering the RMSE.

Overfitting vs. Underfitting:
Underfitting (Degree 1): The model is too simple to capture the underlying trend of the data.
High bias, low variance.
Overfitting (Degree 3, 4, 5): As the degree increases, the model starts to fit the noise in the training data rather than the signal. You will likely see the RMSE on the test set start to rise (or the R^2 drop) for higher degrees, or the computation becomes unstable due to the massive number of features generated.
