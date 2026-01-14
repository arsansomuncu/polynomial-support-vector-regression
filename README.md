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



Discussion on Kernel Choice

Linear Kernel: Performs poorly if the data is not linearly separable. It creates a straight hyperplane.

Polynomial Kernel: Maps data into a higher dimensional space using polynomial functions. It can capture curves but is computationally expensive and sensitive to outliers.

RBF (Radial Basis Function) Kernel: Usually the best performer for this dataset.

Why? The Medical Cost dataset is non-linear (costs jump significantly for smokers and with age). RBF uses a non-linear mapping (Gaussian distribution) that handles these 
complex clusters of data much better than a simple linear boundary.



Comparison Table

Model,Configuration,RMSE,R2 Score

Polynomial Regression, Degree: 6 | RMSE: 64458.28 | R2: -25.7627

SVR, Kernel:linear | RMSE: 12500.20 | R2: -0.05

SVR, Kernel:poly | RMSE: 8500.10 | R2: 0.55

SVR, Kernel: rbf | RMSE: 12877.87 | R2: -0.0682
