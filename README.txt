Testes efetuados sobre um modelo de NeuralNetworks, provalemente será melhor verificar tudo sobre outros modelos.

Missing Values -> Best results with lr_type=constant, learning rate=0.1 and 5000 max iter, with accuracy=0.7142857142857143 - Limpo.
               -> Best results with lr_type=constant, learning rate=0.1 and 500 max iter, with accuracy=0.6756756756756757 - 0.
               -> Best results with lr_type=constant, learning rate=0.1 and 500 max iter, with accuracy=0.6756756756756757 - Freq.

Dummification -> Best results with lr_type=constant, learning rate=0.1 and 500 max iter, with accuracy=0.7142857142857143 - Limpo.

Scaling -> Provalemente terá que ser refeito após "outlier imputation", mas penso que MinMax é a melhor normalização por observação gráfica.
        -> Best results with lr_type=adaptive, learning rate=0.9 and 300 max iter, with accuracy=0.7714285714285715 - MinMaxLimpoDum.
        -> Best results with lr_type=constant, learning rate=0.5 and 2500 max iter, with accuracy=0.7714285714285715 - zscoreLimpoDum.

Balancing -> Best results with lr_type=constant, learning rate=0.1 and 750 max iter, with accuracy=0.6666666666666666 - UnderLimpoDum.
          -> Best results with lr_type=adaptive, learning rate=0.1 and 500 max iter, with accuracy=0.7959183673469388 - OverLimpoDum.
          -> Método de smote está a criar valores que supostamente deveriam ser binários.

Discretization -> Best results with lr_type=adaptive, learning rate=0.9 and 300 max iter, with accuracy=0.7714285714285715 - EqFreqLimpoDum.
               -> Best results with lr_type=constant, learning rate=0.5 and 300 max iter, with accuracy=0.7428571428571429 - EqWidLimpoDum.

Classification -> NaiveBayes - tem maior accuracy para o estimador multinomial (Gaussiano tem melhor recall, mas specificity horrível) - accuracy=0.69.
               -> KNN - overfit à database de treino (precisamos de mais dados) - Best results with 1 neighbors and manhattan - accuracy=0.76.
               -> Decision Trees - Best results achieved with entropy criteria, depth=10 and min_impurity_decrease=0.01 - accuracy,=0.88 -> overfitting.
               -> Random Forest - Best results with depth=10, 1.00 features and 200 estimators, with accuracy=0.90 -> overfitting.
               -> Neural Networks - Best results with lr_type=constant, learning rate=0.1 and 100 max iter, with accuracy=0.7755102040816326.
               -> Gradient Boosting - Best results with depth=10, learning rate=0.50 and 25 estimators, with accuracy=0.86 -> overfitting.