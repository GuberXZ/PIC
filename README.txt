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
               -> Best results with lr_type=constant, learning rate=0.5 and 300 max iter, with accuracy=0.7428571428571429 - EqFreqLimpoDum.