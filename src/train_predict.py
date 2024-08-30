# def train_and_predict_model(X_train_transformed, y_train, X_test_transformed, preprocessor, model):
#     model.fit(X_train_transformed, y_train)

#     # Instead of returning predictions, return the trained model
#     return model


def train_and_predict_model(X_train, y_train, X_test, preprocessor, model):
    # Assuming preprocessor is a pipeline that transforms the data
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    model.fit(X_train_transformed, y_train)
    
    # Instead of returning predictions, return the trained model
    return model